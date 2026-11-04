# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plot actual RLT scalars from one RLinf run's TensorBoard directory."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_scalars(root: Path) -> dict[str, dict[int, tuple[float, float]]]:
    """Merge event files, retaining the latest wall-time value at each step."""
    values = {}
    directories = {p.parent for p in root.rglob("events.out.tfevents.*")}
    if (root / "tensorboard").is_dir():
        directories.add(root / "tensorboard")
    for directory in sorted(directories):
        events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        events.Reload()
        for tag in events.Tags()["scalars"]:
            if tag not in {
                "train/rlt_loss",
                "train/vla_loss",
                "train/loss",
                "train/lr",
                "train/learning_rate",
                "train/sac/actor_loss",
                "train/sac/critic_loss",
                "train/actor/bc_loss",
                "train/actor/bc_human_loss",
                "train/actor/bc_ref_loss",
                "train/actor/weighted_bc",
                "train/actor/weighted_q",
                "train/actor/weighted_bc_human",
                "train/actor/weighted_bc_policy",
                "train/actor/bc_excluded_mask_ratio",
                "train/actor/bc_human_weight",
                "train/actor/bc_policy_weight",
                "offline_eval/human_joint_mse_radians",
                "offline_eval/human_mse",
                "offline_eval/reference_human_mse",
                "offline_eval/all_mse",
                "offline_eval/reference_all_mse",
            }:
                continue
            target = values.setdefault(tag, {})
            for event in events.Scalars(tag):
                if not np.isfinite(event.value):
                    raise ValueError(f"Non-finite {tag} at logged step {event.step}")
                if event.wall_time >= target.get(event.step, (-np.inf, 0))[0]:
                    target[event.step] = (event.wall_time, event.value)
    if not values.get("train/rlt_loss") and not values.get("train/sac/critic_loss"):
        raise ValueError(f"No RLT training TensorBoard scalars under {root}")
    return values


def plot_actor_comparison(root: Path, output: Path) -> None:
    """Draw one comparison and per-run panels from saved logs and validation."""
    import torch
    from omegaconf import OmegaConf

    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for directory in root.iterdir():
        if not (directory / "actor_config.yaml").is_file():
            continue
        cfg = OmegaConf.load(directory / "actor_config.yaml")
        report = json.loads((directory / "report.json").read_text())
        validation = [
            json.loads(line)
            for line in (directory / "validation.jsonl").read_text().splitlines()
        ]
        values = read_scalars(Path(cfg.runner.logger.log_path))
        runs.append((directory.name, cfg, report, validation, values))
    if not runs:
        raise ValueError(f"No completed actor experiments under {root}")
    modes = {"base_residual": 0, "state_delta": 1, "absolute": 2}
    runs.sort(
        key=lambda r: (r[1].algorithm.bc_weight, modes[r[1].actor.model.g1_action_mode])
    )
    paths = {r[1].algorithm.offline_validation_path for r in runs}
    if len(paths) != 1:
        raise ValueError("Comparison requires the same validation cache")
    cache_path = Path(paths.pop())
    data = torch.load(cache_path, map_location="cpu", weights_only=True)
    norm_path = Path(runs[0][1].actor.model.g1_action_norm_stats_path)
    stats = json.loads(norm_path.read_text())["norm_stats"]["actions"]
    low = torch.tensor(stats["q01"][:14])
    scale = data["action_scale"]
    human = data["intervene_flags"].bool()
    target_delta = (data["actions"] + 1) * scale + low
    ref_delta_error = (data["curr_obs"]["ref_chunk"] - data["actions"]) * scale
    baseline = {
        "validation_cache": str(cache_path),
        "validation_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "chunks": len(data["actions"]),
        "human_joint_elements": int(human.sum()),
        "vla_human_mse_rad2": float(ref_delta_error[human].square().mean()),
        "hold_query_state_human_mse_rad2": float(target_delta[human].square().mean()),
        "hold_definition": "Repeat the query-start joint state for every H10 step",
    }
    colors = {
        "base_residual": "#188f83",
        "state_delta": "#d49328",
        "absolute": "#8870a7",
    }
    titles = {
        "base_residual": "Base action + correction",
        "state_delta": "State + delta",
        "absolute": "Absolute action",
    }
    critic_tag, actor_tag = "train/sac/critic_loss", "train/sac/actor_loss"
    bc_tags = [
        "train/actor/bc_loss",
        "train/actor/bc_human_loss",
        "train/actor/bc_ref_loss",
    ]

    def series(values, tag):
        items = sorted(values[tag].items())
        return np.array([p[0] for p in items]), np.array([p[1][1] for p in items])

    def draw_mean(ax, values, tag, window, color, label=None, style="-", raw=False):
        x, y = series(values, tag)
        if raw:
            ax.plot(x, y, color=color, alpha=0.15, linewidth=0.5)
        total = np.r_[0.0, np.cumsum(y)]
        ends = np.arange(1, len(y) + 1)
        starts = np.maximum(0, ends - window)
        mean = (total[ends] - total[starts]) / (ends - starts)
        ax.plot(x, mean, color=color, linestyle=style, linewidth=1.6, label=label)

    def layout(axes):
        for ax in axes.flat:
            ax.set_xlabel("Critic updates")
            ax.grid(alpha=0.16)
            ax.set_xlim(0, max(r[2]["critic_updates"] for r in runs))
        for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
            ax.set_yscale("log")
        axes[0, 1].set_yscale("symlog", linthresh=1e-4)
        axes[0, 0].set_ylabel("Critic loss")
        axes[0, 1].set_ylabel("Actor objective (symlog)")
        axes[1, 0].set_ylabel("Normalized action MSE")
        axes[1, 1].set_ylabel("Joint MSE (rad²)")

    def baselines(ax):
        ax.axhline(
            baseline["vla_human_mse_rad2"],
            color="#777777",
            linestyle="--",
            label="VLA reference",
            linewidth=1.4,
        )
        ax.axhline(
            baseline["hold_query_state_human_mse_rad2"],
            color="#5796ad",
            linestyle=":",
            label="Hold query state",
            linewidth=1.4,
        )

    def save(fig, stem):
        fig.subplots_adjust(
            left=0.065, right=0.985, top=0.86, bottom=0.105, hspace=0.30, wspace=0.16
        )
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{suffix}", dpi=170)
        plt.close(fig)

    manifest = {
        "baselines": baseline,
        "experiments": {},
        "source": "RLinf TensorBoard and validation.jsonl",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})
    overview, axes = plt.subplots(2, 2, figsize=(17, 10.7))
    for ax, title in zip(
        axes.flat,
        [
            "Critic loss (100 logged-point mean)",
            "Actor objective (100 logged-point mean)",
            "BC MSE (100 logged-point mean)",
            "Validation human-action MSE (unsmoothed)",
        ],
        strict=True,
    ):
        ax.set_title(title)
    for name, cfg, report, validation, values in runs:
        mode, bc, q = (
            cfg.actor.model.g1_action_mode,
            cfg.algorithm.bc_weight,
            cfg.algorithm.q_weight,
        )
        color, style = colors[mode], "-" if bc == 5 else "--"
        label = f"{mode.replace('base_residual', 'base').replace('state_delta', 'state')}_bc{bc}"
        split_bc = cfg.algorithm.get("bc_human_weight") is not None
        bc_description = f"BC={bc:g}"
        objective_title = f"Actor objective = {bc:g} BC − {q:g} Q"
        if split_bc:
            human_weight, policy_weight = (
                cfg.algorithm.bc_human_weight,
                cfg.algorithm.bc_policy_weight,
            )
            color, style = "#188f83", "-"
            label = f"state_H{human_weight:g}/P{policy_weight:g}"
            bc_description = f"BC human={human_weight:g}, success={policy_weight:g}"
            objective_title = f"Actor objective = {human_weight:g} BC human + {policy_weight:g} BC success − {q:g} Q"
        for ax, tag in [
            (axes[0, 0], critic_tag),
            (axes[0, 1], actor_tag),
            (axes[1, 0], bc_tags[0]),
        ]:
            draw_mean(ax, values, tag, 100, color, label, style)
        vx = np.array([r["step"] for r in validation])
        vy = np.array([r["human_joint_mse_radians"] for r in validation])
        axes[1, 1].plot(vx, vy, color=color, linestyle=style, linewidth=1.5)
        expected = report["best_evaluation"]["reference_human_joint_mse_radians"]
        if not np.isclose(expected, baseline["vla_human_mse_rad2"], rtol=1e-6):
            raise ValueError(f"Validation baseline mismatch: {name}")
        objective_error = max(
            abs(
                v[1]
                - (
                    values["train/actor/weighted_bc"][step][1]
                    - values["train/actor/weighted_q"][step][1]
                )
            )
            for step, v in values[actor_tag].items()
        )
        if objective_error > 1e-6:
            raise ValueError(
                f"Actor objective does not equal weighted BC minus Q: {name}"
            )
        fig, single = plt.subplots(2, 2, figsize=(17, 10.7))
        draw_mean(single[0, 0], values, critic_tag, 50, color, raw=True)
        draw_mean(single[0, 1], values, actor_tag, 50, color, raw=True)
        for tag, line_color, legend in zip(
            bc_tags,
            [color, "#d94e42", "#368bb4"],
            [
                "Eligible (batch mean)" if split_bc else "All",
                "Human target",
                "Successful policy" if split_bc else "Policy reference",
            ],
            strict=True,
        ):
            draw_mean(single[1, 0], values, tag, 50, line_color, legend, raw=True)
        single[1, 0].legend()
        single[1, 1].plot(
            vx,
            vy,
            "o-",
            color=color,
            linewidth=1.6,
            markersize=4,
            label="Validation human MSE",
        )
        best = next(r for r in validation if r["step"] == report["best_step"])
        single[1, 1].plot(
            best["step"],
            best["human_joint_mse_radians"],
            "*",
            color=color,
            markersize=16,
            label=f"Best @ critic {best['step']:,}",
        )
        baselines(single[1, 1])
        single[1, 1].legend(fontsize=10)
        for ax, title in zip(
            single.flat,
            [
                "Critic loss",
                objective_title,
                "BC MSE: all / human / policy",
                "Fixed five-episode validation",
            ],
            strict=True,
        ):
            ax.set_title(title)
        layout(single)
        fig.suptitle(
            f"{titles[mode]} | {bc_description}, Q={q:g} | {report['critic_updates']:,} critic / {report['actor_updates']:,} actor",
            fontsize=18,
            y=0.975,
        )
        fig.text(
            0.5,
            0.927,
            f"Frozen RLT {report['feature_step']:,} | 14 joints | H10 | batch {cfg.actor.global_batch_size} | LR {cfg.actor.optim.lr:g} | seed {cfg.actor.seed}",
            ha="center",
            color="#444444",
        )
        fig.text(
            0.5,
            0.033,
            "Real logged losses: faint = raw logged points; solid = 50-point trailing mean. Validation is unsmoothed; star = selected best.",
            ha="center",
            fontsize=10,
        )
        save(fig, name)
        with (output / f"{name}_scalars.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["critic_updates", "tag", "value"])
            for tag in [critic_tag, actor_tag, *bc_tags]:
                writer.writerows(
                    (step, tag, v[1]) for step, v in sorted(values[tag].items())
                )
        manifest["experiments"][name] = {
            "log_directory": cfg.runner.logger.log_path,
            "best_critic_step": best["step"],
            "points_per_tag": {
                t: len(values[t]) for t in [critic_tag, actor_tag, *bc_tags]
            },
            "objective_formula_max_abs_error": objective_error,
        }
    baselines(axes[1, 1])
    axes[0, 0].legend(ncol=2, fontsize=10)
    axes[1, 1].legend(fontsize=10)
    layout(axes)
    budgets = sorted({(r[2]["critic_updates"], r[2]["actor_updates"]) for r in runs})
    budget_label = " vs ".join(f"{critic:,} / {actor:,}" for critic, actor in budgets)
    overview.suptitle(
        f"{len(runs)} G1 actor experiments | Q=0.1 | Critic / actor budgets: {budget_label}",
        fontsize=18,
        y=0.975,
    )
    overview.text(
        0.5,
        0.927,
        "Frozen RLT 20,000 | 14 joints | H10 | batch 256 | critic:actor = 2:1 | seed 42",
        ha="center",
        color="#444444",
    )
    overview.text(
        0.5,
        0.033,
        "Means use 100 logged points (partial windows at the start). Validation: raw rad² on the same 255 held-out chunks; no early stopping.",
        ha="center",
        fontsize=10,
    )
    save(overview, "overview")
    if len(budgets) > 1:
        # A common update range prevents the longer run from hiding the matched budget.
        comparison, common_axes = plt.subplots(2, 2, figsize=(17, 10.7))
        limit = min(b[0] for b in budgets)
        for name, cfg, report, validation, values in runs:
            split = cfg.algorithm.get("bc_human_weight") is not None
            color = (
                "#188f83"
                if split
                else ("#d49328" if cfg.algorithm.bc_weight == 5 else "#8870a7")
            )
            label = (
                "Human 50 / success 5"
                if split
                else f"Uniform BC {cfg.algorithm.bc_weight:g}"
            )
            filtered = {
                tag: {step: value for step, value in scalars.items() if step <= limit}
                for tag, scalars in values.items()
            }
            for ax, tag in [
                (common_axes[0, 0], critic_tag),
                (common_axes[0, 1], actor_tag),
                (common_axes[1, 0], bc_tags[0]),
            ]:
                draw_mean(ax, filtered, tag, 100, color, label)
            selected = [v for v in validation if v["step"] <= limit]
            common_axes[1, 1].plot(
                [v["step"] for v in selected],
                [v["human_joint_mse_radians"] for v in selected],
                color=color,
                label=label,
            )
        layout(common_axes)
        for ax, title in zip(
            common_axes.flat,
            [
                "Critic loss",
                "Actor objective",
                "BC MSE (training masks differ)",
                "Validation human MSE (same mask)",
            ],
            strict=True,
        ):
            ax.set_xlim(0, limit)
            ax.set_title(title)
        baselines(common_axes[1, 1])
        common_axes[0, 0].legend()
        common_axes[1, 1].legend(fontsize=9)
        comparison.suptitle(
            f"State-delta actors | Matched first {limit:,} critic / {limit // 2:,} actor updates",
            fontsize=18,
            y=0.975,
        )
        comparison.text(
            0.5,
            0.927,
            "Frozen RLT 20,000 | 14 joints | H10 | Q=0.1 | same recorded validation",
            ha="center",
            color="#444444",
        )
        comparison.text(
            0.5,
            0.033,
            "100 logged-point means; unsmoothed validation. Split BC excludes autonomous failures; all samples remain in Q training.",
            ha="center",
            fontsize=10,
        )
        save(comparison, "matched_budget")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        help="Completed actor runs: comparison and four-panel figures",
    )
    parser.add_argument(
        "--run-dir", type=Path, help="Stage 1 validation and split directory"
    )
    parser.add_argument(
        "--config", type=Path, help="Stage 1 YAML for figure annotations"
    )
    parser.add_argument(
        "--actor-config", type=Path, help="Stage 2 YAML for figure annotations"
    )
    parser.add_argument(
        "--window", type=int, default=100, help="Logged points per mean"
    )
    args = parser.parse_args()
    if args.ablation_dir is not None:
        plot_actor_comparison(args.ablation_dir, args.output_dir)
        return
    if args.log_dir is None:
        parser.error("--log-dir is required unless --ablation-dir is supplied")
    if args.window < 1:
        parser.error("--window must be positive")
    if (args.run_dir is None) != (args.config is None):
        parser.error("--run-dir and --config must be supplied together")
    values = read_scalars(args.log_dir)
    tags = sorted(values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if "train/rlt_loss" not in values:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4), layout="constrained")
        for axis, names, title in zip(
            axes,
            [
                ["train/sac/critic_loss"],
                ["train/sac/actor_loss"],
                ["offline_eval/human_mse", "offline_eval/reference_human_mse"],
            ],
            ["Critic loss", "Actor loss", "Held-out intervention MSE"],
            strict=True,
        ):
            for tag in names:
                items = sorted(values.get(tag, {}).items())
                axis.plot(
                    [x[0] for x in items],
                    [x[1][1] for x in items],
                    label=tag.split("/")[-1],
                    linewidth=1,
                )
            axis.set_title(title)
            axis.set_xlabel("Critic updates")
            axis.grid(alpha=0.2)
            axis.legend(frameon=False)
        if args.actor_config:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(args.actor_config)
            model, algorithm = cfg.actor.model, cfg.algorithm
            default_mode = (
                "base_residual"
                if model.get("g1_action_norm_stats_path") is not None
                else "state_delta"
            )
            mode = model.get("g1_action_mode", default_mode)
            bc_caption = f"BC={algorithm.bc_weight:g}"
            if algorithm.get("bc_human_weight") is not None:
                bc_caption = f"BC human={algorithm.bc_human_weight:g} / success={algorithm.bc_policy_weight:g}"
            fig.suptitle(
                f"RLinf G1 | {mode} | {bc_caption}, Q={algorithm.q_weight:g}\n"
                f"14 joints | H={model.num_action_chunks} | batch={cfg.actor.global_batch_size} | "
                f"critic:actor={algorithm.critic_actor_ratio}:1 | seed={cfg.actor.seed}",
                weight="bold",
                fontsize=12,
            )
        else:
            fig.suptitle("RLinf G1 | Official RLT actor-critic", weight="bold")
        fig.savefig(args.output_dir / "actor_critic_loss.png", dpi=180)
        plt.close(fig)
        with (args.output_dir / "actor_critic_loss.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["critic_updates", *tags])
            for step in sorted(set().union(*(set(v) for v in values.values()))):
                writer.writerow(
                    [step, *[values[t].get(step, (0, ""))[1] for t in tags]]
                )
        return
    steps = sorted(values["train/rlt_loss"])
    # SFTRunner logs zero-based _step after completing the optimizer update.
    x = np.asarray(steps) + 1
    y = np.asarray([values["train/rlt_loss"][s][1] for s in steps])
    if np.any(y <= 0):
        raise ValueError("Log-scale reconstruction loss must be positive")
    window = min(args.window, len(y))
    means = np.convolve(y, np.ones(window) / window, mode="valid")
    mean_steps = x[window - 1 :]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rlt_loss.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["optimizer_step", *tags])
        for step in sorted(set().union(*(set(v) for v in values.values()))):
            writer.writerow(
                [step + 1, *[values[t].get(step, (0, ""))[1] for t in tags]]
            )

    context = None
    if args.run_dir is not None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.config)
        split = json.loads((args.run_dir / "episode_split.json").read_text())
        initial = json.loads(
            (args.run_dir / "stage1_initial_validation.json").read_text()
        )
        target = json.loads((args.run_dir / "training_target.json").read_text())
        validation = {0: float(initial["rlt_mse"])}
        for line in (args.run_dir / "stage1_validation.jsonl").read_text().splitlines():
            entry = json.loads(line)
            validation[int(entry["step"])] = float(entry["rlt_mse"])
        if any(not np.isfinite(v) or v <= 0 for v in validation.values()):
            raise ValueError("Validation MSE must be positive and finite")
        context = {
            "config": str(args.config.resolve()),
            "validation_mse": validation,
            "validation_samples": initial["samples"],
            "train_episodes": len(split["train"]),
            "validation_episodes": len(split["validation"]),
            "target_steps": target["maximum_steps"],
            "physical_gpus": target["physical_gpus"],
        }
        o = cfg.actor.optim
        m = cfg.actor.model.openpi
        rates = values.get("train/learning_rate") or values.get("train/lr")
        if rates:
            context["logged_lr_peak"] = max(v[1] for v in rates.values())
            context["logged_lr_final"] = rates[max(rates)][1]
            lr_label = f"LR logged {context['logged_lr_peak']:.2g} to {context['logged_lr_final']:.2g}"
        else:
            lr_label = f"LR configured {o.lr:g} / min {o.min_lr:g}"
        context["configured_lr"] = float(o.lr)
        context["configured_min_lr"] = float(o.min_lr)
        annotations = [
            (
                f"Frozen VLA: {m.freeze_vla} | alpha={m.rlt_alpha:g} | "
                f"VLA {cfg.actor.model.precision.upper()} / RLT "
                f"{cfg.actor.fsdp_config.mixed_precision_rules.rlt_features.upper()} | "
                f"{m.num_images_in_input} cameras | prefix {m.rlt_prefix_seq_len} | seed {cfg.actor.seed}"
            ),
            (
                f"Global batch {cfg.actor.global_batch_size} | micro batch {cfg.actor.micro_batch_size} | "
                f"physical GPU {context['physical_gpus']} | {lr_label} | "
                f"{o.lr_scheduler} | warmup {o.lr_warmup_steps:,}"
            ),
            (
                f"{context['train_episodes']:,} train / {context['validation_episodes']} validation episodes | "
                f"{context['validation_samples']} fixed validation frames | "
                f"target {context['target_steps']:,} steps"
            ),
        ]
        context["figure_annotations"] = annotations
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.14, top=0.68, wspace=0.22)
        for index, annotation in enumerate(annotations):
            fig.text(
                0.065, 0.9 - index * 0.038, annotation, fontsize=11, color="#444444"
            )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    fig.suptitle("RLinf | RL Token Reconstruction Loss", weight="bold")
    for ax in axes:
        ax.plot(x, y, color="#7fa9c5", alpha=0.5, linewidth=0.7, label="Logged loss")
        ax.plot(
            mean_steps,
            means,
            color="#007f89",
            linewidth=2,
            label=f"{window}-point mean",
        )
        ax.set_xlabel("Completed optimizer step")
        ax.set_ylabel("RLT reconstruction MSE")
        ax.grid(alpha=0.2)
        if context:
            val_steps = sorted(context["validation_mse"])
            ax.plot(
                val_steps,
                [validation[s] for s in val_steps],
                "o-",
                color="#c84832",
                linewidth=1.5,
                markersize=4,
                label="Fixed held-out validation",
            )
        else:
            ax.legend(frameon=False)
    axes[0].set_yscale("log")
    axes[0].set_title("Full training (log scale)")
    start = max(0, len(x) // 2 - 1)
    axes[1].set_xlim(float(x[start]) - 0.5, float(x[-1]) + 0.5)
    tail = np.concatenate([y[start:], means[mean_steps >= x[start]]])
    margin = max(float(np.ptp(tail)) * 0.1, float(tail.mean()) * 0.01)
    axes[1].set_ylim(float(tail.min()) - margin, float(tail.max()) + margin)
    axes[1].set_title("Second half (linear scale)")
    if context:
        val_tail = [v for s, v in validation.items() if s >= x[start]]
        if val_tail:
            lower, upper = axes[1].get_ylim()
            axes[1].set_ylim(
                min(lower, min(val_tail) - margin), max(upper, max(val_tail) + margin)
            )
        best_step = min(validation, key=validation.get)
        last_val_step = max(validation)
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper left",
            bbox_to_anchor=(0.065, 0.785),
            ncol=3,
            frameon=False,
            fontsize=11,
        )
        fig.text(
            0.065,
            0.055,
            f"Train {y[0]:.4f} to {y[-1]:.4f} | validation {validation[0]:.4f} to "
            f"{validation[last_val_step]:.4f} | best {validation[best_step]:.4f} @ {best_step:,} steps",
            fontsize=11,
            weight="bold",
        )
        fig.text(
            0.065,
            0.023,
            f"Source: RLinf TensorBoard + validation JSONL. Raw logged values and {window}-point mean; "
            "validation is unsmoothed. Reconstruction quality, not robot success.",
            fontsize=9,
            color="#555555",
        )
    for suffix in ["png", "pdf"]:
        fig.savefig(args.output_dir / f"rlt_loss.{suffix}", dpi=180)
    plt.close(fig)
    summary = {
        "source": str(args.log_dir.resolve()),
        "logged_points": len(y),
        "last_completed_step": int(x[-1]),
        "first_logged_rlt_loss": float(y[0]),
        "last_logged_rlt_loss": float(y[-1]),
        "last_window_mean": float(means[-1]),
        "window_points": window,
        "note": "Training loss only; this does not establish held-out or robot performance.",
    }
    if context:
        summary["run_context"] = context
    if len(y) >= 2 * window:
        previous = float(y[-2 * window : -window].mean())
        summary["previous_window_mean"] = previous
        summary["last_window_relative_improvement"] = (
            previous - float(means[-1])
        ) / previous
    (args.output_dir / "rlt_loss.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
