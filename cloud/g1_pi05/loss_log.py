# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Publish one cumulative CSV and a two-panel plot for a cloud training run."""

import csv
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from assets_io import oss

STAGES = ("value", "policy")
INTERVAL_SECONDS = 3600


def stage_history(job, stage, limit=None):
    """Read only complete update records, including when a training step is appending."""
    path = job / f"{stage}-loss.csv"
    losses = []
    if path.is_file():
        with path.open() as src:
            for number, line in enumerate(src, 1):
                if not line.endswith("\n"):
                    break
                step, loss = line.rstrip("\n").split(",")
                if int(step) != number:
                    raise ValueError(f"Discontinuous {stage} loss at {number}")
                float(loss)
                losses.append(loss)
                if number == limit:
                    break
    if limit is not None and len(losses) != limit:
        raise ValueError(f"{stage} checkpoint {limit} has {len(losses)} loss records")
    return losses


def restore_history(job, stage, losses):
    # Rewind the local curve to the same optimizer step as the restored checkpoint.
    with tempfile.NamedTemporaryFile(mode="w", dir=job, delete=False) as dest:
        for step, loss in enumerate(losses, 1):
            dest.write(f"{step},{float(loss)}\n")
    os.replace(dest.name, job / f"{stage}-loss.csv")


def publish(job, uri):
    stages = ("policy",) if os.getenv("JOB_MODE") == "policy" else STAGES
    history = {stage: stage_history(job, stage) for stage in stages}
    if not any(history.values()):
        return
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    with tempfile.TemporaryDirectory(dir=job) as tmp:
        csv_path = Path(tmp) / "loss.csv"
        image_path = Path(tmp) / "loss.png"
        fig = Figure(figsize=(9, 6), layout="constrained")
        FigureCanvasAgg(fig)
        with csv_path.open("w", newline="") as dest:
            writer = csv.writer(dest)
            writer.writerow(("stage", "step", "loss"))
            axes = fig.subplots(len(stages), 1, squeeze=False)[:, 0]
            for (stage, losses), ax in zip(history.items(), axes):
                steps = range(1, len(losses) + 1)
                writer.writerows(
                    (stage, step, loss) for step, loss in zip(steps, losses)
                )
                ax.set(
                    title=f"{stage.capitalize()} training loss",
                    xlabel="Training step",
                    ylabel="Loss",
                )
                ax.grid(alpha=0.22)
                if losses:
                    ax.plot(
                        steps,
                        [float(v) for v in losses],
                        color="#236081",
                        marker="o" if len(losses) == 1 else None,
                    )
        fig.savefig(image_path, dpi=130)
        for file in (csv_path, image_path):
            oss("cp", file, uri + "/" + file.name, "-f")


@contextmanager
def hourly_updates(job, uri, interval=INTERVAL_SECONDS):
    # Checkpoint uploads may take longer than an hour. Keep the wall clock independent.
    stop = threading.Event()

    def watch():
        deadline = time.monotonic() + interval
        while not stop.wait(max(0, deadline - time.monotonic())):
            try:
                publish(job, uri)
            except Exception:
                logging.exception("Hourly loss upload failed")
            deadline += interval
            while deadline <= time.monotonic():
                deadline += interval

    worker = threading.Thread(target=watch, name="hourly-loss", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
