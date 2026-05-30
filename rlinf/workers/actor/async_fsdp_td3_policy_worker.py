"""Async TD3 policy worker — background trajectory reception, non-blocking training."""

import asyncio
import queue
import threading

import torch

from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.workers.actor.fsdp_td3_policy_worker import EmbodiedTD3FSDPPolicy


class AsyncEmbodiedTD3FSDPPolicy(EmbodiedTD3FSDPPolicy):
    should_stop = False

    async def recv_rollout_trajectories(self, input_channel):
        if getattr(self, "_recv_queue", None) is None:
            self._recv_queue = queue.Queue()
        if (
            getattr(self, "_recv_rollout_thread", None) is None
            or not self._recv_rollout_thread.is_alive()
        ):
            self._recv_rollout_thread = threading.Thread(
                target=self._recv_rollout_thread_main,
                args=(input_channel,),
                daemon=True,
            )
            self._recv_rollout_thread.start()

    def _recv_rollout_thread_main(self, input_channel):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        while not self.should_stop:
            for _ in range(split_num):
                trajectory = input_channel.get()
                self._recv_queue.put(trajectory)

    def _drain_received_trajectories(self, max_trajectories=None):
        if getattr(self, "_recv_queue", None) is None:
            return
        recv_list = []
        processed = 0
        while True:
            try:
                recv_list.append(self._recv_queue.get_nowait())
                processed += 1
                if max_trajectories is not None and processed >= max_trajectories:
                    break
            except queue.Empty:
                break
        if not recv_list:
            return
        self.replay_buffer.add_trajectories(recv_list)
        if self.demo_buffer is not None:
            intervene_list = []
            for traj in recv_list:
                trajs = traj.extract_intervene_traj()
                if trajs is not None:
                    intervene_list.extend(trajs)
            if intervene_list:
                self.demo_buffer.add_trajectories(intervene_list)

    async def _wait_for_replay_buffer_ready(self, min_buffer_size):
        while True:
            self._drain_received_trajectories(
                max_trajectories=self.cfg.actor.get("recv_drain_max_trajectories", 256)
            )
            if await self.replay_buffer.is_ready_async(min_buffer_size):
                return
            await asyncio.sleep(1)

    @Worker.timer("run_training")
    async def run_training(self):
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        await self._wait_for_replay_buffer_ready(min_buffer_size)

        torch.distributed.barrier()

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}
        for _ in range(self.cfg.algorithm.get("update_epoch", 1)):
            await asyncio.sleep(0)
            append_to_dict(metrics, self.update_one_epoch())
            self.update_step += 1

        mean_metrics = self.process_train_metrics(metrics)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metrics

    async def stop(self):
        self.should_stop = True
        self.buffer_dataset.close()
        recv_thread = getattr(self, "_recv_rollout_thread", None)
        if recv_thread is not None and recv_thread.is_alive():
            await asyncio.to_thread(recv_thread.join, 5)
