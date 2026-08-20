"""Suite-parallel LIBERO environments and PPO rollout storage."""

import random
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv, _get_suite, get_task_init_states
from torch import Tensor


Task = tuple[str, int]
IMAGE_PREFIX = "observation.images."


def pack_images(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Store model images as native uint8 pixels; leave every other input unchanged."""
    return {
        key: value.mul(255).round().clamp_(0, 255).to(torch.uint8)
        if key.startswith(IMAGE_PREFIX)
        else value
        for key, value in batch.items()
    }


def unpack_images(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Restore uint8 rollout images to SmolVLA's float32 [0, 1] input domain."""
    return {
        key: value.to(torch.float32).div(255) if key.startswith(IMAGE_PREFIX) else value
        for key, value in batch.items()
    }


def canonicalize_images(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Use exactly the same image representation during rollout and PPO replay."""
    return unpack_images(pack_images(batch))


def suite_tasks(suite: str) -> list[Task]:
    return [(suite, task_id) for task_id in range(len(_get_suite(suite).tasks))]


def horizon(suite: str) -> int:
    return TASK_SUITE_MAX_STEPS[suite]


def worker_counts(tasks: list[Task], workers: dict[str, int]) -> list[int]:
    """Resolve task, suite, then default worker counts."""
    return [workers.get(f"{suite}_{task}", workers.get(suite, workers["default"])) for suite, task in tasks]


def active_tasks(tasks: list[Task], workers: list[int]) -> tuple[list[Task], list[int]]:
    """Drop tasks disabled with a zero worker count."""
    if len(tasks) != len(workers):
        raise ValueError("Each task needs a worker count.")
    active = [(task, count) for task, count in zip(tasks, workers) if count]
    return [task for task, _ in active], [count for _, count in active]


def task_groups(tasks: list[Task], workers: list[int], limit: int) -> list[tuple[list[Task], list[int]]]:
    """Pack ordered tasks without exceeding the concurrent environment limit."""
    if limit < 1:
        raise ValueError("Concurrent environment limit must be positive.")
    groups, group_tasks, group_workers = [], [], []
    for task, count in zip(tasks, workers):
        if count > limit:
            raise ValueError(f"{task} needs {count} workers, above limit {limit}.")
        if group_workers and sum(group_workers) + count > limit:
            groups.append((group_tasks, group_workers))
            group_tasks, group_workers = [], []
        group_tasks.append(task)
        group_workers.append(count)
    return groups + [(group_tasks, group_workers)] if group_tasks else groups


class SuiteSchedule:
    """Return suites in order; tasks within each selected suite run together."""

    def __init__(self, suites: list[str], per_update: int):
        if not 1 <= per_update <= len(suites):
            raise ValueError("per_update must be between 1 and the number of suites.")
        self.suites, self.per_update, self.index = suites, per_update, 0

    def next(self) -> list[str]:
        selected = [self.suites[(self.index + offset) % len(self.suites)] for offset in range(self.per_update)]
        self.index = (self.index + self.per_update) % len(self.suites)
        return selected


def _env(task: Task, episode_index: int, workers_per_task: int, **kwargs):
    suite, task_id = task
    return LiberoEnv(
        task_suite=_get_suite(suite),
        task_suite_name=suite,
        task_id=task_id,
        episode_index=episode_index,
        n_envs=workers_per_task,
        **kwargs,
    )


def make_env(tasks: list[Task], workers: list[int], **kwargs):
    """Create one heterogeneous vector environment for a single LIBERO suite."""
    if len(tasks) != len(workers) or any(count < 1 for count in workers):
        raise ValueError("Each task needs one or more workers.")

    fns = []
    for task, count in zip(tasks, workers):
        suite, task_id = task
        if kwargs.get("init_states", True):
            max_init_states = len(
                get_task_init_states(
                    _get_suite(suite),
                    task_id,
                    is_libero_plus=kwargs.get("is_libero_plus", False),
                )
            )
            episode_indices = [random.randrange(max_init_states) for _ in range(count)]
        else:
            episode_indices = list(range(count))

        fns.extend(
            partial(_env, task, episode_index, count, **kwargs)
            for episode_index in episode_indices
        )
    return AsyncVectorEnv(
        fns,
        context="forkserver",
        shared_memory=True,
        # Reset terminated workers explicitly at action-chunk boundaries, after
        # all chunk actions have been executed.
        autoreset_mode=AutoresetMode.DISABLED,
    )


@dataclass
class Transition:
    task: int
    worker: int
    batch: dict[str, Tensor]
    path: Tensor
    old_log_prob: Tensor
    old_value: Tensor
    noise_std: float
    reward: float
    done: bool
    steps: int


@dataclass
class TaskResult:
    task: Task
    successes: int = 0
    failures: int = 0

    @property
    def episodes(self) -> int:
        return self.successes + self.failures

    @property
    def label(self) -> str:
        return f"{self.task[0]}_{self.task[1]} | success_rate={self.successes}/{self.episodes}"


class RolloutBuffer:
    """One fresh PPO batch and its task-local bootstrap contexts."""

    def __init__(self):
        self.items: list[Transition] = []
        self.bootstrap: dict[int, Tensor] = {}

    def extend(self, other: "RolloutBuffer") -> None:
        self.items.extend(other.items)
        self.bootstrap.update(other.bootstrap)

    def tensors(self, device: torch.device) -> dict[str, Tensor]:
        return {
            "path": torch.cat([item.path for item in self.items]).to(device),
            "old_log_prob": torch.cat([item.old_log_prob for item in self.items]).to(device),
            "old_value": torch.cat([item.old_value for item in self.items]).to(device),
            "noise_std": torch.tensor([item.noise_std for item in self.items], device=device),
            "reward": torch.tensor([item.reward for item in self.items], device=device),
            "done": torch.tensor([item.done for item in self.items], device=device),
            "steps": torch.tensor([item.steps for item in self.items], device=device),
            "task": torch.tensor([item.task for item in self.items], device=device),
            "worker": torch.tensor([item.worker for item in self.items], device=device),
        }


def merge(batches: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Join PPO minibatches while padding dynamic language lengths."""
    result = {}
    for key in batches[0]:
        values = [batch[key] for batch in batches]
        shape = [max(value.shape[index] for value in values) for index in range(1, values[0].ndim)]
        padded = []
        for value in values:
            padding = [
                item
                for index in range(value.ndim - 1, 0, -1)
                for item in (0, shape[index - 1] - value.shape[index])
            ]
            padded.append(F.pad(value, padding))
        result[key] = torch.cat(padded)
    return result
