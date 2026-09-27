"""Episode-local observation sequences for the SmolVLA memory policy."""

import logging
import re
from bisect import bisect_right
from typing import Any

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.sampler import EpisodeAwareSampler

SEQUENCE_IS_PAD = "sequence_is_pad"


class EpisodeSequenceDataset(Dataset):
    """Keep frame indices as sequence anchors so the existing sampler can be reused.

    Observations are strided; each underlying frame still supplies its original
    continuous action chunk and action padding mask. Short episode tails repeat
    the final frame and are excluded through ``sequence_is_pad``.
    """

    def __init__(self, dataset, sequence_length: int, observation_stride: int):
        if sequence_length < 1 or observation_stride < 1:
            raise ValueError("sequence_length and observation_stride must be positive")
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.observation_stride = observation_stride
        episodes = dataset.meta.episodes
        selected = dataset.episodes
        if selected is None:
            selected = range(len(episodes["dataset_from_index"]))
        mapping = dataset.absolute_to_relative_idx
        boundaries = []
        for episode in selected:
            start = int(episodes["dataset_from_index"][episode])
            end = int(episodes["dataset_to_index"][episode])
            if mapping is not None:
                start, end = mapping[start], mapping[end - 1] + 1
            boundaries.append((start, end))
        boundaries.sort()
        self._starts = [start for start, _ in boundaries]
        self._ends = [end for _, end in boundaries]

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        end = self._ends[bisect_right(self._starts, index) - 1]
        indices = [index + step * self.observation_stride for step in range(self.sequence_length)]
        frames = [self.dataset[min(frame_index, end - 1)] for frame_index in indices]
        sequence = {}
        for key in frames[0]:
            values = [frame[key] for frame in frames]
            sequence[key] = default_collate(values) if isinstance(values[0], torch.Tensor) else values
        sequence[SEQUENCE_IS_PAD] = torch.tensor([frame_index >= end for frame_index in indices])
        return sequence

    def __getitems__(self, indices):
        # DataLoader otherwise discovers the wrapped dataset's frame-only bulk loader.
        return [self[index] for index in indices]


class TaskEpisodeSampler(EpisodeAwareSampler):
    """Uniform environment task, uniform episode, then a valid window start.

    Short episodes start at their first frame. Sampling uses replacement, with
    the same number of draws per epoch and resume state as the frame sampler.
    """

    def __init__(self, dataset: EpisodeSequenceDataset, chunk_size: int, seed: int = 0):
        episodes = dataset.meta.episodes
        if "source_prefix" not in episodes.column_names:
            raise ValueError("TaskEpisodeSampler requires 'source_prefix' in episode metadata.")
        super().__init__(
            episodes["dataset_from_index"],
            episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            shuffle=True,
            seed=seed,
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
        self.window_frames = (dataset.sequence_length - 1) * dataset.observation_stride + chunk_size
        selected = dataset.episodes
        if selected is None:
            selected = range(len(episodes["dataset_from_index"]))
        task_episodes: dict[str, list[tuple[int, int]]] = {}
        source_prefixes = episodes["source_prefix"]
        for episode in selected:
            first = int(episodes["dataset_from_index"][episode])
            end = int(episodes["dataset_to_index"][episode])
            if end <= first:
                continue
            # RoboCasa prefixes are pretrain/atomic/CloseFridge/<date>, etc.
            prefix = source_prefixes[episode]
            match = re.search(r"(?:^|/)(?:atomic|composite)/([^/]+)(?:/|$)", str(prefix))
            if match is None:
                raise ValueError(f"Cannot extract environment task from source_prefix={prefix!r}.")
            task = match.group(1)
            task_episodes.setdefault(task, []).append((first, end))
        if not task_episodes:
            raise ValueError("TaskEpisodeSampler found no non-empty episodes.")
        task_names = sorted(task_episodes)
        self._task_episodes = [task_episodes[task] for task in task_names]
        logging.info(
            "TaskEpisodeSampler: %d environment tasks; episodes per task: %s",
            len(task_names),
            {task: len(task_episodes[task]) for task in task_names},
        )

    def _iter_epoch(self, epoch: int, start: int):
        generator = self._epoch_generator(epoch)
        # Bounded random blocks avoid storing an entire epoch's draws. Replay
        # skipped blocks too, so resuming reproduces the same global stream.
        for offset in range(0, self._num_frames, 4096):
            count = min(4096, self._num_frames - offset)
            choices = torch.rand(count, 3, generator=generator, dtype=torch.float64)
            if offset + count <= start:
                continue
            for task_draw, episode_draw, frame_draw in choices[max(0, start - offset) :].tolist():
                episodes = self._task_episodes[int(task_draw * len(self._task_episodes))]
                first, end = episodes[int(episode_draw * len(episodes))]
                start_count = max(1, end - first - self.window_frames + 1)
                index = first + int(frame_draw * start_count)
                if self._absolute_to_relative is not None:
                    index = self._absolute_to_relative[index]
                yield index


def collate_episode_sequences(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors as B,T,... and keep language/task entries in B*T order."""
    return {
        key: (
            default_collate([sample[key] for sample in samples])
            if isinstance(samples[0][key], torch.Tensor)
            else [value for sample in samples for value in sample[key]]
        )
        for key in samples[0]
    }


def flatten_episode_sequence_batch(batch: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    """Expose B*T frames to existing processors, retaining B,T for the policy."""
    mask = batch[SEQUENCE_IS_PAD]
    if mask.ndim != 2:
        raise ValueError("sequence_is_pad must have shape [batch, sequence_length]")
    flattened = {}
    for key, value in batch.items():
        if key == SEQUENCE_IS_PAD:
            continue
        if isinstance(value, torch.Tensor):
            if value.shape[:2] != mask.shape:
                raise ValueError(f"Sequence tensor {key!r} must start with shape {tuple(mask.shape)}")
            value = value.flatten(0, 1)
        flattened[key] = value
    return flattened, mask
