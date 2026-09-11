"""Read RoboCasa's LeRobot-v2.1 files without importing LeRobot v2.1.

The current LeRobot checkout uses the v3 metadata format, while RoboCasa365
ships per-task v2.1 directories.  This adapter reads the original JSONL,
Parquet, and MP4 files directly and exposes the small map-style dataset surface
that the v3 trainer consumes.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lerobot.datasets.video_utils import decode_video_frames
from lerobot.utils.constants import ACTION, DEFAULT_FEATURES, IMAGENET_STATS, OBS_STATE


CAMERA_KEYS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot0_agentview_right",
)


@dataclass(frozen=True)
class _VectorSlice:
    """A slice of one raw vector column described by modality.json."""

    columns: tuple[str, ...]
    start: int
    end: int


@dataclass(frozen=True)
class _Source:
    root: Path
    info: dict[str, Any]
    stats: dict[str, Any]
    tasks: dict[int, str]
    episode_tasks: dict[int, int | str | None]
    episodes: tuple[tuple[int, int], ...]
    state_slices: tuple[_VectorSlice, ...]
    action_slices: tuple[_VectorSlice, ...]
    video_keys: dict[str, str]
    chunk_size: int
    fps: float


@dataclass(frozen=True)
class _Episode:
    source_index: int
    episode_index: int
    length: int
    start: int
    stop: int


@dataclass
class RoboCasaDatasetMeta:
    """Subset of LeRobot v3 metadata consumed by the training pipeline."""

    features: dict[str, dict[str, Any]]
    stats: dict[str, dict[str, torch.Tensor]]
    camera_keys: list[str]
    episodes: dict[str, list[Any]]
    fps: float
    depth_keys: set[str]
    has_language_columns: bool = False


class RoboCasaDataset(Dataset[dict[str, Any]]):
    """A virtual multi-task dataset over raw RoboCasa v2.1 task directories.

    ``root`` may be a task directory itself or a parent containing many task
    directories. Every discovered ``meta/info.json`` becomes one source, and
    all of their episodes are presented as one frame-indexed PyTorch dataset.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        action_delta_indices: Sequence[int],
        observation_delta_indices: Sequence[int],
        image_transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
        video_backend: str = "torchcodec",
        tolerance_s: float = 1e-4,
        parquet_cache_size: int = 2,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.action_delta_indices = tuple(int(index) for index in action_delta_indices)
        self.observation_delta_indices = tuple(int(index) for index in observation_delta_indices)
        self.image_transforms = image_transforms
        self.video_backend = video_backend
        self.tolerance_s = tolerance_s
        self.parquet_cache_size = parquet_cache_size
        self._parquet_cache: OrderedDict[tuple[int, int], pd.DataFrame] = OrderedDict()

        if not self.action_delta_indices:
            raise ValueError("RoboCasa training requires at least one action delta index.")
        if self.observation_delta_indices != (0,):
            raise ValueError(
                "This initial RoboCasa loader supports only observation_delta_indices=[0]. "
                "Add temporal observation support before setting n_obs_steps > 1."
            )
        if parquet_cache_size <= 0:
            raise ValueError(f"parquet_cache_size must be positive, got {parquet_cache_size}.")

        self._sources = self._discover_sources()
        self._episodes = self._build_episodes()
        self.meta = self._build_meta()
        self.episodes: list[int] | None = None
        self.absolute_to_relative_idx: dict[int, int] | None = None

    @property
    def num_frames(self) -> int:
        return self._episodes[-1].stop

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def fps(self) -> float:
        return self.meta.fps

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return self.meta.features

    def __len__(self) -> int:
        return self.num_frames

    def __getstate__(self) -> dict[str, Any]:
        """Do not copy cached Parquet tables into DataLoader worker processes."""
        state = self.__dict__.copy()
        state["_parquet_cache"] = OrderedDict()
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self._episode_for_index(index)
        local_index = index - episode.start
        source = self._sources[episode.source_index]
        table = self._episode_table(episode)

        if len(table) != episode.length:
            raise ValueError(
                f"{source.root}: episode {episode.episode_index} metadata says {episode.length} frames, "
                f"but its Parquet file contains {len(table)}."
            )

        timestamps = self._timestamps(table, source, episode)
        state = self._vector_at(table, source.state_slices, local_index, source, "state")
        action, action_is_pad = self._action_chunk(table, source, local_index)
        task_index, task = self._task(table, local_index, source, episode)

        item: dict[str, Any] = {
            OBS_STATE: torch.from_numpy(state),
            ACTION: torch.from_numpy(action),
            "action_is_pad": torch.from_numpy(action_is_pad),
            "task": task,
            "task_index": torch.tensor(task_index, dtype=torch.int64),
            "index": torch.tensor(index, dtype=torch.int64),
            "episode_index": torch.tensor(self._episodes.index(episode), dtype=torch.int64),
            "frame_index": torch.tensor(local_index, dtype=torch.int64),
            "timestamp": torch.tensor(timestamps[local_index], dtype=torch.float32),
        }

        for camera_key, video_key in source.video_keys.items():
            video_path = self._video_path(source, episode.episode_index, video_key)
            if not video_path.is_file():
                raise FileNotFoundError(f"RoboCasa video file not found: {video_path}")
            frame = decode_video_frames(
                video_path,
                [float(timestamps[local_index])],
                self.tolerance_s,
                backend=self.video_backend,
                return_uint8=True,
            ).squeeze(0)
            item[camera_key] = self.image_transforms(frame) if self.image_transforms else frame

        return item

    def _discover_sources(self) -> tuple[_Source, ...]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"RoboCasa dataset root does not exist: {self.root}")

        info_paths = [self.root / "meta" / "info.json"]
        if not info_paths[0].is_file():
            info_paths = sorted(self.root.rglob("meta/info.json"))
        if not info_paths:
            raise FileNotFoundError(f"No RoboCasa meta/info.json files were found below {self.root}.")

        sources = tuple(self._load_source(path.parent.parent) for path in info_paths)
        fps_values = {source.fps for source in sources}
        if len(fps_values) != 1:
            raise ValueError(f"All RoboCasa task sources must use one fps value, got {sorted(fps_values)}.")
        return sources

    def _load_source(self, root: Path) -> _Source:
        meta_root = root / "meta"
        info = _load_json(meta_root / "info.json")
        modality = _load_json(meta_root / "modality.json")
        stats = _load_json(meta_root / "stats.json")
        tasks = _load_tasks(meta_root / "tasks.jsonl")
        episode_records = _load_jsonl(meta_root / "episodes.jsonl")

        episodes: list[tuple[int, int]] = []
        episode_tasks: dict[int, int | str | None] = {}
        for record in episode_records:
            episode_index = int(record["episode_index"])
            length = int(record["length"])
            if length <= 0:
                continue
            episodes.append((episode_index, length))
            episode_tasks[episode_index] = _first_task(record)
        if not episodes:
            raise ValueError(f"{root}: no non-empty episodes in meta/episodes.jsonl.")

        video_keys = _video_keys(modality.get("video", {}), root)
        return _Source(
            root=root,
            info=info,
            stats=stats,
            tasks=tasks,
            episode_tasks=episode_tasks,
            episodes=tuple(sorted(episodes)),
            state_slices=_vector_slices(modality.get("state", {}), "state"),
            action_slices=_vector_slices(modality.get("action", {}), "action"),
            video_keys=video_keys,
            chunk_size=int(info.get("chunks_size", info.get("chunk_size", 1000))),
            fps=float(info["fps"]),
        )

    def _build_episodes(self) -> tuple[_Episode, ...]:
        episodes: list[_Episode] = []
        start = 0
        for source_index, source in enumerate(self._sources):
            for episode_index, length in source.episodes:
                episodes.append(_Episode(source_index, episode_index, length, start, start + length))
                start += length
        return tuple(episodes)

    def _build_meta(self) -> RoboCasaDatasetMeta:
        source = self._sources[0]
        state_dim = _vector_dimension(source.state_slices)
        action_dim = _vector_dimension(source.action_slices)
        features: dict[str, dict[str, Any]] = {
            **DEFAULT_FEATURES,
            OBS_STATE: {"dtype": "float32", "shape": (state_dim,), "names": None},
            ACTION: {"dtype": "float32", "shape": (action_dim,), "names": None},
        }
        for camera_key, video_key in source.video_keys.items():
            raw_feature = source.info.get("features", {}).get(video_key, {})
            shape = tuple(raw_feature.get("shape", (256, 256, 3)))
            names = raw_feature.get("names", ["height", "width", "channels"])
            features[camera_key] = {"dtype": "video", "shape": shape, "names": names}

        for other in self._sources[1:]:
            if _vector_dimension(other.state_slices) != state_dim:
                raise ValueError("RoboCasa sources disagree on observation.state dimensionality.")
            if _vector_dimension(other.action_slices) != action_dim:
                raise ValueError("RoboCasa sources disagree on action dimensionality.")
            if tuple(other.video_keys) != tuple(source.video_keys):
                raise ValueError("RoboCasa sources do not expose the same three camera keys.")

        episode_starts = [episode.start for episode in self._episodes]
        episode_stops = [episode.stop for episode in self._episodes]
        episode_tasks = [
            [task_index]
            if (task_index := _task_index_from_value(
                self._sources[episode.source_index].tasks,
                self._sources[episode.source_index].episode_tasks[episode.episode_index],
            )) >= 0
            else []
            for episode in self._episodes
        ]
        return RoboCasaDatasetMeta(
            features=features,
            stats={
                OBS_STATE: self._aggregate_vector_stats("state"),
                ACTION: self._aggregate_vector_stats("action"),
                **{
                    camera: {name: torch.tensor(value, dtype=torch.float32) for name, value in IMAGENET_STATS.items()}
                    for camera in CAMERA_KEYS
                },
            },
            camera_keys=list(CAMERA_KEYS),
            episodes={
                "dataset_from_index": episode_starts,
                "dataset_to_index": episode_stops,
                "tasks": episode_tasks,
            },
            fps=source.fps,
            depth_keys=set(),
        )

    def _aggregate_vector_stats(self, modality: str) -> dict[str, torch.Tensor]:
        per_source = [_sliced_stats(source, modality) for source in self._sources]
        weights = np.asarray([sum(length for _, length in source.episodes) for source in self._sources], dtype=np.float64)
        means = np.stack([stat["mean"] for stat in per_source])
        stds = np.stack([stat["std"] for stat in per_source])
        mean = np.average(means, axis=0, weights=weights)
        variance = np.average(stds**2 + means**2, axis=0, weights=weights) - mean**2
        return {
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(np.sqrt(np.maximum(variance, 0.0)), dtype=torch.float32),
            "min": torch.tensor(np.min(np.stack([stat["min"] for stat in per_source]), axis=0), dtype=torch.float32),
            "max": torch.tensor(np.max(np.stack([stat["max"] for stat in per_source]), axis=0), dtype=torch.float32),
        }

    def _episode_for_index(self, index: int) -> _Episode:
        if not 0 <= index < self.num_frames:
            raise IndexError(f"Frame index {index} is outside [0, {self.num_frames}).")
        stops = [episode.stop for episode in self._episodes]
        return self._episodes[int(np.searchsorted(stops, index, side="right"))]

    def _episode_table(self, episode: _Episode) -> pd.DataFrame:
        cache_key = (episode.source_index, episode.episode_index)
        cached = self._parquet_cache.pop(cache_key, None)
        if cached is not None:
            self._parquet_cache[cache_key] = cached
            return cached

        source = self._sources[episode.source_index]
        path = self._data_path(source, episode.episode_index)
        if not path.is_file():
            raise FileNotFoundError(f"RoboCasa Parquet file not found: {path}")
        table = pd.read_parquet(path)
        self._parquet_cache[cache_key] = table
        while len(self._parquet_cache) > self.parquet_cache_size:
            self._parquet_cache.popitem(last=False)
        return table

    def _vector_at(
        self,
        table: pd.DataFrame,
        slices: Sequence[_VectorSlice],
        row: int,
        source: _Source,
        modality: str,
    ) -> np.ndarray:
        vector = _load_vector(table, slices, source.root, modality)
        return vector[row]

    def _action_chunk(self, table: pd.DataFrame, source: _Source, local_index: int) -> tuple[np.ndarray, np.ndarray]:
        actions = _load_vector(table, source.action_slices, source.root, "action")
        offsets = np.asarray(self.action_delta_indices, dtype=np.int64)
        indices = local_index + offsets
        is_pad = (indices < 0) | (indices >= len(actions))
        output = np.zeros((len(offsets), actions.shape[1]), dtype=np.float32)
        valid = ~is_pad
        output[valid] = actions[indices[valid]]
        return output, is_pad.astype(np.bool_)

    def _timestamps(self, table: pd.DataFrame, source: _Source, episode: _Episode) -> np.ndarray:
        if "timestamp" not in table:
            raise KeyError(f"{source.root}: episode {episode.episode_index} has no timestamp column.")
        return table["timestamp"].to_numpy(dtype=np.float64)

    def _task(
        self,
        table: pd.DataFrame,
        local_index: int,
        source: _Source,
        episode: _Episode,
    ) -> tuple[int, str]:
        if "task_index" in table:
            task_index = int(table.iloc[local_index]["task_index"])
            return task_index, source.tasks.get(task_index, source.root.name)

        task = source.episode_tasks[episode.episode_index]
        task_index = _task_index_from_value(source.tasks, task)
        if isinstance(task, str):
            return task_index, task
        return task_index, source.tasks.get(task_index, source.root.name)

    @staticmethod
    def _path(source: _Source, pattern_name: str, episode_index: int, video_key: str | None = None) -> Path:
        pattern = source.info.get(pattern_name)
        if not isinstance(pattern, str):
            raise KeyError(f"{source.root}: meta/info.json is missing {pattern_name!r}.")
        values = {
            "episode_chunk": episode_index // source.chunk_size,
            "episode_index": episode_index,
            "video_key": video_key,
        }
        try:
            return source.root / pattern.format(**values)
        except KeyError as error:
            raise ValueError(f"{source.root}: unsupported {pattern_name} template {pattern!r}.") from error

    def _data_path(self, source: _Source, episode_index: int) -> Path:
        return self._path(source, "data_path", episode_index)

    def _video_path(self, source: _Source, episode_index: int, video_key: str) -> Path:
        return self._path(source, "video_path", episode_index, video_key)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required RoboCasa metadata file not found: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required RoboCasa metadata file not found: {path}")
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _load_tasks(path: Path) -> dict[int, str]:
    tasks = _load_jsonl(path)
    return {int(record["task_index"]): str(record["task"]) for record in tasks}


def _first_task(record: dict[str, Any]) -> int | str | None:
    value = record.get("task_index", record.get("tasks"))
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None or isinstance(value, str):
        return value
    return int(value)


def _task_index_from_value(tasks: dict[int, str], value: int | str | None) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return next((index for index, task in tasks.items() if task == value), -1)
    return -1


def _vector_slices(metadata: dict[str, Any], modality: str) -> tuple[_VectorSlice, ...]:
    if not metadata:
        raise ValueError(f"RoboCasa modality.json has no {modality!r} metadata.")
    slices = []
    default_column = OBS_STATE if modality == "state" else ACTION
    for name, spec in metadata.items():
        if not isinstance(spec, dict) or "start" not in spec or "end" not in spec:
            raise ValueError(f"Invalid {modality} modality entry {name!r}.")
        original = spec.get("original_key")
        candidates = tuple(dict.fromkeys(filter(None, (original, name, f"{modality}.{name}", default_column))))
        slices.append(_VectorSlice(candidates, int(spec["start"]), int(spec["end"])))
    return tuple(sorted(slices, key=lambda item: item.start))


def _video_keys(metadata: dict[str, Any], root: Path) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for camera in CAMERA_KEYS:
        for name, spec in metadata.items():
            original = spec.get("original_key") if isinstance(spec, dict) else None
            candidates = {name, original, f"observation.images.{name}"}
            if camera in candidates:
                resolved[camera] = original or name
                break
        else:
            raise ValueError(f"{root}: modality.json does not define required camera {camera!r}.")
    return resolved


def _vector_dimension(slices: Iterable[_VectorSlice]) -> int:
    return max(vector_slice.end for vector_slice in slices)


def _load_vector(table: pd.DataFrame, slices: Sequence[_VectorSlice], root: Path, modality: str) -> np.ndarray:
    output = np.empty((len(table), _vector_dimension(slices)), dtype=np.float32)
    filled = np.zeros(output.shape[1], dtype=bool)
    for vector_slice in slices:
        column = next((candidate for candidate in vector_slice.columns if candidate in table), None)
        if column is None:
            raise KeyError(f"{root}: no Parquet column for {modality} slice {vector_slice.columns}.")
        values = np.stack(table[column].to_numpy()).astype(np.float32, copy=False)
        if values.ndim != 2 or values.shape[1] < vector_slice.end:
            raise ValueError(
                f"{root}: column {column!r} cannot supply {modality}[{vector_slice.start}:{vector_slice.end}]."
            )
        output[:, vector_slice.start : vector_slice.end] = values[:, vector_slice.start : vector_slice.end]
        filled[vector_slice.start : vector_slice.end] = True
    if not filled.all():
        raise ValueError(f"{root}: modality.json leaves gaps in the flattened {modality} vector.")
    return output


def _sliced_stats(source: _Source, modality: str) -> dict[str, np.ndarray]:
    slices = source.state_slices if modality == "state" else source.action_slices
    output: dict[str, np.ndarray] = {}
    for stat_name in ("mean", "std", "min", "max"):
        vector = np.empty(_vector_dimension(slices), dtype=np.float64)
        for vector_slice in slices:
            stat = None
            for column in vector_slice.columns:
                if column in source.stats and stat_name in source.stats[column]:
                    stat = np.asarray(source.stats[column][stat_name], dtype=np.float64)
                    break
            if stat is None or stat.ndim != 1 or len(stat) < vector_slice.end:
                raise ValueError(f"{source.root}: stats.json lacks {stat_name!r} for {modality}.")
            vector[vector_slice.start : vector_slice.end] = stat[vector_slice.start : vector_slice.end]
        output[stat_name] = vector
    return output
