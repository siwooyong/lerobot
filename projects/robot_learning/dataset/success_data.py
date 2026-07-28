"""Canonical LIBERO frame conversion and success-only episode recording."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_SHAPE = (256, 256, 3)
STATE_SHAPE = (8,)
ACTION_SHAPE = (7,)


class WritableLeRobotDataset(Protocol):
    """The LeRobotDataset writer surface used by this project."""

    def add_frame(self, frame: dict[str, Any]) -> None: ...

    def save_episode(self) -> None: ...

    def clear_episode_buffer(self, delete_images: bool = True) -> None: ...


def libero_dataset_features() -> dict[str, dict[str, Any]]:
    """Return the user-defined feature schema of ``HuggingFaceVLA/libero``."""

    return {
        IMAGE_KEYS[0]: {
            "dtype": "image",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        },
        IMAGE_KEYS[1]: {
            "dtype": "image",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": STATE_SHAPE,
            "names": ["state"],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": ACTION_SHAPE,
            "names": ["actions"],
        },
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _image_to_hwc_uint8(value: Any, env_index: int) -> np.ndarray:
    image = _to_numpy(value)
    if image.ndim != 4:
        raise ValueError(f"Expected batched image BCHW or BHWC, got shape {image.shape}.")

    image = image[env_index]
    if image.shape == (3, 256, 256):
        image = np.transpose(image, (1, 2, 0))
    elif image.shape != IMAGE_SHAPE:
        raise ValueError(
            f"Expected one LIBERO image with shape (3, 256, 256) or {IMAGE_SHAPE}, got {image.shape}."
        )

    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("Observation image contains NaN or Inf.")
        if image.min() < 0.0 or image.max() > 1.0:
            raise ValueError(
                f"Float observation image must be in [0, 1], got [{image.min()}, {image.max()}]."
            )
        image = np.rint(image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise TypeError(f"Observation image must be float or uint8, got {image.dtype}.")

    return np.ascontiguousarray(image)


def _batched_vector(value: Any, env_index: int, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == len(shape):
        array = array[None, ...]
    if array.ndim != len(shape) + 1:
        raise ValueError(f"{name} must be batched with shape (B, {shape[0]}), got {array.shape}.")
    vector = np.asarray(array[env_index], dtype=np.float32)
    if vector.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {vector.shape}.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return vector


def canonical_frame(
    observation: dict[str, Any],
    action: Any,
    *,
    env_index: int,
    task: str,
) -> dict[str, Any]:
    """Convert a post-``LiberoProcessorStep`` observation into one dataset frame.

    The policy preprocessor is intentionally excluded: resize, padding,
    normalization, and device transfer belong to model input processing, not
    persisted demonstration data.
    """

    missing = [key for key in (*IMAGE_KEYS, STATE_KEY) if key not in observation]
    if missing:
        raise KeyError(f"Canonical LIBERO observation is missing {missing}.")
    if not task:
        raise ValueError("A non-empty natural-language task is required.")

    return {
        IMAGE_KEYS[0]: _image_to_hwc_uint8(observation[IMAGE_KEYS[0]], env_index),
        IMAGE_KEYS[1]: _image_to_hwc_uint8(observation[IMAGE_KEYS[1]], env_index),
        STATE_KEY: _batched_vector(observation[STATE_KEY], env_index, STATE_SHAPE, STATE_KEY),
        ACTION_KEY: _batched_vector(action, env_index, ACTION_SHAPE, ACTION_KEY),
        "task": task,
    }


class SuccessfulEpisodeRecorder:
    """Write one sequential environment stream and keep only successes."""

    def __init__(self, dataset: WritableLeRobotDataset) -> None:
        self.dataset = dataset
        self.successes = 0
        self.failures = 0

    def record_batch(
        self,
        *,
        observation: dict[str, Any],
        action: Any,
        success: Any,
        done: Any,
        tasks: Sequence[str],
    ) -> None:
        """Record one transition from a batch-size-one rollout.

        The observation must be the canonical pre-action observation, while
        ``action`` must be the postprocessed action actually passed to
        ``env.step``.
        """

        action_array = _to_numpy(action)
        success_array = _to_numpy(success).reshape(-1).astype(bool)
        done_array = _to_numpy(done).reshape(-1).astype(bool)
        if action_array.ndim != 2 or action_array.shape[0] != 1:
            raise ValueError(
                "Successful rollout recording requires environment batch size 1 "
                "to prevent frames from different episodes being mixed."
            )
        if success_array.shape != (1,) or done_array.shape != (1,) or len(tasks) != 1:
            raise ValueError("success, done, and tasks must each contain exactly one environment.")

        self.dataset.add_frame(
            canonical_frame(
                observation,
                action_array,
                env_index=0,
                task=tasks[0],
            )
        )

        if success_array[0]:
            self.dataset.save_episode()
            self.successes += 1
        elif done_array[0]:
            self.dataset.clear_episode_buffer()
            self.failures += 1


@dataclass(frozen=True)
class TaskCollectionResult:
    suite: str
    task_id: int
    task_description: str
    attempts: int
    successes: int


def collect_task(
    *,
    suite: str,
    task_id: int,
    task_description: str,
    env: Any,
    policy: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    recorder: SuccessfulEpisodeRecorder,
    successes_required: int,
    max_attempts: int,
    first_seed: int,
    rollout_fn: Callable[..., dict[str, Any]],
    attempt_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TaskCollectionResult:
    """Collect successful episodes for one task using LeRobot's rollout engine."""

    if successes_required < 0:
        raise ValueError("successes_required must be non-negative.")
    if max_attempts < 0:
        raise ValueError("max_attempts must be non-negative.")

    initial_successes = recorder.successes
    attempts = 0
    while recorder.successes - initial_successes < successes_required and attempts < max_attempts:
        seed = first_seed + attempts
        before = recorder.successes
        rollout_data = rollout_fn(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=[seed],
            transition_callback=recorder.record_batch,
        )
        attempts += 1
        accepted = recorder.successes > before
        success_values = _to_numpy(rollout_data["success"]).astype(bool)
        reported_success = bool(success_values.any())
        if accepted != reported_success:
            raise RuntimeError(
                "Rollout success and recorder commit disagree; refusing to continue with an inconsistent dataset."
            )

        action_values = _to_numpy(rollout_data[ACTION_KEY])
        record = {
            "suite": suite,
            "task_id": task_id,
            "task": task_description,
            "attempt": attempts,
            "seed": seed,
            "success": accepted,
            "frames": int(action_values.shape[1]),
        }
        if attempt_callback is not None:
            attempt_callback(record)

    return TaskCollectionResult(
        suite=suite,
        task_id=task_id,
        task_description=task_description,
        attempts=attempts,
        successes=recorder.successes - initial_successes,
    )
