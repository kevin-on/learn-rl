from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

type VideoMetadataValue = bool | int | float | str
type VideoMetadata = Mapping[str, VideoMetadataValue]


@dataclass(frozen=True)
class SavedEpisodeVideo:
    recording_id: int
    metadata: dict[str, VideoMetadataValue]
    num_frames: int
    path: Path


class EpisodeVideoRecorder:
    """Collect evaluation frames synchronously and encode finished videos in parallel."""

    def __init__(
        self,
        *,
        video_dir: Path,
        max_episodes: int,
        fps: int = 30,
        name_prefix: str = "eval",
        frame_stride: int = 1,
        crf: int = 28,
        preset: str = "medium",
        encoder_workers: int = 1,
    ) -> None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive.")
        if fps <= 0:
            raise ValueError("fps must be positive.")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive.")
        if not 0 <= crf <= 51:
            raise ValueError("crf must be in [0, 51].")
        if encoder_workers <= 0:
            raise ValueError("encoder_workers must be positive.")

        self.video_dir = video_dir
        self.max_episodes = max_episodes
        self.fps = fps
        self.name_prefix = name_prefix
        self.frame_stride = frame_stride
        self.crf = crf
        self.preset = preset
        self._frame_counts: dict[int, int] = {}
        self._frames: dict[int, list[NDArray[np.uint8]]] = {}
        self._finished_recordings: set[int] = set()
        self._executor = ThreadPoolExecutor(max_workers=encoder_workers)
        self._encoding_jobs: list[tuple[Future[None], SavedEpisodeVideo]] = []
        self._videos: list[SavedEpisodeVideo] = []
        self._closed = False

    @property
    def videos(self) -> tuple[SavedEpisodeVideo, ...]:
        return tuple(self._videos)

    def capture_frame(self, recording_id: int, render_frame: Callable[[], Any]) -> None:
        if self._closed:
            raise RuntimeError("cannot capture frames after close().")
        if recording_id < 0:
            raise ValueError("recording_id must be non-negative.")
        if recording_id >= self.max_episodes:
            return
        if recording_id in self._finished_recordings:
            raise RuntimeError("cannot capture frames after finish_recording().")

        frame_index = self._frame_counts.get(recording_id, 0)
        self._frame_counts[recording_id] = frame_index + 1
        if frame_index % self.frame_stride != 0:
            return

        frames = self._frames.setdefault(recording_id, [])
        frames.append(rgb_frame(render_frame()))

    def finish_recording(
        self,
        recording_id: int,
        metadata: VideoMetadata | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("cannot finish recordings after close().")
        if recording_id < 0:
            raise ValueError("recording_id must be non-negative.")
        if (
            recording_id >= self.max_episodes
            or recording_id in self._finished_recordings
        ):
            return

        self._finished_recordings.add(recording_id)
        frames = self._frames.pop(recording_id, [])
        self._frame_counts.pop(recording_id, None)
        if not frames:
            return

        metadata_dict = dict(metadata or {})
        self.video_dir.mkdir(parents=True, exist_ok=True)
        path = self.video_dir / (
            f"{self.name_prefix}-episode-{recording_id:03d}"
            f"-{metadata_filename_part(metadata_dict)}.mp4"
        )
        saved_video = SavedEpisodeVideo(
            recording_id=recording_id,
            metadata=metadata_dict,
            num_frames=len(frames),
            path=path,
        )
        future = self._executor.submit(
            self._write_video,
            path,
            frames,
        )
        self._encoding_jobs.append((future, saved_video))

    def close(self) -> None:
        if self._closed:
            return

        try:
            self.wait()
        finally:
            self._executor.shutdown(wait=True)
            self._closed = True

    def wait(self) -> None:
        errors: list[Exception] = []
        while True:
            if not self._encoding_jobs:
                break
            future, saved_video = self._encoding_jobs.pop(0)

            try:
                future.result()
            except Exception as exc:
                error = RuntimeError(f"failed to encode video: {saved_video.path}")
                error.__cause__ = exc
                errors.append(error)
            else:
                self._videos.append(saved_video)

        if errors:
            raise ExceptionGroup("one or more videos failed to encode", errors)

    def __enter__(self) -> "EpisodeVideoRecorder":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _write_video(self, path: Path, frames: list[NDArray[np.uint8]]) -> None:
        write_video(
            path=path,
            frames=frames,
            fps=self.fps,
            crf=self.crf,
            preset=self.preset,
        )


def write_video(
    *,
    path: Path,
    frames: list[NDArray[np.uint8]],
    fps: int,
    crf: int,
    preset: str,
) -> None:
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    except ImportError as exc:
        msg = (
            "moviepy is required to save evaluation videos. "
            "Install the project dependencies with `uv sync`."
        )
        raise RuntimeError(msg) from exc

    clip = ImageSequenceClip(frames, fps=fps)
    try:
        clip.write_videofile(
            str(path),
            codec="libx264",
            audio=False,
            preset=preset,
            ffmpeg_params=[
                "-crf",
                str(crf),
            ],
            pixel_format="yuv420p",
            logger=None,
        )
    finally:
        close = getattr(clip, "close", None)
        if callable(close):
            close()


def metadata_filename_part(metadata: VideoMetadata) -> str:
    if not metadata:
        return "recording"

    return "-".join(
        f"{filename_token(str(key))}-{metadata_value_filename_part(value)}"
        for key, value in sorted(metadata.items())
    )


def metadata_value_filename_part(value: VideoMetadataValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value).replace("-", "neg")
    if isinstance(value, float):
        return f"{value:.3f}".replace("-", "neg").replace(".", "p")
    return filename_token(value)


def filename_token(value: str) -> str:
    token = "".join(
        character if character.isalnum() else "-" for character in value.strip().lower()
    ).strip("-")
    token = "-".join(part for part in token.split("-") if part)
    return token or "value"


def rgb_frame(rendered: Any) -> NDArray[np.uint8]:
    if isinstance(rendered, list):
        if not rendered:
            raise ValueError("render() returned an empty frame list.")
        rendered = rendered[-1]

    frame = np.asarray(rendered)
    if frame.ndim == 4:
        if frame.shape[0] < 1:
            raise ValueError("batched render frame must contain at least one env.")
        frame = frame[0]

    if frame.ndim != 3 or frame.shape[-1] != 3:
        msg = f"render() must return an RGB frame; got shape {frame.shape}."
        raise ValueError(msg)

    if frame.dtype != np.uint8:
        if not np.issubdtype(frame.dtype, np.integer):
            raise TypeError(f"render() frame dtype must be uint8; got {frame.dtype}.")
        frame = frame.astype(np.uint8)

    return np.array(frame, dtype=np.uint8, copy=True)
