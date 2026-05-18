from pathlib import Path

import numpy as np
import pytest

from videos import (
    EpisodeVideoRecorder,
    metadata_filename_part,
    metadata_value_filename_part,
    rgb_frame,
)


class FakeRenderEnv:
    def __init__(self) -> None:
        self.value = 0

    def render(self) -> np.ndarray:
        self.value += 1
        return np.full((1, 4, 5, 3), self.value, dtype=np.uint8)


def test_rgb_frame_selects_first_env_from_batched_render() -> None:
    rendered = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    rendered[0, :, :, :] = 17
    rendered[1, :, :, :] = 23

    frame = rgb_frame(rendered)

    assert frame.shape == (4, 5, 3)
    assert frame.dtype == np.uint8
    assert np.all(frame == 17)


def test_metadata_filename_part_is_filesystem_friendly() -> None:
    assert metadata_value_filename_part(-123.4567) == "neg123p457"
    assert metadata_value_filename_part(12.0) == "12p000"
    assert (
        metadata_filename_part(
            {
                "env id": "HalfCheetah-v5",
                "return": -123.4567,
            }
        )
        == "env-id-halfcheetah-v5-return-neg123p457"
    )


def test_episode_video_recorder_writes_limited_recordings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env = FakeRenderEnv()
    recorder = EpisodeVideoRecorder(
        video_dir=tmp_path,
        max_episodes=1,
        fps=12,
        name_prefix="policy",
        encoder_workers=2,
    )
    written: list[tuple[Path, list[np.ndarray]]] = []

    def fake_write_video(path: Path, frames: list[np.ndarray]) -> None:
        written.append((path, frames))

    monkeypatch.setattr(recorder, "_write_video", fake_write_video)

    recorder.capture_frame(0, env.render)
    recorder.capture_frame(0, env.render)
    recorder.finish_recording(0, metadata={"return": -1.25})
    recorder.capture_frame(1, env.render)
    recorder.finish_recording(1, metadata={"return": 5.0})
    recorder.close()

    assert env.value == 2
    assert len(written) == 1
    path, frames = written[0]
    assert path == tmp_path / "policy-episode-000-return-neg1p250.mp4"
    assert [frame.shape for frame in frames] == [(4, 5, 3), (4, 5, 3)]
    assert len(recorder.videos) == 1
    assert recorder.videos[0].num_frames == 2
    assert recorder.videos[0].metadata == {"return": -1.25}


def test_episode_video_recorder_supports_interleaved_recordings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env = FakeRenderEnv()
    recorder = EpisodeVideoRecorder(
        video_dir=tmp_path,
        max_episodes=2,
        frame_stride=2,
        encoder_workers=2,
    )
    written: list[tuple[Path, list[np.ndarray]]] = []

    def fake_write_video(path: Path, frames: list[np.ndarray]) -> None:
        written.append((path, frames))

    monkeypatch.setattr(recorder, "_write_video", fake_write_video)

    recorder.capture_frame(0, env.render)
    recorder.capture_frame(1, env.render)
    recorder.capture_frame(0, env.render)
    recorder.capture_frame(1, env.render)
    recorder.capture_frame(0, env.render)
    recorder.capture_frame(1, env.render)
    recorder.finish_recording(1, metadata={"return": 2.0})
    recorder.finish_recording(0, metadata={"return": 1.0})
    recorder.close()

    assert env.value == 4
    assert len(written) == 2
    assert sorted(path.name for path, _frames in written) == [
        "eval-episode-000-return-1p000.mp4",
        "eval-episode-001-return-2p000.mp4",
    ]
    assert sorted(len(frames) for _path, frames in written) == [2, 2]
    assert sorted(video.recording_id for video in recorder.videos) == [0, 1]


def test_episode_video_recorder_does_not_render_skipped_frames(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recorder = EpisodeVideoRecorder(
        video_dir=tmp_path,
        max_episodes=1,
        frame_stride=3,
    )
    render_calls = 0
    written: list[tuple[Path, list[np.ndarray]]] = []

    def render() -> np.ndarray:
        nonlocal render_calls
        render_calls += 1
        return np.full((4, 5, 3), render_calls, dtype=np.uint8)

    def fake_write_video(path: Path, frames: list[np.ndarray]) -> None:
        written.append((path, frames))

    monkeypatch.setattr(recorder, "_write_video", fake_write_video)

    for _frame_index in range(5):
        recorder.capture_frame(0, render)
    recorder.capture_frame(1, render)
    recorder.finish_recording(0, metadata={"return": 1.0})
    recorder.close()

    assert render_calls == 2
    assert len(written) == 1
    assert len(written[0][1]) == 2


def test_wait_records_successful_videos_before_raising_encode_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recorder = EpisodeVideoRecorder(
        video_dir=tmp_path,
        max_episodes=2,
        encoder_workers=1,
    )

    def render() -> np.ndarray:
        return np.zeros((4, 5, 3), dtype=np.uint8)

    def fake_write_video(path: Path, _frames: list[np.ndarray]) -> None:
        if "episode-001" in path.name:
            raise RuntimeError("encoder failed")

    monkeypatch.setattr(recorder, "_write_video", fake_write_video)

    recorder.capture_frame(0, render)
    recorder.finish_recording(0, metadata={"return": 1.0})
    recorder.capture_frame(1, render)
    recorder.finish_recording(1, metadata={"return": 2.0})

    with pytest.raises(ExceptionGroup, match="one or more videos failed to encode"):
        recorder.wait()

    try:
        assert [video.recording_id for video in recorder.videos] == [0]
    finally:
        recorder.close()
