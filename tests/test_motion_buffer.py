import pytest
import torch

from supermimic.buffer import FIELDS, ReplayBuffer
from supermimic.motion import MotionLib


def test_empty_motion_error():
    with pytest.raises(FileNotFoundError, match="No motion"):
        MotionLib([], None, "cpu")


def test_terminal_frames_do_not_create_reset_bins(cpu_motion):
    motion = cpu_motion
    assert motion.clip_len.tolist() == [51, 101]
    assert motion.num_bins == 3
    assert bool((motion.bin_start < motion.clip_end[motion.bin_clip]).all())
    assert torch.equal(motion.frame_bin[motion.clip_end], motion.frame_bin[motion.clip_end - 1])
    future = motion.future_frames(motion.clip_end, 10)
    assert torch.equal(future, motion.clip_end[:, None].expand(-1, 10))


def test_replay_wraparound_and_episode_masks():
    buffer = ReplayBuffer(5, 1, "cpu")
    for t in range(8):
        tr = {key: torch.zeros_like(value[0]) for key, value in buffer.data.items()}
        tr["f"][:] = t
        tr["s"][:] = t
        tr["ep_step"][:] = t % 3
        tr["done"][:] = t % 3 == 2
        buffer.add(tr)
    assert buffer.size == 5 and set(buffer.data) == set(FIELDS)
    assert buffer.data["f"][buffer._row(torch.arange(5)), 0].tolist() == list(range(3, 8))
    windows = buffer.sample_windows(128, 3)
    assert torch.equal(windows["s"][:, 1:, 0], windows["s"][:, :-1, 0] + 1)
    assert torch.equal(windows["mask"][:, 1], (~windows["done"][:, 0]).float())
    starts, history = buffer.sample_starts(128, 4)
    assert torch.equal(starts["f"], history["f"][:, -1])
    assert bool((history["f"] >= (starts["f"] // 3 * 3)[:, None]).all())
