"""Validate state roundtrips, tracked-body FK and stepping on the actual motion library."""

import argparse
from pathlib import Path
import time

import torch

from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, MAX_STEP_REWARD
from supermimic.motion import MotionLib
from supermimic.robot import TRACKED_BODIES, default_joint_pos_list
from supermimic.rotations import quat_to_mat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--motion_dir", default=str(Path(__file__).resolve().parents[1] / "data/lafan1_g1"))
    args = parser.parse_args()
    device = args.device
    kin = G1Kinematics(device)
    motion = MotionLib.from_dir(args.motion_dir, kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    print(f"Loaded {len(motion.names)} clips, {motion.num_frames} frames at 50 Hz, {motion.num_bins} reset bins")
    with TrackingSim(make_sim(device), args.num_envs, motion, mdp, eval_mode=True) as env:
        ids = torch.arange(args.num_envs, device=device)
        clips = ids % len(motion.names)
        frames = motion.clip_start[clips] + torch.minimum(motion.clip_len[clips] // 2, torch.full_like(ids, 100))
        env.reset(ids, frames)
        ref = list(motion.ref_state(frames))
        ref[4] = ref[4].clamp(kin.q_lo, kin.q_hi)
        for name, actual, expected in zip(("p", "R", "v", "w", "q", "qd"), env.state(), ref):
            error = (actual - expected).abs().max().item()
            print(f"Roundtrip {name}: max error {error:.3e}")
            torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-5)
        body_ids = [env.robot.body_names.index(b) for b in TRACKED_BODIES]
        bp, bR, bv, bw = kin.forward(*env.state())
        poses = env.robot.data.body_link_pose_w[:, body_ids]
        velocities = env.robot.data.body_link_vel_w[:, body_ids]
        for name, actual, expected in (
            ("position", bp, poses[..., :3]), ("rotation", bR, quat_to_mat(poses[..., 3:])),
            ("linear velocity", bv, velocities[..., :3]), ("angular velocity", bw, velocities[..., 3:]),
        ):
            print(f"FK {name}: max error {(actual - expected).abs().max().item():.3e}")
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            future = torch.minimum(env.f + 1, motion.clip_end[motion.clip_id[env.f]])
            action = (motion.q[future] - env.default_q) / env.alpha
            out = env.step(action)
            assert bool(torch.isfinite(out["s1"]).all()), "Nonfinite simulator state"
            assert bool(torch.isfinite(out["r"]).all()), "Nonfinite reward"
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        print(f"PASS: {args.steps} steps, {args.num_envs * args.steps / elapsed:.0f} transitions/s")
        if args.steps:
            print(f"Final open-loop normalized reward: {out['r'].mean().item() / MAX_STEP_REWARD:.4f}")


if __name__ == "__main__":
    main()
