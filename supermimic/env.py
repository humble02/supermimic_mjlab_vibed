"""Motion-tracking simulator on mjlab (native MJCF and MuJoCo Warp).

The environment is a thin batched wrapper: it applies the joint setpoint, steps MuJoCo-Warp, labels
the transition with the analytic reward/termination of :mod:`supermimic.mdp`, and resets episodes to
reference-motion states drawn by return-based bin sampling (paper Appendix D.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import G1_JOINT_NAMES, action_scale, make_robot_cfg
from .rotations import mat_to_quat, quat_to_mat, so3_exp


@dataclass
class TrackingSimCfg:
    device: str = "cuda:0"
    decimation: int = 4
    physics: SimulationCfg = field(default_factory=lambda: SimulationCfg(
        njmax=500, nconmax=128,
        mujoco=MujocoCfg(timestep=CONTROL_DT / 4, cone="pyramidal", integrator="implicitfast"),
    ))


def make_sim(device: str = "cuda:0", args=None) -> TrackingSimCfg:
    return TrackingSimCfg(device=device)


def _add_ground(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.0, 0.0, 0.05), contype=2, conaffinity=1,
        friction=(0.3, 0.005, 0.0001), condim=3,
    )


@requires_model_fields("geom_friction", "body_ipos", recompute=RecomputeLevel.set_const)
def _randomization_fields():
    """Declare independent per-world parameters and their derived model constants."""


class TrackingSim:
    """Batched G1 motion-tracking simulator.

    Uses mjlab's Scene, Entity and Simulation directly; no application launcher is needed.
    State quaternions use wxyz; root velocities are measured at the link origin in world space.
    """

    def __init__(self, sim_cfg, num_envs: int, motion: MotionLib, mdp: MDP, gamma: float = 0.99, eval_mode: bool = False,
                 max_episode_steps: int = 500, reset_noise: bool = True):
        self.device = sim_cfg.device
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.cfg = sim_cfg
        self.scene = Scene(SceneCfg(
            num_envs=num_envs, env_spacing=0.0,
            entities={"robot": make_robot_cfg()}, spec_fn=_add_ground,
        ), device=self.device)
        self.sim = Simulation(num_envs, sim_cfg.physics, self.scene.compile(), self.device)
        self.sim.expand_model_fields(_randomization_fields.model_fields)
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.robot = self.scene["robot"]
        assert list(self.robot.joint_names) == G1_JOINT_NAMES, self.robot.joint_names
        self.N = num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        dev = self.device
        self.default_q = self.robot.data.default_joint_pos[0].clone()
        self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=dev)
        self.torso_id = self.robot.find_bodies("torso_link")[0][0]

        # ---- sim context xi = [friction, joint offsets(29), torso com offset(3)] ----
        self.xi = torch.zeros(self.N, 33, device=dev)
        self._setup_friction()
        if eval_mode:
            self.xi[:, 0] = 1.0
        else:
            self.xi[:, 0] = torch.empty(self.N, device=dev).uniform_(0.3, 1.2)
            com = torch.stack(
                (
                    torch.empty(self.N, device=dev).uniform_(-0.025, 0.025),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                ),
                -1,
            )
            self.xi[:, 30:33] = com
            torso_global_id = self.robot.indexing.body_ids[self.torso_id]
            self.sim.model.body_ipos[:, torso_global_id] += com
            self.sim.recompute_constants(RecomputeLevel.set_const)
        self._apply_friction()

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, 29, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    # ------------------------------------------------------------------ friction
    def _setup_friction(self):
        self._robot_geom_ids = self.robot.indexing.geom_ids
        floor = self.sim.mj_model.geom("floor").id
        self.sim.model.geom_friction[:, floor, 0] = 1.0 if self.eval_mode else 0.3

    def _apply_friction(self):
        self.sim.model.geom_friction[:, self._robot_geom_ids, 0] = self.xi[:, :1]

    # ------------------------------------------------------------------ state
    def state(self):
        d = self.robot.data
        pose = d.root_link_pose_w.clone()
        vel = d.root_link_vel_w.clone()
        p = pose[:, :3] - self.scene.env_origins
        R = quat_to_mat(pose[:, 3:7])  # mjlab uses wxyz.
        return (p, R, vel[:, :3], vel[:, 3:6], d.joint_pos.clone(), d.joint_vel.clone())

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m = self.motion
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            p = p + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                 torch.empty(n, device=dev).uniform_(-0.01, 0.01)), -1)  # fmt: skip
            rv = torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            R = so3_exp(rv) @ R
            v = v + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.5, 0.5), torch.empty(n, device=dev).uniform_(-0.5, 0.5),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            w = w + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.52, 0.52), torch.empty(n, device=dev).uniform_(-0.52, 0.52),
                 torch.empty(n, device=dev).uniform_(-0.78, 0.78)), -1)  # fmt: skip
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        quat = mat_to_quat(R)
        # Clear contacts, warm-start accelerations and actuator targets before writing the reset.
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        pose = torch.cat((p + self.scene.env_origins[env_ids], quat), -1)
        self.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        if not self.eval_mode:
            self.xi[env_ids, 1:30] = torch.empty(n, 29, device=dev).uniform_(-0.01, 0.01)
        self.done_eval[env_ids] = False
        self.failed_eval[env_ids] = False
        self.scene.write_data_to_sim()
        self.refresh()

    def refresh(self):
        """Propagate written states so that ``state()`` is consistent after a reset."""
        self.sim.forward()
        self.scene.update(0.0)

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1:30] + self.alpha * a
        self.robot.set_joint_position_target(target)
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.cfg.physics.mujoco.timestep)
        # MuJoCo step integrates qpos after computing xpos/cvel. Refresh so all
        # components of s1 describe the same post-integration instant.
        self.refresh()
        S1 = self.state()
        f1 = torch.minimum(f + 1, self.motion.clip_end[self.motion.clip_id[f]])
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms,
        }  # fmt: skip
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            # Multiple environments may finish from the same bin in this step.
            # Aggregate them before the EMA; repeated advanced-index writes race.
            totals = torch.zeros_like(self.bin_G).scatter_add_(0, b, G)
            counts = torch.zeros_like(self.bin_G).scatter_add_(0, b, torch.ones_like(G))
            used = counts > 0
            self.bin_G[used] = 0.8 * self.bin_G[used] + 0.2 * totals[used] / counts[used]
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        # mjlab owns no external application. Drop graph/entity references to
        # release GPU resources when used as a context manager.
        self.robot = None
        self.scene = None
        self.sim = None
