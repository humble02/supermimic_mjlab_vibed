import numpy as np
import pytest
import torch

from supermimic.env import TrackingSim, make_sim
from supermimic.mdp import flatten_state
from supermimic.robot import G1_JOINT_NAMES, TRACKED_BODIES, _kd, _kp, motor_of
from supermimic.rotations import quat_to_mat

pytestmark = pytest.mark.gpu


@pytest.fixture
def env(gpu_components):
    _, motion, mdp = gpu_components
    with TrackingSim(make_sim(), 4, motion, mdp, eval_mode=True) as sim:
        yield sim


def assert_fk(env):
    bp, bR, bv, bw = env.mdp.kin.forward(*env.state())
    ids = [env.robot.body_names.index(name) for name in TRACKED_BODIES]
    poses = env.robot.data.body_link_pose_w[:, ids]
    velocities = env.robot.data.body_link_vel_w[:, ids]
    torch.testing.assert_close(bp, poses[..., :3], atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(bR, quat_to_mat(poses[..., 3:]), atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(bv, velocities[..., :3], atol=5e-5, rtol=1e-5)
    torch.testing.assert_close(bw, velocities[..., 3:], atol=5e-5, rtol=1e-5)


def test_roundtrip_partial_reset_and_post_step_fk(env):
    frames = torch.tensor([4, 15, 61, 82], device=env.device)
    env.reset(torch.arange(4, device=env.device), frames)
    for actual, expected in zip(env.state(), env.motion.ref_state(frames)):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=1e-5)
    assert_fk(env)
    time_before = env.sim.data.time.clone()
    env.step(torch.zeros(4, 29, device=env.device))
    torch.testing.assert_close(env.sim.data.time - time_before, torch.full_like(time_before, 0.02), atol=1e-6, rtol=0)
    assert_fk(env)
    before = flatten_state(env.state()).clone()
    selected = torch.tensor([1, 3], device=env.device)
    env.reset(selected, frames[selected] + 2)
    after = flatten_state(env.state())
    torch.testing.assert_close(after[[0, 2]], before[[0, 2]], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(after[selected], flatten_state(env.motion.ref_state(frames[selected] + 2)), atol=3e-5, rtol=1e-5)


def test_actuators_and_collision_masks(env):
    model = env.sim.mj_model
    assert list(env.robot.joint_names) == G1_JOINT_NAMES
    for name in G1_JOINT_NAMES:
        joint = model.joint(f"robot/{name}")
        actuator_id = np.where(model.actuator_trnid[:, 0] == joint.id)[0].item()
        effort, _, armature = motor_of(name)
        assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(_kp(armature))
        assert model.actuator_biasprm[actuator_id, 2] == pytest.approx(-_kd(armature))
        assert model.actuator_forcerange[actuator_id].tolist() == [-effort, effort]
        assert model.dof_armature[joint.dofadr].item() == pytest.approx(armature)
    assert model.geom("floor").contype.item() == 2
    robot_ids = env.robot.indexing.geom_ids.cpu().numpy()
    collision = robot_ids[model.geom_contype[robot_ids] != 0]
    assert len(collision) > 0
    assert np.all(model.geom_contype[collision] == 1)
    assert np.all(model.geom_conaffinity[collision] == 2)
    torch.testing.assert_close(env.xi[:, 0], torch.ones(4, device=env.device))
    assert not bool(env.xi[:, 1:].any())


def test_eval_clip_end_does_not_cross_into_next_clip(env):
    ids = torch.arange(4, device=env.device)
    ends = env.motion.clip_end[ids % 2]
    env.reset(ids, ends - 1)
    for _ in range(3):
        out = env.step(torch.zeros(4, 29, device=env.device))
        assert bool(out["done"].all())
        assert torch.equal(env.f, ends)
        assert bool(torch.isfinite(out["s1"]).all())


def test_randomization_and_autoreset_terminal_transition(gpu_components, monkeypatch):
    _, motion, mdp = gpu_components
    # Isolate timeout behavior from tracking failures.
    monkeypatch.setattr(mdp, "termination", lambda state, frames: torch.zeros_like(frames, dtype=torch.bool))
    with TrackingSim(make_sim(), 4, motion, mdp, max_episode_steps=1, reset_noise=False) as env:
        mu = env.sim.model.geom_friction[:, env.robot.indexing.geom_ids, 0]
        torch.testing.assert_close(mu, env.xi[:, :1].expand_as(mu))
        torso = env.robot.indexing.body_ids[env.torso_id]
        base = env.sim.get_default_field("body_ipos")[torso]
        torch.testing.assert_close(env.sim.model.body_ipos[:, torso] - base, env.xi[:, 30:], atol=1e-6, rtol=0)
        assert torch.unique(env.xi[:, 0]).numel() == 4
        assert env.sim.model.body_invweight0.shape[0] == 4
        before = flatten_state(env.state()).clone()
        xi = env.xi.clone()
        out = env.step(torch.zeros(4, 29, device=env.device))
        torch.testing.assert_close(out["s"], before)
        torch.testing.assert_close(out["xi"], xi)
        assert bool(out["done"].all()) and not bool(out["term"].any())
        assert not bool(env.ep_step.any())
        assert not torch.equal(out["s1"], flatten_state(env.state()))
        assert_fk(env)
