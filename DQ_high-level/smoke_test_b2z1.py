"""Smoke test: construct B2Z1PickMulti env with a tiny number of envs and step it.

Run from DQ_high-level:
    LD_LIBRARY_PATH=/opt/conda/envs/dqwbc/lib \
    /opt/conda/envs/dqwbc/bin/python smoke_test_b2z1.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'isaacgym', 'python'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaacgym  # noqa: F401  (must be imported before torch)

from termcolor import cprint

from utils.config import get_params, load_cfg
from legged_gym.envs.manip_loco.b2z1_config import B2Z1RoughCfg
from envs import B2Z1PickMulti

import torch


def main():
    args = get_params()
    args.eval = True
    args.debug = True  # forces numEnvs=34
    args.use_roboinfo = True
    args.observe_gait_commands = True
    args.rand_control = True
    args.small_value_set_zero = False
    args.headless = True

    cfg_file = "DQ_teacher_b2z1.yaml"
    cfg = load_cfg("data/cfg/" + cfg_file)
    cfg['env']['useTanh'] = False
    cfg['env']['smallValueSetZero'] = args.small_value_set_zero
    cfg['env']['cameraMode'] = "full"
    cfg['env']['wandb'] = False
    if args.debug:
        cfg['env']['numEnvs'] = 16

    cfg_terrain = B2Z1RoughCfg()

    # Build the grasp predict point tensors from the real transform data
    from train_multistate_DQ_teacher import get_predict_point
    num_envs = cfg['env']['numEnvs']
    camera_6p_tensor, grasp_cv_tensor, cube_init_tensor = get_predict_point(
        cube_predict_info_path="contact_grasp_info_mul",
        cube_root_states_info_path="30all_nomove_cube_root_states5.pt",
        num_env=num_envs, intervel=23, delta_height=0.1)

    cprint(f"Constructing B2Z1PickMulti with {num_envs} envs ...", "cyan")
    env = B2Z1PickMulti(
        cfg=cfg,
        rl_device=args.rl_device,
        sim_device=args.sim_device,
        graphics_device_id=args.graphics_device_id,
        headless=args.headless,
        use_roboinfo=args.use_roboinfo,
        observe_gait_commands=args.observe_gait_commands,
        robot_start_pose=(-0.85, 0, 0.55),
        rand_control=args.rand_control,
        cfg_terrain=cfg_terrain,
        camera_6p_tensor=camera_6p_tensor,
        grasp_cv_tensor=grasp_cv_tensor,
        cube_init_tensor=cube_init_tensor,
    )
    cprint("Env constructed OK!", "green")

    # First reset (matches the trainer flow), then step
    env.reset()
    obs, rew, reset, extras = env.step(torch.zeros(num_envs, env.num_actions, device=env.device))
    cprint(f"step done, obs shape={list(obs['obs'].shape)} rew={rew.shape}", "green")

    # Step with random actions
    for _ in range(5):
        obs, rew, reset, extras = env.step(torch.randn(num_envs, env.num_actions, device=env.device) * 0.01)
        print("rew mean:", round(rew.mean().item(), 4), "reset rate:", round((reset > 0).float().mean().item(), 3))

    cprint("SMOKE TEST PASSED", "green")


if __name__ == "__main__":
    main()