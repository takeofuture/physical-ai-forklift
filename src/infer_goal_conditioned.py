#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import rclpy
from stable_baselines3 import SAC

# Same directory: src/train_goal_conditioned.py
import train_goal_conditioned as train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for Goal-Conditioned SAC forklift"
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_success_model.zip",
        help="Stable-Baselines3 SAC checkpoint",
    )

    # Goal
    parser.add_argument("--target-row", type=int, default=1)
    parser.add_argument("--target-bin", type=int, default=10)
    parser.add_argument("--target-level", type=int, default=1)
    parser.add_argument(
        "--target-face",
        choices=["auto", "low", "high"],
        default="auto",
    )

    # Spawn pose
    parser.add_argument("--spawn-x", type=float, default=38.0)
    parser.add_argument("--spawn-y", type=float, default=14.0)
    parser.add_argument("--spawn-z", type=float, default=0.10)
    parser.add_argument("--spawn-yaw", type=float, default=-1.571)

    # Model / world
    parser.add_argument("--world-name", default="warehouse_world")
    parser.add_argument("--model-name", default="forklift_simple")
    parser.add_argument(
        "--model-sdf",
        default="models/forklift_simple/model.sdf",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Additional inference safety limit",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="SB3 device: auto, cpu, cuda, cuda:0, etc.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic SAC actions instead of deterministic inference",
    )

    return parser.parse_args()


def make_train_args(args):
    """
    Build the argument object expected by
    train_goal_conditioned.build_env().

    Defaults match the final train13 configuration.
    """

    return SimpleNamespace(
        # World / robot
        world_name=args.world_name,
        model_name=args.model_name,

        # Explicit XYZ fallback
        goal_x=0.0,
        goal_y=0.0,
        goal_z=2.5,

        model_sdf=args.model_sdf,

        # Spawn
        spawn_x=args.spawn_x,
        spawn_y=args.spawn_y,
        spawn_yaw=args.spawn_yaw,
        spawn_z=args.spawn_z,

        # Fixed inference target
        target_row=args.target_row,
        target_bin=args.target_bin,
        target_level=args.target_level,
        target_face=args.target_face,

        # IMPORTANT:
        # inference uses the explicitly requested target
        randomize_target=False,

        # Required by build_env even though randomization is disabled
        target_rows=str(args.target_row),
        target_bins=str(args.target_bin),
        target_levels=str(args.target_level),

        disable_target_map=False,

        # ROS / Gazebo topics
        odom_topic=None,
        cmd_vel_topic=None,
        lift_cmd_topic=None,
        joint_topic=None,
        contact_topics="",
        gz_pose_topic="",
        gz_pose_timeout=8.0,

        # Debug / visualization
        debug_every_steps=50,
        no_target_marker=False,
        xy_reference_point="base",

        # Lift / PHASE4 parameters from final train13
        max_lift_force=150.0,
        phase4_max_lift_force=800.0,
        lift_max=8.4,

        phase4_force_deadband=0.10,
        phase4_force_taper_band=0.55,
        phase4_overshoot_penalty_gain=45.0,
        phase4_big_overshoot_penalty_gain=120.0,

        phase_pre_lift_hold_force=0.0,
        phase4_entry_stop_sec=0.80,
        phase4_entry_stop_dt=0.05,

        # Do NOT run the training-time actuator self-test before a demo.
        # reset() still respawns the forklift normally.
        no_respawn_self_test=True,
        respawn_self_test_warning_only=False,
        selftest_require_lift_down=False,
        selftest_drive_v=0.80,
        selftest_drive_sec=1.20,
        selftest_lift_force=800.0,
        selftest_lift_sec=1.40,
    )


def main():
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    model_sdf = Path(args.model_sdf)

    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    if not model_sdf.exists():
        print(f"ERROR: forklift SDF not found: {model_sdf}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Goal-Conditioned Forklift Inference")
    print("=" * 70)
    print(f"Checkpoint : {checkpoint}")
    print(f"World      : {args.world_name}")
    print(f"Robot      : {args.model_name}")
    print(f"Model SDF  : {model_sdf}")
    print(
        f"Target     : row={args.target_row} "
        f"bin={args.target_bin} "
        f"level={args.target_level} "
        f"face={args.target_face}"
    )
    print(f"Spawn      : ({args.spawn_x}, {args.spawn_y}, {args.spawn_z})")
    print(f"Device     : {args.device}")
    print(
        "Policy     : "
        + ("stochastic" if args.stochastic else "deterministic")
    )
    print("=" * 70)

    rclpy.init()

    env = None

    try:
        train_args = make_train_args(args)

        # Reuse exactly the same Gazebo/ROS/Gym environment as training
        env = train.build_env(train_args)

        print("\nLoading SAC model...")

        model = SAC.load(
            str(checkpoint),
            env=env,
            device=args.device,
        )

        print("Model loaded.")
        print("\nResetting environment and spawning forklift...")

        obs, info = env.reset()

        print("\nInference started.")
        print("-" * 70)

        total_reward = 0.0
        step_count = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            action, _ = model.predict(
                obs,
                deterministic=(not args.stochastic),
            )

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            step_count += 1

            if step_count >= args.max_steps:
                print(
                    f"\nINFERENCE_SAFETY_STOP: "
                    f"max_steps={args.max_steps}"
                )
                break

        success = bool(info.get("is_success", False))

        print()
        print("=" * 70)
        print("INFERENCE RESULT")
        print("=" * 70)
        print(f"Target       : row={args.target_row} "
              f"bin={args.target_bin} "
              f"level={args.target_level}")
        print(f"Steps        : {step_count}")
        print(f"Total reward : {total_reward:.3f}")
        print(f"Success      : {success}")
        print(f"Terminated   : {terminated}")
        print(f"Truncated    : {truncated}")

        if "phase" in info:
            print(f"Final phase  : {info['phase']}")

        print("=" * 70)

        sys.exit(0 if success else 2)

    finally:
        if env is not None:
            env.close()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
