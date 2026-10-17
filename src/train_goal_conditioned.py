
#!/usr/bin/env python3
# train12_goal_conditioned: rule-based route decomposition + Goal-Conditioned SAC.
# Based on train11, but target row/bin/level can be randomized per episode and
# goal/waypoint features are appended to the observation.
# The phase-gated actuator publishing is kept:
#   PHASE0-3 publish cmd_vel only and DO NOT publish lift_force_cmd.
#   PHASE4 publishes lift_force_cmd only; before PHASE4 is enabled, cmd_vel is hard-stopped.
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import threading
import time
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

try:
    from ros_gz_interfaces.msg import Contacts
    HAS_CONTACTS = True
except Exception:
    Contacts = None
    HAS_CONTACTS = False

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList


PHASE_NAMES = {
    0: "to_portal",
    1: "along_aisle",
    2: "align_xy",
    3: "face_rack",
    4: "lift_only",
}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def parse_int_set(spec: str, lo: int, hi: int) -> Tuple[int, ...]:
    """Parse CLI ranges such as '0,1,2', '8-12', or '0-3,10'."""
    spec = (spec or "").strip()
    if not spec:
        return tuple(range(lo, hi + 1))
    vals = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a_s, b_s = part.split('-', 1)
            a, b = int(a_s), int(b_s)
            if a > b:
                a, b = b, a
            vals.extend(range(a, b + 1))
        else:
            vals.append(int(part))
    vals = sorted({max(lo, min(hi, v)) for v in vals})
    if not vals:
        vals = list(range(lo, hi + 1))
    return tuple(vals)


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def point_dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def local_to_world_pose(x: float, y: float, yaw: float, lx: float, ly: float) -> Tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return x + c * lx - s * ly, y + s * lx + c * ly


def sample_collision_points_for_pose(cfg: "EnvConfig", x: float, y: float, yaw: float) -> List[Tuple[str, float, float]]:
    pts_local = [
        ("rear_center", -cfg.base_half_len, 0.0),
        ("front_center", cfg.base_half_len, 0.0),
        ("rear_left", -cfg.base_half_len, cfg.base_half_wid),
        ("rear_right", -cfg.base_half_len, -cfg.base_half_wid),
        ("front_left", cfg.base_half_len, cfg.base_half_wid),
        ("front_right", cfg.base_half_len, -cfg.base_half_wid),
        ("fork_left", cfg.fork_tip_x, cfg.fork_y),
        ("fork_right", cfg.fork_tip_x, -cfg.fork_y),
    ]
    return [(name, *local_to_world_pose(x, y, yaw, lx, ly)) for name, lx, ly in pts_local]


def geometry_collision_for_pose(cfg: "EnvConfig", x: float, y: float, yaw: float) -> Tuple[bool, str]:
    half_x = cfg.warehouse_x / 2.0 - cfg.wall_collision_margin
    half_y = cfg.warehouse_y / 2.0 - cfg.wall_collision_margin
    for name, px, py in sample_collision_points_for_pose(cfg, x, y, yaw):
        if abs(px) >= half_x or abs(py) >= half_y:
            return True, f"wall:{name}"

    m = cfg.rack_collision_margin
    xmin = cfg.map_cfg.rack_x_min - m
    xmax = cfg.map_cfg.rack_x_max + m
    half_w = cfg.map_cfg.rack_wid / 2.0 + m
    for row_idx, row_y in enumerate(cfg.map_cfg.row_y_centers):
        ymin = row_y - half_w
        ymax = row_y + half_w
        for name, px, py in sample_collision_points_for_pose(cfg, x, y, yaw):
            if xmin <= px <= xmax and ymin <= py <= ymax:
                return True, f"rack{row_idx}:{name}"
    return False, ""


class TargetMarkerManager:
    def __init__(self, world_name: str, model_name: str = "rl_target_marker"):
        self.world_name = world_name
        self.model_name = model_name
        self.tmpdir = Path(tempfile.gettempdir()) / "forklift_rl_marker"
        self.tmpdir.mkdir(parents=True, exist_ok=True)
        self.last_pose: Optional[Tuple[float, float, float]] = None

    def _remove_model(self, name: str):
        cmd = [
            "gz", "service", "-s", f"/world/{self.world_name}/remove",
            "--reqtype", "gz.msgs.Entity", "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000", "--req", f'name: "{name}", type: MODEL',
        ]
        try:
            subprocess.run(cmd, check=False, capture_output=True, text=True)
        except Exception:
            pass

    def _write_model_file(self, goal_z: float, pad_xy: float, pole_xy: float, top_xy: float, top_z: float) -> Path:
        sdf_path = self.tmpdir / f"{self.model_name}.sdf"
        marker_radius = max(0.18, top_xy * 0.60)
        marker_z = max(0.35, goal_z)
        model = f'''<?xml version="1.0" ?>
<sdf version="1.10">
  <model name="{self.model_name}">
    <static>true</static>
    <link name="goal_marker">
      <pose>0 0 {marker_z:.3f} 0 0 0</pose>
      <visual name="visual">
        <geometry><sphere><radius>{marker_radius:.3f}</radius></sphere></geometry>
        <material><ambient>1 0.2 0.2 0.85</ambient><diffuse>1 0.2 0.2 0.85</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
'''
        sdf_path.write_text(model, encoding="utf-8")
        return sdf_path

    def ensure(self, goal_x: float, goal_y: float, goal_z: float, pad_xy: float = 1.2, pole_xy: float = 0.08, top_xy: float = 0.35, top_z: float = 0.35, force: bool = False):
        pose = (round(goal_x, 4), round(goal_y, 4), round(goal_z, 4))
        if (not force) and self.last_pose == pose:
            return
        self._remove_model(self.model_name)
        sdf_path = self._write_model_file(goal_z, pad_xy, pole_xy, top_xy, top_z)
        req = (
            f'sdf_filename: "{sdf_path}", name: "{self.model_name}", ' 
            f'pose: {{position: {{x: {goal_x:.4f}, y: {goal_y:.4f}, z: 0.0}}, orientation: {{x: 0, y: 0, z: 0, w: 1}}}}'
        )
        cmd = [
            "gz", "service", "-s", f"/world/{self.world_name}/create",
            "--reqtype", "gz.msgs.EntityFactory", "--reptype", "gz.msgs.Boolean",
            "--timeout", "5000", "--req", req,
        ]
        try:
            res = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"WARN marker spawn rc={res.returncode} stdout={res.stdout.strip()} stderr={res.stderr.strip()}", flush=True)
            else:
                self.last_pose = pose
        except Exception as e:
            print(f"WARN marker spawn exception: {e}", flush=True)


@dataclass
class RewardZone:
    phase: int
    idx: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


@dataclass
class WarehouseMap:
    rack_len: float = 70.0
    rack_wid: float = 2.0
    num_bins: int = 20
    num_rows: int = 3
    aisle_gap: float = 4.0
    shelf_zs: Tuple[float, ...] = (0.1, 2.1, 4.1, 6.1)
    level_target_offset: float = 0.4
    portal_margin: float = 3.0
    wp0_tol: float = 0.9
    wp1_tol: float = 1.2
    xy_tol: float = 0.6

    phase0_zone_count: int = 6
    phase1_zone_count: int = 12
    phase0_half_width_x: float = 2.4
    phase1_half_width_y: float = 2.2

    @property
    def bin_pitch(self) -> float:
        return self.rack_len / self.num_bins

    @property
    def row_pitch(self) -> float:
        return self.rack_wid + self.aisle_gap

    @property
    def rack_x_min(self) -> float:
        return -self.rack_len / 2.0

    @property
    def rack_x_max(self) -> float:
        return self.rack_len / 2.0

    @property
    def row_y_centers(self) -> List[float]:
        total_rows_span = (self.num_rows - 1) * self.row_pitch
        row_y0 = -total_rows_span / 2.0
        return [row_y0 + i * self.row_pitch for i in range(self.num_rows)]

    @property
    def lane_y_centers(self) -> List[float]:
        rows = self.row_y_centers
        half = self.row_pitch / 2.0
        lanes = [rows[0] - half]
        lanes.extend([r + half for r in rows])
        return lanes

    @property
    def bin_x_centers(self) -> List[float]:
        return [self.rack_x_min + (i + 0.5) * self.bin_pitch for i in range(self.num_bins)]

    @property
    def level_target_zs(self) -> List[float]:
        return [z + self.level_target_offset for z in self.shelf_zs]

    def clamp_target(self, row: int, bin_idx: int, level: int) -> Tuple[int, int, int]:
        return (
            max(0, min(self.num_rows - 1, row)),
            max(0, min(self.num_bins - 1, bin_idx)),
            max(0, min(len(self.shelf_zs) - 1, level)),
        )

    def choose_lane(self, row: int, start_y: float, face: str = "auto") -> Tuple[int, float, str]:
        low_idx = row
        high_idx = row + 1
        lanes = self.lane_y_centers
        if face == "low":
            idx = low_idx
        elif face == "high":
            idx = high_idx
        else:
            idx = low_idx if abs(start_y - lanes[low_idx]) <= abs(start_y - lanes[high_idx]) else high_idx
        face_name = "low" if idx == low_idx else "high"
        return idx, lanes[idx], face_name

    def build_plan(self, row: int, bin_idx: int, level: int, start_x: float, start_y: float, face: str = "auto") -> Dict[str, Any]:
        row, bin_idx, level = self.clamp_target(row, bin_idx, level)
        lane_idx, lane_y, face_name = self.choose_lane(row, start_y, face)
        goal_x = self.bin_x_centers[bin_idx]
        goal_y = lane_y
        goal_z = self.level_target_zs[level]
        portal_sign = 1.0 if start_x >= 0.0 else -1.0
        portal_x = portal_sign * (self.rack_x_max + self.portal_margin)
        wp0 = (portal_x, lane_y)
        wp1 = (goal_x, goal_y)
        zones0 = self._build_segment_zones(0, (start_x, start_y), wp0, self.phase0_zone_count, orth_half=self.phase0_half_width_x)
        zones1 = self._build_segment_zones(1, wp0, wp1, self.phase1_zone_count, orth_half=self.phase1_half_width_y)
        target_yaw = math.pi / 2.0 if face_name == "low" else -math.pi / 2.0
        return {
            "row": row,
            "bin": bin_idx,
            "level": level,
            "face": face_name,
            "lane_idx": lane_idx,
            "lane_y": lane_y,
            "goal_x": goal_x,
            "goal_y": goal_y,
            "goal_z": goal_z,
            "target_yaw": target_yaw,
            "portal_x": portal_x,
            "wp0": wp0,
            "wp1": wp1,
            "zones0": zones0,
            "zones1": zones1,
        }

    def _build_segment_zones(self, phase: int, a: Tuple[float, float], b: Tuple[float, float], count: int, orth_half: float) -> List[RewardZone]:
        ax, ay = a
        bx, by = b
        zones: List[RewardZone] = []
        if abs(ax - bx) < 1e-6:
            # vertical strip stack
            ys = np.linspace(ay, by, count + 1)
            xmin = ax - orth_half
            xmax = ax + orth_half
            for i in range(count):
                y0, y1 = float(ys[i]), float(ys[i + 1])
                zones.append(RewardZone(phase, i, xmin, xmax, min(y0, y1), max(y0, y1)))
        elif abs(ay - by) < 1e-6:
            # horizontal strip stack
            xs = np.linspace(ax, bx, count + 1)
            ymin = ay - orth_half
            ymax = ay + orth_half
            for i in range(count):
                x0, x1 = float(xs[i]), float(xs[i + 1])
                zones.append(RewardZone(phase, i, min(x0, x1), max(x0, x1), ymin, ymax))
        else:
            # fallback rectangular bounds around segment
            xs = np.linspace(ax, bx, count + 1)
            ys = np.linspace(ay, by, count + 1)
            for i in range(count):
                x0, x1 = float(xs[i]), float(xs[i + 1])
                y0, y1 = float(ys[i]), float(ys[i + 1])
                zones.append(RewardZone(phase, i, min(x0, x1) - orth_half, max(x0, x1) + orth_half,
                                        min(y0, y1) - orth_half, max(y0, y1) + orth_half))
        return zones


@dataclass
class EnvConfig:
    world_name: str = "warehouse_world"
    model_name: str = "forklift_simple"

    target_row: int = 2
    target_bin: int = 10
    target_level: int = 1
    target_face: str = "auto"

    # Goal-conditioned training mode.
    # If randomize_target=True, reset() samples row/bin/level from these candidates.
    # The rule-based route decomposition still converts row/bin/level -> wp0/wp1/goal_z/yaw.
    randomize_target: bool = False
    target_rows: Tuple[int, ...] = (1,)
    target_bins: Tuple[int, ...] = (10,)
    target_levels: Tuple[int, ...] = (1,)

    use_target_map: bool = True
    goal_x: float = 0.0
    goal_y: float = 0.0
    goal_z: float = 2.5

    # reset / respawn
    model_sdf: str = ""
    spawn_x: float = 38.0
    spawn_y: float = 14.0
    spawn_yaw: float = -1.571
    spawn_z: float = 0.10
    spawn_pose_tol_xy: float = 0.35
    spawn_pose_tol_yaw: float = 0.35
    respawn_remove_wait_sec: float = 0.20
    respawn_settle_sec: float = 0.80

    map_cfg: WarehouseMap = field(default_factory=WarehouseMap)

    # success thresholds
    xy_tol: float = 0.60
    z_tol: float = 0.12
    yaw_tol: float = 0.20
    xy_reference_point: str = "base"

    # stop thresholds
    stop_v_thresh: float = 0.05
    stop_yaw_thresh: float = 0.05
    stop_lift_thresh: float = 0.02

    # default action scaling (used as fallback)
    max_v: float = 3.2
    max_yaw_rate: float = 2.8
    # Deprecated velocity-style name kept for compatibility with older logs.
    # train9 uses force command for lift_joint through /lift_force_cmd.
    max_lift_rate: float = 0.45
    max_lift_force: float = 150.0

    # phase-specific action profiles
    phase0_control_dt: float = 0.12
    phase0_action_repeat: int = 10
    phase0_max_v: float = 3.2
    phase0_max_yaw_rate: float = 2.8

    phase1_control_dt: float = 0.10
    phase1_action_repeat: int = 8
    phase1_max_v: float = 2.8
    phase1_max_yaw_rate: float = 2.2

    phase2_control_dt: float = 0.06
    phase2_action_repeat: int = 3
    phase2_max_v: float = 0.85
    phase2_max_yaw_rate: float = 0.90

    phase3_control_dt: float = 0.06
    phase3_action_repeat: int = 3
    phase3_max_yaw_rate: float = 0.85

    phase4_control_dt: float = 0.05
    phase4_action_repeat: int = 2
    # Deprecated velocity-style name kept only so old CLI/log mental model is clear.
    # PHASE4 action[2] is converted to FORCE, not velocity.
    phase4_max_lift_rate: float = 0.35
    phase4_max_lift_force: float = 150.0

    # actuator masks / safety wrappers
    # train11 does NOT use a pre-lift force during PHASE0-3.
    # This field is kept only for CLI compatibility with train10 and is ignored by step().
    # PHASE0-3: publish cmd_vel only; do not publish lift_force_cmd.
    # PHASE4: publish lift_force_cmd only; stop cmd_vel at phase entry.
    phase_pre_lift_hold_force: float = 0.0

    # PHASE3 -> PHASE4 transition safety stop:
    # yaw is judged in PHASE3 while the robot may still be rotating. Before setting
    # phase=4, explicitly send cmd_vel=(0,0) through ROS2->Gazebo for a short period
    # so PHASE4 lift starts from a stopped horizontal/rotational state.
    phase4_entry_stop_sec: float = 0.80
    phase4_entry_stop_dt: float = 0.05

    # respawn self-test: after each respawn, briefly test drive and lift, log results,
    # then respawn once more to restore a clean training start pose.
    respawn_self_test: bool = True
    respawn_self_test_required: bool = True
    # In force-control lift mode, a down-force test can be noisy because the controller/model
    # may continue drifting upward briefly. By default, do not block training on lift-down;
    # still log it. Enable strict behavior with --selftest-require-lift-down.
    respawn_self_test_require_lift_down: bool = False
    respawn_self_test_drive_v: float = 0.25
    respawn_self_test_drive_sec: float = 0.45
    respawn_self_test_lift_force: float = 150.0
    respawn_self_test_lift_sec: float = 0.65
    respawn_self_test_settle_sec: float = 0.20
    respawn_self_test_min_move_delta: float = 0.03
    respawn_self_test_min_lift_delta: float = 0.03
    respawn_self_test_max_return_xy: float = 0.40
    respawn_self_test_max_return_lift: float = 0.15

    # lift limits
    # IMPORTANT for tall-shelf training: this must match / be slightly above the SDF lift_joint upper limit.
    # Old train12 used 2.6 because the SDF upper limit was 2.4.
    # After raising the SDF lift_joint upper to 8.4, clipping at 2.6 makes level=2/3 impossible.
    lift_min: float = 0.0
    lift_max: float = 8.4

    # PHASE4 lift safety / reward shaping for multi-level goals.
    # Stage C was trained only with level=1, so the inherited policy tends to keep lifting.
    # These guards keep Goal-Conditioned SAC learnable without changing the observation/action shape.
    phase4_force_deadband: float = 0.10          # if |goal_z-lift_z| <= this, send 0 force
    phase4_force_taper_band: float = 0.55       # reduce force near target
    phase4_near_force_scale_min: float = 0.12   # minimum multiplier inside taper band
    phase4_wrong_direction_scale: float = 0.0   # block force that moves away from target
    phase4_overshoot_penalty_gain: float = 45.0
    phase4_big_overshoot_penalty_gain: float = 120.0
    phase4_big_overshoot_margin: float = 0.25
    phase4_abs_z_penalty_gain: float = 0.08
    phase4_wrong_direction_penalty_gain: float = 2.5

    # timing / truncation
    max_steps: int = 900
    no_progress_limit: int = 80
    collision_grace_steps: int = 8

    # generic penalties
    step_penalty: float = -0.01
    action_penalty: float = -0.00015
    idle_penalty: float = -0.05
    out_of_bounds_penalty: float = -140.0
    backtrack_penalty: float = -0.40

    # encourage actual motion only when useful
    idle_cmd_ratio_thresh: float = 0.08
    idle_speed_thresh: float = 0.03

    # phase penalties
    lift_before_xy_penalty: float = -0.20
    move_after_xy_penalty: float = -0.65

    # route / heading shaping
    route_speed_gain: float = 0.16
    route_reverse_gain: float = 0.24
    seg_progress_gain0: float = 18.0
    seg_progress_gain1: float = 16.0
    seg_progress_gain2: float = 12.0
    z_progress_gain: float = 10.0
    heading_gain: float = 0.10
    heading_progress_gain: float = 0.18
    spin_penalty_gain: float = 0.02

    # zone rewards
    zone_bonus0: float = 7.0
    zone_bonus1: float = 6.0
    zone_bonus2: float = 0.0

    # once a closer distance level has been reached, same/farther levels become penalties
    zone_same_penalty0: float = -0.45
    zone_farther_penalty0: float = -0.85
    zone_same_penalty1: float = -0.35
    zone_farther_penalty1: float = -0.75
    outside_active_zone_penalty0: float = -0.10
    outside_active_zone_penalty1: float = -0.20
    wrong_aisle_penalty: float = -1.20

    # stricter phase-0 shaping: keep x near portal_x and mostly decrease |y-lane_y|
    phase0_route_speed_gain: float = 0.03
    phase0_route_reverse_gain: float = 0.45
    phase0_heading_gain: float = 0.00
    phase0_heading_progress_gain: float = 0.05
    phase0_y_progress_gain: float = 22.0
    phase0_x_dev_penalty_gain: float = 0.90
    phase0_x_dev_growth_penalty_gain: float = 2.40
    phase0_corridor_soft_half_width: float = 1.60
    phase0_corridor_hard_half_width: float = 3.00
    phase0_corridor_soft_penalty_gain: float = 1.25
    phase0_corridor_terminate_penalty: float = -90.0
    phase0_zone_bonus0: float = 10.0
    phase0_same_penalty0: float = -0.85
    phase0_farther_penalty0: float = -1.60
    phase0_outside_active_zone_penalty0: float = -0.35

    # forward-rotation driving policy bias for phase 0
    phase01_min_forward_speed: float = 0.70
    phase01_min_forward_raw_thresh: float = 0.08
    phase01_rotate_in_place_heading: float = 1.05
    phase01_caution_heading: float = 0.55
    phase01_rotate_scale: float = 0.12
    phase01_caution_scale: float = 0.45
    phase01_reverse_penalty_gain: float = -0.14
    phase01_reverse_scale: float = 0.45
    phase01_reverse_heading_allow: float = 1.45
    phase01_reverse_xdev_allow: float = 2.60

    # phase 1: suppress needless yaw, keep moving forward down the aisle centerline
    phase1_min_forward_speed: float = 1.05
    phase1_min_forward_raw_thresh: float = 0.05
    phase1_rotate_in_place_heading: float = 2.35
    phase1_caution_heading: float = 0.80
    phase1_rotate_scale: float = 0.10
    phase1_caution_scale: float = 0.78
    phase1_reverse_penalty_gain: float = -0.30
    phase1_reverse_scale: float = 0.08
    phase1_reverse_heading_allow: float = 2.75
    phase1_yaw_soft_cap_heading: float = 0.75
    phase1_yaw_soft_cap_scale: float = 0.35
    phase1_yaw_very_small_heading: float = 0.30
    phase1_yaw_very_small_scale: float = 0.18
    phase1_yaw_penalty_gain: float = 0.035
    phase1_yaw_flip_penalty: float = 0.45
    phase1_yaw_flip_min_mag: float = 0.55
    phase1_heading_gain: float = 0.02
    phase1_heading_progress_gain: float = 0.05
    phase1_route_speed_gain: float = 0.22
    phase1_route_reverse_gain: float = 0.55
    phase1_cross_track_gain: float = 0.90
    phase1_cross_track_progress_gain: float = 1.10
    phase1_cross_track_growth_penalty_gain: float = 1.60
    phase1_center_soft_half_width: float = 0.90
    phase1_center_hard_half_width: float = 1.80
    phase1_center_soft_penalty_gain: float = 1.60
    phase1_center_hard_penalty_gain: float = 3.50

    # phase 2: short-hold fine alignment near the target XY
    phase2_min_forward_speed: float = 0.20
    phase2_min_forward_raw_thresh: float = 0.08
    phase2_rotate_in_place_heading: float = 1.10
    phase2_caution_heading: float = 0.35
    phase2_rotate_scale: float = 0.10
    phase2_caution_scale: float = 0.40
    phase2_reverse_penalty_gain: float = -0.60
    phase2_yaw_penalty_gain: float = 0.020
    phase2_xy_progress_gain: float = 20.0

    # phase 3: rotate in place to face the rack before lifting
    phase3_yaw_progress_gain: float = 16.0
    phase3_yaw_penalty_gain: float = 0.015
    phase3_move_penalty_gain: float = -0.80
    phase3_lift_penalty_gain: float = -0.40

    # bonuses
    portal_bonus: float = 30.0
    aisle_bonus: float = 40.0
    xy_bonus: float = 80.0
    rotate_bonus: float = 60.0
    z_bonus: float = 100.0
    success_bonus: float = 220.0

    # failure penalties
    collision_penalty: float = -160.0
    timeout_penalty: float = -18.0
    stuck_penalty: float = -45.0

    # progress thresholds
    min_portal_progress: float = 0.02
    min_aisle_progress: float = 0.02
    min_xy_progress: float = 0.01
    min_z_progress: float = 0.001

    # bounds around work area
    x_limit_margin: float = 6.0
    y_limit_margin: float = 6.0

    # debugging / logging
    debug_every_steps: int = 50

    # gz pose stream
    gz_pose_topic: str = ""
    gz_pose_timeout: float = 8.0

    # multi-contact
    contact_topics: Tuple[str, ...] = (
        "/world/warehouse_world/model/forklift_simple/link/base_link/sensor/base_link_contact/contact",
        "/world/warehouse_world/model/forklift_simple/link/fork_carriage_link/sensor/fork_carriage_link_contact/contact",
        "/world/warehouse_world/model/forklift_simple/link/left_fork_link/sensor/left_fork_link_contact/contact",
        "/world/warehouse_world/model/forklift_simple/link/right_fork_link/sensor/right_fork_link_contact/contact",
    )
    contact_hold_sec: float = 0.18

    # geometry collision fallback
    warehouse_x: float = 100.0
    warehouse_y: float = 200.0
    wall_collision_margin: float = 0.15
    rack_collision_margin: float = 0.06
    base_half_len: float = 0.70
    base_half_wid: float = 0.50
    fork_tip_x: float = 1.45
    fork_y: float = 0.18

    # target marker
    spawn_target_marker: bool = True
    marker_name: str = "rl_target_marker"
    marker_pad_xy: float = 1.20
    marker_pole_xy: float = 0.08
    marker_top_xy: float = 0.35
    marker_top_z: float = 0.35


class SaveBestSuccessCallback(BaseCallback):
    def __init__(self, save_dir: str, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.check_freq = check_freq
        self.best_success = -1.0
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        if hasattr(self.model, "ep_success_buffer") and len(self.model.ep_success_buffer) > 0:
            mean_success = float(np.mean(self.model.ep_success_buffer))
            if mean_success > self.best_success:
                self.best_success = mean_success
                path = self.save_dir / "best_success_model"
                self.model.save(str(path))
                if self.verbose:
                    print(f"[Callback] New best success_rate={mean_success:.3f} -> saved to {path}.zip", flush=True)
        return True


class GzPoseStreamReader:
    def __init__(self, world_name: str, model_name: str, topic: Optional[str] = None):
        self.world_name = world_name
        self.model_name = model_name
        self.topic = topic or f"/world/{world_name}/dynamic_pose/info"
        self.proc: Optional[subprocess.Popen[str]] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.last_update = 0.0
        self.last_names: List[str] = []
        self.debug_names_printed = False

    def start(self):
        cmd = ["gz", "topic", "-e", "-t", self.topic]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def wait_for_pose(self, timeout: float = 8.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                ok = self.last_update > 0.0
            if ok:
                return
            time.sleep(0.05)
        raise RuntimeError(f"Timed out waiting for Gazebo world pose stream on {self.topic}")

    def get_pose(self) -> Dict[str, float]:
        with self.lock:
            return {
                "world_x": float(self.x),
                "world_y": float(self.y),
                "base_z": float(self.z),
                "world_yaw": float(self.yaw),
                "last_pose_time": float(self.last_update),
            }

    def _reader_loop(self):
        assert self.proc is not None and self.proc.stdout is not None
        brace_depth = 0
        buf: List[str] = []
        while not self.stop_event.is_set():
            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.02)
                continue
            if not line.strip():
                continue
            buf.append(line)
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0 and buf:
                msg = "".join(buf)
                buf = []
                brace_depth = 0
                self._parse_pose_message(msg)

    def _parse_pose_message(self, msg_text: str):
        pose_blocks = self._extract_blocks(msg_text, "pose")
        names_seen: List[str] = []
        best: Optional[Tuple[str, float, float, float, float]] = None
        for blk in pose_blocks:
            name = self._extract_string_field(blk, "name")
            if not name:
                continue
            names_seen.append(name)
            score = self._name_score(name)
            if score < 0:
                continue
            pos_blk = self._extract_first_block(blk, "position")
            ori_blk = self._extract_first_block(blk, "orientation")
            if pos_blk is None or ori_blk is None:
                continue
            x = self._extract_float_field(pos_blk, "x")
            y = self._extract_float_field(pos_blk, "y")
            z = self._extract_float_field(pos_blk, "z")
            qx = self._extract_float_field(ori_blk, "x")
            qy = self._extract_float_field(ori_blk, "y")
            qz = self._extract_float_field(ori_blk, "z")
            qw = self._extract_float_field(ori_blk, "w")
            if None in (x, y, z, qx, qy, qz, qw):
                continue
            yaw = yaw_from_quat(qx, qy, qz, qw)
            candidate = (name, x, y, z, yaw)
            if best is None or score > self._name_score(best[0]):
                best = candidate
        if names_seen:
            with self.lock:
                self.last_names = names_seen
            if not self.debug_names_printed:
                print(f"GZ_POSE_NAMES sample: {names_seen[:12]}", flush=True)
                self.debug_names_printed = True
        if best is None:
            return
        _, x, y, z, yaw = best
        with self.lock:
            self.x = x
            self.y = y
            self.z = z
            self.yaw = yaw
            self.last_update = time.time()

    def _name_score(self, name: str) -> int:
        if name == self.model_name:
            return 100
        if name == f"{self.model_name}::base_link":
            return 90
        if name.startswith(self.model_name + "::"):
            return 50
        return -1

    @staticmethod
    def _extract_blocks(text: str, field_name: str) -> List[str]:
        blocks: List[str] = []
        token = field_name + " {"
        i = 0
        while True:
            start = text.find(token, i)
            if start < 0:
                break
            brace_start = text.find("{", start)
            if brace_start < 0:
                break
            depth = 1
            j = brace_start + 1
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                blocks.append(text[start:j])
                i = j
            else:
                break
        return blocks

    @staticmethod
    def _extract_first_block(text: str, field_name: str) -> Optional[str]:
        blocks = GzPoseStreamReader._extract_blocks(text, field_name)
        return blocks[0] if blocks else None

    @staticmethod
    def _extract_string_field(text: str, field_name: str) -> Optional[str]:
        m = re.search(rf'{re.escape(field_name)}\s*:\s*"([^"]+)"', text)
        return m.group(1) if m else None

    @staticmethod
    def _extract_float_field(text: str, field_name: str) -> Optional[float]:
        m = re.search(rf'{re.escape(field_name)}\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', text)
        return float(m.group(1)) if m else None


class GazeboRosBridge(Node):
    def __init__(
        self,
        cfg: EnvConfig,
        odom_topic: Optional[str] = None,
        cmd_vel_topic: Optional[str] = None,
        lift_cmd_topic: Optional[str] = None,
        joint_topic: Optional[str] = None,
        contact_topics: Optional[List[str]] = None,
    ):
        super().__init__("forklift_rl_bridge")
        self.cfg = cfg
        self.odom_topic = odom_topic or f"/model/{cfg.model_name}/odometry"
        self.cmd_vel_topic = cmd_vel_topic or f"/model/{cfg.model_name}/cmd_vel"
        # train9: lift controller is force-command mode, so the Float64 value is force.
        self.lift_cmd_topic = lift_cmd_topic or f"/model/{cfg.model_name}/lift_force_cmd"
        self.joint_topic = joint_topic or f"/world/{cfg.world_name}/model/{cfg.model_name}/joint_state"
        self.contact_topics = contact_topics or list(cfg.contact_topics)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.lift_pub = self.create_publisher(Float64, self.lift_cmd_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._odom_cb, 10)
        self.joint_sub = self.create_subscription(JointState, self.joint_topic, self._joint_cb, 10)

        self.contact_subs = []
        self.contact_expire_times: Dict[str, float] = {}
        if HAS_CONTACTS:
            for topic in self.contact_topics:
                self.contact_expire_times[topic] = 0.0
                self.contact_subs.append(self.create_subscription(Contacts, topic, self._make_contact_cb(topic), 10))

        self.pose_reader = GzPoseStreamReader(cfg.world_name, cfg.model_name, cfg.gz_pose_topic or None)
        self.pose_reader.start()

        self.state_lock = threading.Lock()
        self.last_odom_time = 0.0
        self.last_joint_time = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.lift_z = 0.0
        self.lift_vel = 0.0

    def _make_contact_cb(self, topic: str):
        def _cb(msg):
            if len(msg.contacts) > 0:
                with self.state_lock:
                    self.contact_expire_times[topic] = time.time() + self.cfg.contact_hold_sec
        return _cb

    def _odom_cb(self, msg: Odometry):
        with self.state_lock:
            self.linear_vel = msg.twist.twist.linear.x
            self.angular_vel = msg.twist.twist.angular.z
            self.last_odom_time = time.time()

    def _joint_cb(self, msg: JointState):
        with self.state_lock:
            if "lift_joint" in msg.name:
                idx = msg.name.index("lift_joint")
                if idx < len(msg.position):
                    self.lift_z = msg.position[idx]
                if idx < len(msg.velocity):
                    self.lift_vel = msg.velocity[idx]
            self.last_joint_time = time.time()

    def clear_collision(self):
        with self.state_lock:
            for k in self.contact_expire_times:
                self.contact_expire_times[k] = 0.0

    def send_drive(self, v_cmd: float, yaw_cmd: float):
        twist = Twist()
        twist.linear.x = float(v_cmd)
        twist.angular.z = float(yaw_cmd)
        self.cmd_pub.publish(twist)

    def send_lift(self, lift_cmd: float):
        # lift_cmd is FORCE for /model/<name>/lift_force_cmd.
        # It is not velocity and not absolute position.
        lift = Float64()
        lift.data = float(lift_cmd)
        self.lift_pub.publish(lift)

    def send_action(self, v_cmd: float, yaw_cmd: float, lift_cmd: float):
        # Full combined publisher is kept for respawn self-test and emergency stop only.
        # During learning, step() uses phase-gated publishing instead.
        self.send_drive(v_cmd, yaw_cmd)
        self.send_lift(lift_cmd)

    def stop_drive(self):
        for _ in range(3):
            self.send_drive(0.0, 0.0)
            time.sleep(0.03)

    def hard_stop_drive(self, duration_sec: float = 0.80, dt_sec: float = 0.05) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Send repeated cmd_vel=(0,0) commands and spin so Gazebo receives them.

        Used before entering PHASE4. This does not allow driving in PHASE4; it only
        drains the remaining PHASE3 rotation / DiffDrive command state before lift.
        """
        before = self.get_state()
        t_end = time.time() + max(0.0, float(duration_sec))
        dt_sec = max(0.01, float(dt_sec))
        while time.time() < t_end:
            self.send_drive(0.0, 0.0)
            self._spin_for(min(dt_sec, max(0.0, t_end - time.time())), timeout_step=0.02)
        # A few final zeros make the stop command robust against bridge / scheduler timing.
        for _ in range(3):
            self.send_drive(0.0, 0.0)
            self._spin_for(0.03, timeout_step=0.02)
        after = self.get_state()
        return before, after

    def stop_lift(self):
        for _ in range(3):
            self.send_lift(0.0)
            time.sleep(0.03)

    def stop_all(self):
        for _ in range(3):
            self.send_action(0.0, 0.0, 0.0)
            time.sleep(0.03)

    def _spin_for(self, duration_sec: float, timeout_step: float = 0.05):
        t_end = time.time() + max(0.0, float(duration_sec))
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=min(timeout_step, max(0.0, t_end - time.time())))

    def _command_for(self, v_cmd: float, yaw_cmd: float, lift_cmd: float, duration_sec: float):
        self.send_action(v_cmd, yaw_cmd, lift_cmd)
        self._spin_for(duration_sec)
        self.stop()
        self._spin_for(self.cfg.respawn_self_test_settle_sec)
        return self.get_state()

    @staticmethod
    def _xy_delta(a: Dict[str, float], b: Dict[str, float]) -> float:
        return math.hypot(float(b["x"] - a["x"]), float(b["y"] - a["y"]))

    def run_respawn_self_test(self, tag: str = "respawn") -> bool:
        start = self.get_state()
        print(
            f"RESPAWN_SELFTEST_START tag={tag} cmd_vel_topic={self.cmd_vel_topic} lift_topic={self.lift_cmd_topic} "
            f"x={start['x']:.3f} y={start['y']:.3f} yaw={start['yaw']:.3f} lift_z={start['lift_z']:.3f}",
            flush=True,
        )

        fwd = self._command_for(+self.cfg.respawn_self_test_drive_v, 0.0, 0.0, self.cfg.respawn_self_test_drive_sec)
        back = self._command_for(-self.cfg.respawn_self_test_drive_v, 0.0, 0.0, self.cfg.respawn_self_test_drive_sec)
        up = self._command_for(0.0, 0.0, +self.cfg.respawn_self_test_lift_force, self.cfg.respawn_self_test_lift_sec)
        down = self._command_for(0.0, 0.0, -self.cfg.respawn_self_test_lift_force, self.cfg.respawn_self_test_lift_sec)
        self.stop()
        self._spin_for(self.cfg.respawn_self_test_settle_sec)
        final = self.get_state()

        move_forward_delta = self._xy_delta(start, fwd)
        move_return_delta = self._xy_delta(start, back)
        lift_up_delta = float(up["lift_z"] - start["lift_z"])
        lift_return_delta = abs(float(down["lift_z"] - start["lift_z"]))

        drive_ok = move_forward_delta >= self.cfg.respawn_self_test_min_move_delta
        drive_return_ok = move_return_delta <= self.cfg.respawn_self_test_max_return_xy
        lift_up_ok = lift_up_delta >= self.cfg.respawn_self_test_min_lift_delta
        lift_down_ok = (lift_return_delta <= self.cfg.respawn_self_test_max_return_lift) or (down["lift_z"] < up["lift_z"])
        if self.cfg.respawn_self_test_require_lift_down:
            ok = bool(drive_ok and drive_return_ok and lift_up_ok and lift_down_ok)
        else:
            ok = bool(drive_ok and drive_return_ok and lift_up_ok)

        print(
            f"RESPAWN_SELFTEST_RESULT tag={tag} ok={ok} require_lift_down={self.cfg.respawn_self_test_require_lift_down} "
            f"drive_ok={drive_ok} drive_return_ok={drive_return_ok} lift_up_ok={lift_up_ok} lift_down_ok={lift_down_ok} "
            f"move_forward_delta={move_forward_delta:.4f} move_return_delta={move_return_delta:.4f} "
            f"lift_up_delta={lift_up_delta:.4f} lift_return_delta={lift_return_delta:.4f} "
            f"start=({start['x']:.3f},{start['y']:.3f},{start['yaw']:.3f},lift={start['lift_z']:.3f}) "
            f"fwd=({fwd['x']:.3f},{fwd['y']:.3f},lift={fwd['lift_z']:.3f}) "
            f"back=({back['x']:.3f},{back['y']:.3f},lift={back['lift_z']:.3f}) "
            f"up=({up['x']:.3f},{up['y']:.3f},lift={up['lift_z']:.3f}) "
            f"down=({down['x']:.3f},{down['y']:.3f},lift={down['lift_z']:.3f}) "
            f"final=({final['x']:.3f},{final['y']:.3f},lift={final['lift_z']:.3f})",
            flush=True,
        )
        if (not lift_down_ok) and (not self.cfg.respawn_self_test_require_lift_down):
            print(
                f"RESPAWN_SELFTEST_WARN tag={tag} lift_down_ok=False but continuing because require_lift_down=False; clean respawn will reset lift before training",
                flush=True,
            )
        return ok

    def stop(self):
        self.stop_all()

    def get_state(self) -> Dict[str, float]:
        pose = self.pose_reader.get_pose()
        with self.state_lock:
            now = time.time()
            collision = any(now < exp for exp in self.contact_expire_times.values())
            return {
                "x": float(pose["world_x"]),
                "y": float(pose["world_y"]),
                "base_z": float(pose["base_z"]),
                "yaw": float(pose["world_yaw"]),
                "linear_vel": float(self.linear_vel),
                "angular_vel": float(self.angular_vel),
                "lift_z": float(self.lift_z),
                "lift_vel": float(self.lift_vel),
                "collision": bool(collision),
                "last_pose_time": float(pose["last_pose_time"]),
            }

    def wait_for_topics(self, timeout: float = 10.0):
        self.pose_reader.wait_for_pose(timeout=self.cfg.gz_pose_timeout)
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.state_lock:
                odom_ok = self.last_odom_time > 0.0
                joint_ok = self.last_joint_time > 0.0
            if odom_ok and joint_ok:
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError(f"topic wait timeout. odom={self.odom_topic}, joint={self.joint_topic}")

    def reset_world(self):
        self.stop()
        self.clear_collision()
        cmd = [
            "gz", "service", "-s", f"/world/{self.cfg.world_name}/control",
            "--reqtype", "gz.msgs.WorldControl", "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000", "--req", "reset: {all: true}, pause: false",
        ]
        try:
            res = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"WARN reset_world rc={res.returncode} stdout={res.stdout.strip()} stderr={res.stderr.strip()}", flush=True)
        except Exception as e:
            print(f"WARN reset_world exception: {e}", flush=True)
        time.sleep(0.8)
        self.stop()
        self.clear_collision()

    def _remove_model(self, name: str):
        cmd = [
            "gz", "service", "-s", f"/world/{self.cfg.world_name}/remove",
            "--reqtype", "gz.msgs.Entity", "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000", "--req", f'name: "{name}", type: MODEL',
        ]
        try:
            subprocess.run(cmd, check=False, capture_output=True, text=True)
        except Exception:
            pass

    def _resolve_model_sdf(self) -> Path:
        candidates = []
        if self.cfg.model_sdf:
            candidates.append(Path(self.cfg.model_sdf))
        script_dir = Path(__file__).resolve().parent
        candidates.extend([
            Path.cwd() / "models" / self.cfg.model_name / "model.sdf",
            script_dir / "models" / self.cfg.model_name / "model.sdf",
            script_dir.parent / "models" / self.cfg.model_name / "model.sdf",
        ])
        for cand in candidates:
            if cand.exists():
                return cand.resolve()
        raise FileNotFoundError(
            f"Could not find model.sdf for {self.cfg.model_name}. Use --model-sdf to specify it explicitly."
        )

    def _spawn_model(self, sdf_path: Path, x: float, y: float, yaw: float, z: float):
        qx, qy, qz, qw = quat_from_yaw(yaw)
        req = (
            f'sdf_filename: "{sdf_path}", name: "{self.cfg.model_name}", ' 
            f'pose: {{position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}}, ' 
            f'orientation: {{x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}, w: {qw:.6f}}}}}'
        )
        cmd = [
            "gz", "service", "-s", f"/world/{self.cfg.world_name}/create",
            "--reqtype", "gz.msgs.EntityFactory", "--reptype", "gz.msgs.Boolean",
            "--timeout", "5000", "--req", req,
        ]
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"spawn rc={res.returncode} stdout={res.stdout.strip()} stderr={res.stderr.strip()}"
            )

    def _wait_for_spawn_pose(self, target_x: float, target_y: float, target_yaw: float, timeout: float = 8.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            s = self.get_state()
            dxy = math.hypot(s["x"] - target_x, s["y"] - target_y)
            dyaw = abs(wrap_to_pi(s["yaw"] - target_yaw))
            if dxy <= self.cfg.spawn_pose_tol_xy and dyaw <= self.cfg.spawn_pose_tol_yaw and self.last_odom_time > 0.0 and self.last_joint_time > 0.0:
                return s
        s = self.get_state()
        raise RuntimeError(
            f"spawn pose mismatch actual=({s['x']:.3f},{s['y']:.3f},{s['yaw']:.3f}) expected=({target_x:.3f},{target_y:.3f},{target_yaw:.3f})"
        )

    def respawn_model(self, x: float, y: float, yaw: float, z: Optional[float] = None):
        z = self.cfg.spawn_z if z is None else z
        self.stop()
        self.clear_collision()
        with self.state_lock:
            self.last_odom_time = 0.0
            self.last_joint_time = 0.0

        sdf_path = self._resolve_model_sdf()
        self._remove_model(self.cfg.model_name)
        time.sleep(self.cfg.respawn_remove_wait_sec)
        self._spawn_model(sdf_path, x, y, yaw, z)
        time.sleep(self.cfg.respawn_settle_sec)
        s = self._wait_for_spawn_pose(x, y, yaw, timeout=max(8.0, self.cfg.gz_pose_timeout))
        self.stop()
        self.clear_collision()

        if self.cfg.respawn_self_test:
            test_ok = self.run_respawn_self_test(tag=f"{self.cfg.model_name}@{time.strftime('%H%M%S')}")
            if (not test_ok) and self.cfg.respawn_self_test_required:
                raise RuntimeError("RESPAWN_SELFTEST failed; refusing to start training with a broken drive/lift setup")

            # The self-test intentionally moves the model. Respawn once more so each episode
            # starts from the exact configured pose and lift lower position.
            self._remove_model(self.cfg.model_name)
            time.sleep(self.cfg.respawn_remove_wait_sec)
            with self.state_lock:
                self.last_odom_time = 0.0
                self.last_joint_time = 0.0
            self._spawn_model(sdf_path, x, y, yaw, z)
            time.sleep(self.cfg.respawn_settle_sec)
            s = self._wait_for_spawn_pose(x, y, yaw, timeout=max(8.0, self.cfg.gz_pose_timeout))
            self.stop()
            self.clear_collision()
            print(
                f"RESPAWN_SELFTEST_CLEAN_START model={self.cfg.model_name} x={s['x']:.3f} y={s['y']:.3f} yaw={s['yaw']:.3f} lift_z={s['lift_z']:.3f}",
                flush=True,
            )

        hit, kind = geometry_collision_for_pose(self.cfg, s["x"], s["y"], s["yaw"])
        if hit:
            raise RuntimeError(
                f"Spawn pose is colliding. requested=({x:.3f},{y:.3f},{yaw:.3f}) actual=({s['x']:.3f},{s['y']:.3f},{s['yaw']:.3f}) kind={kind}"
            )
        print(
            f"RESPAWN_OK model={self.cfg.model_name} sdf={sdf_path} x={s['x']:.3f} y={s['y']:.3f} yaw={s['yaw']:.3f}",
            flush=True,
        )
        return s

    def close(self):
        self.pose_reader.stop()


class ForkliftRouteEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, bridge: GazeboRosBridge, cfg: EnvConfig):
        super().__init__()
        self.bridge = bridge
        self.cfg = cfg
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Original train11 observation had 24 dimensions.
        # train12 appends 34 goal-conditioning features: discrete target ids,
        # desired XYZ, decomposed waypoints, target yaw, relative errors, and phase one-hot.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(58,), dtype=np.float32)

        self.phase = 0
        self.portal_reached = False
        self.aisle_reached = False
        self.xy_reached = False
        self.rot_reached = False
        self.z_reached = False
        self.steps = 0
        self.no_progress_steps = 0
        self.episode_idx = 0
        self.last_cmd = {"v_cmd": 0.0, "yaw_cmd": 0.0, "lift_cmd": 0.0}
        self.plan: Optional[Dict[str, Any]] = None

        self.prev_segment_dist: Optional[float] = None
        self.prev_route_remaining: Optional[float] = None
        self.prev_z_err: Optional[float] = None
        self.prev_head_abs: Optional[float] = None
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_phase0_y_err: Optional[float] = None
        self.prev_phase0_x_dev: Optional[float] = None
        self.prev_phase1_lane_err: Optional[float] = None
        self.prev_xy_eval_dist: Optional[float] = None
        self.prev_face_yaw_abs: Optional[float] = None

        self.zone_best0 = -1
        self.zone_best1 = -1
        self.marker_mgr = TargetMarkerManager(cfg.world_name)
        self.phase_profile_logged = False

    def _build_plan(self, s: Dict[str, float]) -> Dict[str, Any]:
        if self.cfg.use_target_map:
            return self.cfg.map_cfg.build_plan(
                row=self.cfg.target_row,
                bin_idx=self.cfg.target_bin,
                level=self.cfg.target_level,
                start_x=s["x"],
                start_y=s["y"],
                face=self.cfg.target_face,
            )
        portal_x = (1.0 if s["x"] >= 0 else -1.0) * (self.cfg.map_cfg.rack_x_max + self.cfg.map_cfg.portal_margin)
        wp0 = (portal_x, self.cfg.goal_y)
        wp1 = (self.cfg.goal_x, self.cfg.goal_y)
        return {
            "row": -1, "bin": -1, "level": -1, "face": "explicit_xyz",
            "lane_idx": -1, "lane_y": self.cfg.goal_y,
            "goal_x": self.cfg.goal_x, "goal_y": self.cfg.goal_y, "goal_z": self.cfg.goal_z,
            "target_yaw": -math.pi / 2.0,
            "portal_x": portal_x, "wp0": wp0, "wp1": wp1,
            "zones0": self.cfg.map_cfg._build_segment_zones(0, (s["x"], s["y"]), wp0, self.cfg.map_cfg.phase0_zone_count, self.cfg.map_cfg.phase0_half_width_x),
            "zones1": self.cfg.map_cfg._build_segment_zones(1, wp0, wp1, self.cfg.map_cfg.phase1_zone_count, self.cfg.map_cfg.phase1_half_width_y),
        }

    def _phase_profile(self, phase: int) -> Dict[str, float]:
        if phase == 0:
            return {
                "control_dt": self.cfg.phase0_control_dt,
                "action_repeat": float(self.cfg.phase0_action_repeat),
                "max_v": self.cfg.phase0_max_v,
                "max_yaw_rate": self.cfg.phase0_max_yaw_rate,
                "max_lift_rate": self.cfg.max_lift_rate,
                "max_lift_force": 0.0,
            }
        if phase == 1:
            return {
                "control_dt": self.cfg.phase1_control_dt,
                "action_repeat": float(self.cfg.phase1_action_repeat),
                "max_v": self.cfg.phase1_max_v,
                "max_yaw_rate": self.cfg.phase1_max_yaw_rate,
                "max_lift_rate": self.cfg.max_lift_rate,
                "max_lift_force": 0.0,
            }
        if phase == 2:
            return {
                "control_dt": self.cfg.phase2_control_dt,
                "action_repeat": float(self.cfg.phase2_action_repeat),
                "max_v": self.cfg.phase2_max_v,
                "max_yaw_rate": self.cfg.phase2_max_yaw_rate,
                "max_lift_rate": self.cfg.max_lift_rate,
                "max_lift_force": 0.0,
            }
        if phase == 3:
            return {
                "control_dt": self.cfg.phase3_control_dt,
                "action_repeat": float(self.cfg.phase3_action_repeat),
                "max_v": 0.0,
                "max_yaw_rate": self.cfg.phase3_max_yaw_rate,
                "max_lift_rate": 0.0,
                "max_lift_force": 0.0,
            }
        return {
            "control_dt": self.cfg.phase4_control_dt,
            "action_repeat": float(self.cfg.phase4_action_repeat),
            "max_v": 0.0,
            "max_yaw_rate": 0.0,
            "max_lift_rate": self.cfg.phase4_max_lift_rate,
            "max_lift_force": self.cfg.phase4_max_lift_force,
        }

    def _goal_reference_distances(self, s: Dict[str, float]) -> Dict[str, float]:
        assert self.plan is not None
        gx, gy = self.plan["goal_x"], self.plan["goal_y"]
        fx, fy = self._local_to_world(s, self.cfg.base_half_len, 0.0)
        kx, ky = self._local_to_world(s, self.cfg.fork_tip_x, 0.0)
        return {
            "base": point_dist(s["x"], s["y"], gx, gy),
            "front": point_dist(fx, fy, gx, gy),
            "fork_mid": point_dist(kx, ky, gx, gy),
        }

    def _xy_eval_dist(self, s: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        dists = self._goal_reference_distances(s)
        mode = self.cfg.xy_reference_point
        if mode not in dists:
            mode = "base"
        return mode, dists[mode], dists

    def _face_yaw_err(self, s: Dict[str, float]) -> float:
        assert self.plan is not None
        return wrap_to_pi(self.plan["target_yaw"] - s["yaw"])

    def _local_to_world(self, s: Dict[str, float], lx: float, ly: float) -> Tuple[float, float]:
        c = math.cos(s["yaw"])
        ss = math.sin(s["yaw"])
        return s["x"] + c * lx - ss * ly, s["y"] + ss * lx + c * ly

    def _sample_collision_points(self, s: Dict[str, float]) -> List[Tuple[str, float, float]]:
        pts_local = [
            ("rear_center", -self.cfg.base_half_len, 0.0),
            ("front_center", self.cfg.base_half_len, 0.0),
            ("rear_left", -self.cfg.base_half_len, self.cfg.base_half_wid),
            ("rear_right", -self.cfg.base_half_len, -self.cfg.base_half_wid),
            ("front_left", self.cfg.base_half_len, self.cfg.base_half_wid),
            ("front_right", self.cfg.base_half_len, -self.cfg.base_half_wid),
            ("fork_left", self.cfg.fork_tip_x, self.cfg.fork_y),
            ("fork_right", self.cfg.fork_tip_x, -self.cfg.fork_y),
        ]
        return [(name, *self._local_to_world(s, lx, ly)) for name, lx, ly in pts_local]

    def _wall_geometry_collision(self, s: Dict[str, float]) -> Tuple[bool, str]:
        half_x = self.cfg.warehouse_x / 2.0 - self.cfg.wall_collision_margin
        half_y = self.cfg.warehouse_y / 2.0 - self.cfg.wall_collision_margin
        for name, px, py in self._sample_collision_points(s):
            if abs(px) >= half_x or abs(py) >= half_y:
                return True, f"wall:{name}"
        return False, ""

    def _rack_geometry_collision(self, s: Dict[str, float]) -> Tuple[bool, str]:
        m = self.cfg.rack_collision_margin
        xmin = self.cfg.map_cfg.rack_x_min - m
        xmax = self.cfg.map_cfg.rack_x_max + m
        half_w = self.cfg.map_cfg.rack_wid / 2.0 + m
        for row_idx, row_y in enumerate(self.cfg.map_cfg.row_y_centers):
            ymin = row_y - half_w
            ymax = row_y + half_w
            for name, px, py in self._sample_collision_points(s):
                if xmin <= px <= xmax and ymin <= py <= ymax:
                    return True, f"rack{row_idx}:{name}"
        return False, ""

    def _geometry_collision(self, s: Dict[str, float]) -> Tuple[bool, str]:
        hit, kind = self._wall_geometry_collision(s)
        if hit:
            return True, kind
        hit, kind = self._rack_geometry_collision(s)
        if hit:
            return True, kind
        return False, ""

    def _active_xy_target(self) -> Tuple[float, float]:
        assert self.plan is not None
        if self.phase == 0:
            return self.plan["wp0"]
        return self.plan["wp1"]

    def _segment_distance(self, s: Dict[str, float], phase: int) -> float:
        assert self.plan is not None
        if phase == 0:
            return point_dist(s["x"], s["y"], *self.plan["wp0"])
        return point_dist(s["x"], s["y"], *self.plan["wp1"])

    def _route_remaining(self, s: Dict[str, float]) -> float:
        assert self.plan is not None
        d0 = point_dist(s["x"], s["y"], *self.plan["wp0"])
        d1 = point_dist(*self.plan["wp0"], *self.plan["wp1"])
        d2 = point_dist(s["x"], s["y"], *self.plan["wp1"])
        return d0 + d1 if self.phase == 0 else d2

    def _stopped(self, s: Dict[str, float]) -> bool:
        return (
            abs(s["linear_vel"]) < self.cfg.stop_v_thresh
            and abs(s["angular_vel"]) < self.cfg.stop_yaw_thresh
            and abs(s["lift_vel"]) < self.cfg.stop_lift_thresh
        )

    def _head_err(self, s: Dict[str, float]) -> float:
        tx, ty = self._active_xy_target()
        bearing = math.atan2(ty - s["y"], tx - s["x"])
        return wrap_to_pi(bearing - s["yaw"])

    def _route_speed(self, s: Dict[str, float], dt: float) -> float:
        tx, ty = self._active_xy_target()
        vx = (s["x"] - self.prev_x) / max(dt, 1e-6)
        vy = (s["y"] - self.prev_y) / max(dt, 1e-6)
        dx = tx - self.prev_x
        dy = ty - self.prev_y
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return 0.0
        ux, uy = dx / n, dy / n
        return vx * ux + vy * uy

    def _phase_zone_progress(self, s: Dict[str, float]) -> Tuple[int, bool, bool]:
        assert self.plan is not None
        if self.phase == 0:
            zones = self.plan["zones0"]
            best = self.zone_best0
        elif self.phase == 1:
            zones = self.plan["zones1"]
            best = self.zone_best1
        else:
            return -1, False, False

        idx = -1
        for z in zones:
            if z.contains(s["x"], s["y"]):
                idx = z.idx
                break
        entered_new = False
        backtracked = False
        if idx >= 0 and idx > best:
            entered_new = True
            best = idx
        if idx >= 0 and best >= 1 and idx < best - 1:
            backtracked = True

        if self.phase == 0:
            self.zone_best0 = best
        elif self.phase == 1:
            self.zone_best1 = best
        return idx, entered_new, backtracked

    def _within_bounds(self, s: Dict[str, float]) -> bool:
        assert self.plan is not None
        x_lim = self.cfg.map_cfg.rack_x_max + self.cfg.map_cfg.portal_margin + self.cfg.x_limit_margin
        # Bound Y around useful lanes rather than the entire warehouse
        useful_lanes = list(self.cfg.map_cfg.lane_y_centers)
        y_min = min(useful_lanes) - self.cfg.y_limit_margin
        y_max = max(useful_lanes) + self.cfg.y_limit_margin
        return (-x_lim <= s["x"] <= x_lim) and (y_min <= s["y"] <= y_max)

    def _nearest_lane_idx(self, y: float) -> int:
        lanes = self.cfg.map_cfg.lane_y_centers
        return int(min(range(len(lanes)), key=lambda i: abs(y - lanes[i])))

    def _distance_level_penalty(self, phase: int, zone_idx: int) -> float:
        if phase == 0:
            best = self.zone_best0
            same_pen = self.cfg.phase0_same_penalty0
            farther_pen = self.cfg.phase0_farther_penalty0
            outside_pen = self.cfg.phase0_outside_active_zone_penalty0
        elif phase == 1:
            best = self.zone_best1
            same_pen = self.cfg.zone_same_penalty1
            farther_pen = self.cfg.zone_farther_penalty1
            outside_pen = self.cfg.outside_active_zone_penalty1
        else:
            return 0.0

        if zone_idx < 0:
            return outside_pen if best >= 0 else 0.0

        if best < 0:
            return 0.0

        if zone_idx == best:
            return same_pen
        if zone_idx < best:
            return farther_pen * float(best - zone_idx)
        return 0.0

    def _phase0_lane_y_err(self, s: Dict[str, float]) -> float:
        assert self.plan is not None
        return abs(s["y"] - self.plan["lane_y"])

    def _phase0_x_dev(self, s: Dict[str, float]) -> float:
        assert self.plan is not None
        return abs(s["x"] - self.plan["portal_x"])

    def _phase1_lane_err(self, s: Dict[str, float]) -> float:
        assert self.plan is not None
        return abs(s["y"] - self.plan["lane_y"])

    def _phase0_corridor_penalty_and_terminate(self, s: Dict[str, float]) -> Tuple[float, bool]:
        if self.phase != 0:
            return 0.0, False
        x_dev = self._phase0_x_dev(s)
        if x_dev > self.cfg.phase0_corridor_hard_half_width:
            return self.cfg.phase0_corridor_terminate_penalty, True
        if x_dev > self.cfg.phase0_corridor_soft_half_width:
            exceed = x_dev - self.cfg.phase0_corridor_soft_half_width
            return -self.cfg.phase0_corridor_soft_penalty_gain * exceed, False
        return 0.0, False

    def _wrong_aisle_penalty_now(self, s: Dict[str, float]) -> float:
        if self.phase not in (1, 2):
            return 0.0
        assert self.plan is not None
        # only care once the robot is inside the rack-length region
        if not (self.cfg.map_cfg.rack_x_min - 1.0 <= s["x"] <= self.cfg.map_cfg.rack_x_max + 1.0):
            return 0.0
        nearest_lane = self._nearest_lane_idx(s["y"])
        return self.cfg.wrong_aisle_penalty if nearest_lane != self.plan["lane_idx"] else 0.0

    def _sample_target_for_episode(self) -> None:
        if not self.cfg.randomize_target:
            return
        # self.np_random is initialized by gym.Env.reset(seed=...).
        self.cfg.target_row = int(self.np_random.choice(np.array(self.cfg.target_rows, dtype=np.int64)))
        self.cfg.target_bin = int(self.np_random.choice(np.array(self.cfg.target_bins, dtype=np.int64)))
        self.cfg.target_level = int(self.np_random.choice(np.array(self.cfg.target_levels, dtype=np.int64)))

    def _goal_condition_features(self, s: Dict[str, float], tx: float, ty: float, d_portal: float, d_goal: float) -> List[float]:
        assert self.plan is not None
        mc = self.cfg.map_cfg
        max_row = max(1, mc.num_rows - 1)
        max_bin = max(1, mc.num_bins - 1)
        max_level = max(1, len(mc.shelf_zs) - 1)
        gx = float(self.plan["goal_x"])
        gy = float(self.plan["goal_y"])
        gz = float(self.plan["goal_z"])
        wp0x, wp0y = self.plan["wp0"]
        wp1x, wp1y = self.plan["wp1"]
        tyaw = float(self.plan["target_yaw"])
        head_err = self._head_err(s)
        face_yaw_err = self._face_yaw_err(s)
        route_remaining = self._route_remaining(s)

        sx = max(1.0, self.cfg.warehouse_x / 2.0)
        sy = max(1.0, self.cfg.warehouse_y / 2.0)
        sz = max(1.0, max(self.cfg.lift_max, max(mc.level_target_zs)))
        sd = max(1.0, self.cfg.warehouse_x + self.cfg.warehouse_y)

        phase_oh = [1.0 if self.phase == i else 0.0 for i in range(5)]
        return [
            float(self.plan["row"]) / max_row,
            float(self.plan["bin"]) / max_bin,
            float(self.plan["level"]) / max_level,
            gx / sx, gy / sy, gz / sz,
            wp0x / sx, wp0y / sy, wp1x / sx, wp1y / sy,
            math.cos(tyaw), math.sin(tyaw),
            (gx - s["x"]) / sx, (gy - s["y"]) / sy, (gz - s["lift_z"]) / sz,
            (wp0x - s["x"]) / sx, (wp0y - s["y"]) / sy,
            (wp1x - s["x"]) / sx, (wp1y - s["y"]) / sy,
            d_portal / sd, d_goal / sd, route_remaining / sd,
            head_err / math.pi, face_yaw_err / math.pi,
            float(self.plan["lane_y"]) / sy, float(self.plan["portal_x"]) / sx,
            s["x"] / sx, s["y"] / sy, s["lift_z"] / sz,
            *phase_oh,
        ]

    def _get_obs(self, s: Dict[str, float]) -> np.ndarray:
        assert self.plan is not None
        tx, ty = self._active_xy_target()
        dx = self.plan["goal_x"] - s["x"]
        dy = self.plan["goal_y"] - s["y"]
        dz = self.plan["goal_z"] - s["lift_z"]
        d_portal = point_dist(s["x"], s["y"], *self.plan["wp0"])
        d_goal = point_dist(s["x"], s["y"], *self.plan["wp1"])
        base_obs = [
            dx, dy, math.cos(s["yaw"]), math.sin(s["yaw"]),
            s["linear_vel"], s["angular_vel"], s["base_z"], s["lift_z"], s["lift_vel"], dz,
            float(self.phase),
            d_portal, d_goal,
            self.plan["wp0"][0] - s["x"], self.plan["wp0"][1] - s["y"],
            1.0 if self.portal_reached else 0.0,
            1.0 if self.aisle_reached else 0.0,
            self._route_remaining(s),
            tx - s["x"], ty - s["y"],
            self._head_err(s),
            float(self.zone_best0),
            float(self.zone_best1),
            1.0 if self._within_bounds(s) else 0.0,
        ]
        goal_obs = self._goal_condition_features(s, tx, ty, d_portal, d_goal)
        obs = np.array(base_obs + goal_obs, dtype=np.float32)
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(f"Observation shape mismatch: got {obs.shape}, expected {self.observation_space.shape}")
        return obs

    def _log_phase_change(self, old_phase: int, new_phase: int, s: Dict[str, float], seg_dist: float, z_err: float):
        print(
            f"PHASE_CHANGE ep={self.episode_idx} step={self.steps} {PHASE_NAMES.get(old_phase, old_phase)}->{PHASE_NAMES.get(new_phase, new_phase)} "
            f"x={s['x']:.3f} y={s['y']:.3f} base_z={s['base_z']:.3f} lift_z={s['lift_z']:.3f} seg_dist={seg_dist:.3f} z_err={z_err:.3f}",
            flush=True,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        try:
            s = self.bridge.respawn_model(
                x=self.cfg.spawn_x,
                y=self.cfg.spawn_y,
                yaw=self.cfg.spawn_yaw,
                z=self.cfg.spawn_z,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to respawn {self.cfg.model_name} at configured spawn pose: {e}") from e
        s["lift_z"] = float(np.clip(s["lift_z"], self.cfg.lift_min, self.cfg.lift_max))
        self._sample_target_for_episode()
        print(
            f"RESET_CHECK ep={self.episode_idx + 1} x={s['x']:.3f} y={s['y']:.3f} base_z={s['base_z']:.3f} lift_z={s['lift_z']:.3f} yaw={s['yaw']:.3f} "
            f"target=({self.cfg.target_row},{self.cfg.target_bin},{self.cfg.target_level}) randomize_target={self.cfg.randomize_target}",
            flush=True,
        )
        self.plan = self._build_plan(s)
        if self.cfg.spawn_target_marker:
            self.marker_mgr.ensure(
                self.plan["goal_x"], self.plan["goal_y"], self.plan["goal_z"],
                pad_xy=self.cfg.marker_pad_xy, pole_xy=self.cfg.marker_pole_xy,
                top_xy=self.cfg.marker_top_xy, top_z=self.cfg.marker_top_z, force=True,
            )

        self.phase = 0
        self.portal_reached = False
        self.aisle_reached = False
        self.xy_reached = False
        self.rot_reached = False
        self.z_reached = False
        self.steps = 0
        self.no_progress_steps = 0
        self.episode_idx += 1
        self.last_cmd = {"v_cmd": 0.0, "yaw_cmd": 0.0, "lift_cmd": 0.0}

        self.prev_segment_dist = self._segment_distance(s, self.phase)
        self.prev_route_remaining = self._route_remaining(s)
        self.prev_z_err = abs(self.plan["goal_z"] - s["lift_z"])
        self.prev_head_abs = abs(self._head_err(s))
        self.prev_x = s["x"]
        self.prev_y = s["y"]
        self.prev_phase0_y_err = self._phase0_lane_y_err(s)
        self.prev_phase0_x_dev = self._phase0_x_dev(s)
        self.prev_phase1_lane_err = self._phase1_lane_err(s)
        self.prev_xy_eval_dist = self._xy_eval_dist(s)[1]
        self.prev_face_yaw_abs = abs(self._face_yaw_err(s))

        self.zone_best0 = -1
        self.zone_best1 = -1

        print(
            f"PLAN ep={self.episode_idx} row={self.plan['row']} bin={self.plan['bin']} level={self.plan['level']} face={self.plan['face']} "
            f"wp0=({self.plan['wp0'][0]:.3f},{self.plan['wp0'][1]:.3f}) wp1=({self.plan['wp1'][0]:.3f},{self.plan['wp1'][1]:.3f}) "
            f"goal_z={self.plan['goal_z']:.3f} zones0={len(self.plan['zones0'])} zones1={len(self.plan['zones1'])}",
            flush=True,
        )
        print(
            f"EP_START ep={self.episode_idx} phase={PHASE_NAMES[self.phase]} x={s['x']:.3f} y={s['y']:.3f} base_z={s['base_z']:.3f} lift_z={s['lift_z']:.3f} "
            f"dx={self.plan['goal_x'] - s['x']:.3f} dy={self.plan['goal_y'] - s['y']:.3f} dz={self.plan['goal_z'] - s['lift_z']:.3f}",
            flush=True,
        )
        print(
            f"TARGET_XYZ goal=({self.plan['goal_x']:.3f}, {self.plan['goal_y']:.3f}, {self.plan['goal_z']:.3f}) start_xy=({s['x']:.3f}, {s['y']:.3f}) portal=({self.plan['wp0'][0]:.3f}, {self.plan['wp0'][1]:.3f}) target_yaw={self.plan['target_yaw']:.3f}",
            flush=True,
        )
        if not self.phase_profile_logged:
            for ph in range(5):
                prof = self._phase_profile(ph)
                hold = prof["control_dt"] * prof["action_repeat"]
                if ph == 4:
                    extra = f"max_lift_force={prof['max_lift_force']:.3f}"
                elif ph == 3:
                    extra = f"max_yaw={prof['max_yaw_rate']:.3f}"
                else:
                    extra = f"max_v={prof['max_v']:.3f} max_yaw={prof['max_yaw_rate']:.3f}"
                print(
                    f"PHASE_CFG phase={PHASE_NAMES[ph]} control_dt={prof['control_dt']:.3f} action_repeat={int(prof['action_repeat'])} hold={hold:.3f} {extra}",
                    flush=True,
                )
            print(f"XY_EVAL_CFG mode={self.cfg.xy_reference_point} xy_tol={self.cfg.xy_tol:.3f} yaw_tol={self.cfg.yaw_tol:.3f} wp1_tol={self.cfg.map_cfg.wp1_tol:.3f}", flush=True)
            self.phase_profile_logged = True
        return self._get_obs(s), {"phase": self.phase}

    def step(self, action: np.ndarray):
        assert self.plan is not None
        self.steps += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        raw_v = float(action[0])
        raw_yaw = float(action[1])
        raw_lift = float(action[2])
        old_phase = self.phase

        prev_cmd = dict(self.last_cmd)
        pre_s = self.bridge.get_state()
        pre_head_err = self._head_err(pre_s)
        pre_phase0_xdev = self._phase0_x_dev(pre_s) if self.phase == 0 and self.plan is not None else 0.0
        pre_phase1_lane_err = self._phase1_lane_err(pre_s) if self.phase == 1 and self.plan is not None else 0.0
        phase_profile = self._phase_profile(self.phase)

        if self.phase in (0, 1, 2, 3):
            v_cmd = raw_v * phase_profile["max_v"]
            yaw_cmd = raw_yaw * phase_profile["max_yaw_rate"]
            # PHASE0-3 action mask:
            #   - action[0]/action[1] are used for cmd_vel
            #   - action[2] is NOT published to lift_force_cmd
            #   - raw_lift is still penalized in reward, so the policy learns not to ask for lift here
            # Important: do not send fake negative force here; if lift drifts, fix the model/controller.
            lift_cmd = 0.0
            publish_drive = True
            publish_lift = False
            command_mask = "drive_only_no_lift_publish"

            if self.phase == 0:
                # Prefer forward + rotation. Reverse is only allowed for large heading errors
                # or for corridor recovery, and even then it is scaled down.
                if v_cmd < 0.0:
                    allow_reverse = (abs(pre_head_err) > self.cfg.phase01_reverse_heading_allow) or (pre_phase0_xdev > self.cfg.phase01_reverse_xdev_allow)
                    if allow_reverse:
                        v_cmd *= self.cfg.phase01_reverse_scale
                    else:
                        v_cmd = 0.0
                elif abs(raw_v) >= self.cfg.phase01_min_forward_raw_thresh and 0.0 < v_cmd < self.cfg.phase01_min_forward_speed:
                    v_cmd = self.cfg.phase01_min_forward_speed

                if abs(pre_head_err) > self.cfg.phase01_rotate_in_place_heading:
                    v_cmd *= self.cfg.phase01_rotate_scale
                elif abs(pre_head_err) > self.cfg.phase01_caution_heading:
                    v_cmd *= self.cfg.phase01_caution_scale

            elif self.phase == 1:
                # Phase 1 should be mostly forward motion with only corrective yaw.
                if v_cmd < 0.0:
                    allow_reverse = abs(pre_head_err) > self.cfg.phase1_reverse_heading_allow and pre_phase1_lane_err < self.cfg.phase1_center_soft_half_width
                    if allow_reverse:
                        v_cmd *= self.cfg.phase1_reverse_scale
                    else:
                        v_cmd = 0.0
                elif abs(raw_v) >= self.cfg.phase1_min_forward_raw_thresh and 0.0 < v_cmd < self.cfg.phase1_min_forward_speed:
                    v_cmd = self.cfg.phase1_min_forward_speed

                abs_head = abs(pre_head_err)
                if abs_head > self.cfg.phase1_rotate_in_place_heading:
                    v_cmd *= self.cfg.phase1_rotate_scale
                elif abs_head > self.cfg.phase1_caution_heading:
                    v_cmd *= self.cfg.phase1_caution_scale

                if abs_head < self.cfg.phase1_yaw_very_small_heading:
                    yaw_cmd *= self.cfg.phase1_yaw_very_small_scale
                elif abs_head < self.cfg.phase1_yaw_soft_cap_heading:
                    yaw_cmd *= self.cfg.phase1_yaw_soft_cap_scale
                else:
                    yaw_cmd *= 0.65

                # When already near the aisle centerline, bias even more strongly toward forward motion.
                if pre_phase1_lane_err < self.cfg.phase1_center_soft_half_width and abs_head < 0.95 and v_cmd > 0.0:
                    v_cmd = max(v_cmd, self.cfg.phase1_min_forward_speed)
            elif self.phase == 2:
                # phase 2: short-hold fine alignment near the goal XY.
                if v_cmd < 0.0:
                    v_cmd = 0.0
                elif abs(raw_v) >= self.cfg.phase2_min_forward_raw_thresh and 0.0 < v_cmd < self.cfg.phase2_min_forward_speed:
                    v_cmd = self.cfg.phase2_min_forward_speed

                abs_head = abs(pre_head_err)
                if abs_head > self.cfg.phase2_rotate_in_place_heading:
                    v_cmd *= self.cfg.phase2_rotate_scale
                elif abs_head > self.cfg.phase2_caution_heading:
                    v_cmd *= self.cfg.phase2_caution_scale

                if abs_head < 0.20:
                    yaw_cmd *= 0.18
                elif abs_head < 0.55:
                    yaw_cmd *= 0.40
                else:
                    yaw_cmd *= 0.70
            else:
                # phase 3: rotate in place to face the rack.
                v_cmd = 0.0
                yaw_cmd = raw_yaw * phase_profile["max_yaw_rate"]
                lift_cmd = 0.0
                publish_drive = True
                publish_lift = False
                command_mask = "drive_only_no_lift_publish"
        else:
            # PHASE4 action mask:
            #   - action[0]/action[1] are blocked from cmd_vel
            #   - action[2] is published as lift force
            #   - raw_v/raw_yaw are still penalized below, so the policy learns not to ask for motion here
            v_cmd = 0.0
            yaw_cmd = 0.0
            # PHASE4: action[2] -> force command for /model/<name>/lift_force_cmd.
            # Positive force raises the fork; negative force lowers it.
            #
            # Multi-level safety: the Stage-C checkpoint was trained only at level=1,
            # and with the old SDF upper limit it could rely on the physical stop.
            # After increasing lift_joint upper for level=2/3, prevent runaway overshoot
            # while keeping SAC's action output/resume compatibility unchanged.
            pre_lift_z = float(np.clip(pre_s["lift_z"], self.cfg.lift_min, self.cfg.lift_max))
            z_signed_pre = float(self.plan["goal_z"] - pre_lift_z)
            z_abs_pre = abs(z_signed_pre)
            lift_cmd_raw = raw_lift * phase_profile["max_lift_force"]
            lift_cmd = lift_cmd_raw
            lift_guard_notes: List[str] = []

            # At target: stop forcing the joint so stopped=True can be achieved.
            if z_abs_pre <= self.cfg.phase4_force_deadband:
                lift_cmd = 0.0
                lift_guard_notes.append("deadband_zero")
            else:
                # If below target, do not command down. If above target, do not command up.
                if z_signed_pre > 0.0 and lift_cmd < 0.0:
                    lift_cmd = self.cfg.phase4_wrong_direction_scale * lift_cmd
                    lift_guard_notes.append("block_down_below_goal")
                elif z_signed_pre < 0.0 and lift_cmd > 0.0:
                    lift_cmd = self.cfg.phase4_wrong_direction_scale * lift_cmd
                    lift_guard_notes.append("block_up_above_goal")

                # Taper the remaining force near the target to avoid overshooting.
                if z_abs_pre < self.cfg.phase4_force_taper_band:
                    scale = max(
                        self.cfg.phase4_near_force_scale_min,
                        z_abs_pre / max(self.cfg.phase4_force_taper_band, 1e-6),
                    )
                    lift_cmd *= scale
                    lift_guard_notes.append(f"taper={scale:.2f}")

            publish_drive = False
            publish_lift = True
            command_mask = "lift_only_no_drive_publish"
            if lift_guard_notes:
                command_mask += ":" + "+".join(lift_guard_notes)

        self.last_cmd = {"v_cmd": v_cmd, "yaw_cmd": yaw_cmd, "lift_cmd": lift_cmd, "mask": command_mask}

        # Phase-gated publishing for learning-time actions.
        # PHASE0-3: do not publish lift_force_cmd at all.
        # PHASE4: publish lift only; stop cmd_vel once when entering PHASE4.
        if publish_drive:
            self.bridge.send_drive(v_cmd, yaw_cmd)
        elif old_phase != 4:
            self.bridge.stop_drive()

        if publish_lift:
            self.bridge.send_lift(lift_cmd)

        total_dt = phase_profile["control_dt"] * phase_profile["action_repeat"]
        t_end = time.time() + total_dt
        while time.time() < t_end:
            rclpy.spin_once(self.bridge, timeout_sec=min(0.02, total_dt))
        s = self.bridge.get_state()
        s["lift_z"] = float(np.clip(s["lift_z"], self.cfg.lift_min, self.cfg.lift_max))

        seg_dist = self._segment_distance(s, self.phase)
        route_remaining = self._route_remaining(s)
        xy_eval_mode, xy_eval_dist, goal_ref_dists = self._xy_eval_dist(s)
        goal_dist = goal_ref_dists["base"]
        z_err = abs(self.plan["goal_z"] - s["lift_z"])
        head_err = self._head_err(s)
        face_yaw_err = self._face_yaw_err(s)
        route_speed = self._route_speed(s, total_dt)
        stopped = self._stopped(s)
        contact_collision = bool(s["collision"])
        geom_collision, geom_collision_kind = self._geometry_collision(s)
        collision = bool(contact_collision or geom_collision)
        collision_kind = "contact" if contact_collision else geom_collision_kind
        in_bounds = self._within_bounds(s)

        progress_seg = float(self.prev_segment_dist - seg_dist)
        progress_route = float(self.prev_route_remaining - route_remaining)
        progress_z = float(self.prev_z_err - z_err)
        progress_xy_eval = 0.0 if self.prev_xy_eval_dist is None else float(self.prev_xy_eval_dist - xy_eval_dist)
        heading_progress = float(self.prev_head_abs - abs(head_err))
        face_yaw_progress = 0.0 if self.prev_face_yaw_abs is None else float(self.prev_face_yaw_abs - abs(face_yaw_err))
        phase0_y_err = self._phase0_lane_y_err(s)
        phase0_x_dev = self._phase0_x_dev(s)
        phase1_lane_err = self._phase1_lane_err(s)
        progress_phase0_y = 0.0 if self.prev_phase0_y_err is None else float(self.prev_phase0_y_err - phase0_y_err)
        growth_phase0_x = 0.0 if self.prev_phase0_x_dev is None else float(phase0_x_dev - self.prev_phase0_x_dev)
        progress_phase1_lane = 0.0 if self.prev_phase1_lane_err is None else float(self.prev_phase1_lane_err - phase1_lane_err)

        reward = 0.0
        terminated = False
        truncated = False
        success = False

        reward += self.cfg.step_penalty
        reward += self.cfg.action_penalty * float(np.sum(np.square(action)))

        if self.phase in (0, 1, 2, 3):
            reward += self.cfg.lift_before_xy_penalty * abs(raw_lift)
        if self.phase == 3:
            reward += self.cfg.phase3_move_penalty_gain * abs(raw_v)
            reward += self.cfg.phase3_lift_penalty_gain * abs(raw_lift)
        if self.phase == 4:
            reward += self.cfg.move_after_xy_penalty * (abs(raw_v) + abs(raw_yaw))
        if self.phase == 0 and v_cmd < 0.0:
            reward += self.cfg.phase01_reverse_penalty_gain * abs(v_cmd)
        if self.phase == 1 and v_cmd < 0.0:
            reward += self.cfg.phase1_reverse_penalty_gain * abs(v_cmd)
        if self.phase == 2 and raw_v < 0.0:
            reward += self.cfg.phase2_reverse_penalty_gain * abs(raw_v)

        idle_cmd = (
            abs(v_cmd) < self.cfg.max_v * self.cfg.idle_cmd_ratio_thresh and
            abs(yaw_cmd) < self.cfg.max_yaw_rate * self.cfg.idle_cmd_ratio_thresh and
            abs(lift_cmd) < self.cfg.max_lift_force * self.cfg.idle_cmd_ratio_thresh
        )
        idle_state = (
            abs(s["linear_vel"]) < self.cfg.idle_speed_thresh and
            abs(s["angular_vel"]) < self.cfg.idle_speed_thresh and
            abs(s["lift_vel"]) < self.cfg.stop_lift_thresh
        )
        is_idle = idle_cmd and idle_state
        if is_idle:
            reward += self.cfg.idle_penalty

        zone_idx, entered_new_zone, backtracked_zone = self._phase_zone_progress(s)

        if self.phase in (0, 1, 2, 3):
            if self.phase == 0:
                # Phase 0 is intentionally strict: almost all positive reward comes from
                # entering a strictly closer distance band and reducing |y-lane_y| while
                # keeping x near portal_x. Heading is only weakly rewarded when real
                # progress is happening.
                reward += self.cfg.phase0_route_speed_gain * max(route_speed, 0.0)
                reward -= self.cfg.phase0_route_reverse_gain * max(-route_speed, 0.0)
                if (entered_new_zone or progress_phase0_y > self.cfg.min_portal_progress) and head_err * head_err < 1.6:
                    reward += self.cfg.phase0_heading_progress_gain * max(heading_progress, 0.0)
                    reward += self.cfg.phase0_heading_gain * math.cos(head_err)
                elif abs(raw_yaw) > 0.6 and heading_progress < -0.01:
                    reward -= self.cfg.spin_penalty_gain * abs(yaw_cmd)
            elif self.phase == 1:
                reward += self.cfg.phase1_route_speed_gain * max(route_speed, 0.0)
                reward -= self.cfg.phase1_route_reverse_gain * max(-route_speed, 0.0)
                if route_speed > 0.0 and abs(head_err) < 1.30:
                    reward += self.cfg.phase1_heading_gain * math.cos(head_err)
                    reward += self.cfg.phase1_heading_progress_gain * max(heading_progress, 0.0)
                reward -= self.cfg.phase1_yaw_penalty_gain * abs(yaw_cmd)
                reward -= self.cfg.phase1_cross_track_gain * phase1_lane_err
                if progress_phase1_lane > 0.0:
                    reward += self.cfg.phase1_cross_track_progress_gain * progress_phase1_lane
                elif progress_phase1_lane < 0.0:
                    reward += self.cfg.phase1_cross_track_growth_penalty_gain * progress_phase1_lane
                if phase1_lane_err > self.cfg.phase1_center_soft_half_width:
                    reward -= self.cfg.phase1_center_soft_penalty_gain * (phase1_lane_err - self.cfg.phase1_center_soft_half_width)
                if phase1_lane_err > self.cfg.phase1_center_hard_half_width:
                    reward -= self.cfg.phase1_center_hard_penalty_gain * (phase1_lane_err - self.cfg.phase1_center_hard_half_width)
                if abs(prev_cmd.get("yaw_cmd", 0.0)) > self.cfg.phase1_yaw_flip_min_mag and abs(yaw_cmd) > self.cfg.phase1_yaw_flip_min_mag:
                    if np.sign(prev_cmd.get("yaw_cmd", 0.0)) != np.sign(yaw_cmd):
                        reward -= self.cfg.phase1_yaw_flip_penalty
                if abs(raw_yaw) > 0.6 and heading_progress < -0.02:
                    reward -= self.cfg.spin_penalty_gain * abs(yaw_cmd)
            elif self.phase == 2:
                reward += self.cfg.phase2_xy_progress_gain * progress_xy_eval
                reward -= self.cfg.phase2_yaw_penalty_gain * abs(yaw_cmd)
                if abs(raw_yaw) > 0.6 and progress_xy_eval < -0.01:
                    reward -= self.cfg.spin_penalty_gain * abs(yaw_cmd)
            else:
                reward += self.cfg.phase3_yaw_progress_gain * face_yaw_progress
                reward -= self.cfg.phase3_yaw_penalty_gain * abs(yaw_cmd)

        if entered_new_zone:
            if self.phase == 0:
                reward += self.cfg.phase0_zone_bonus0
            elif self.phase == 1:
                reward += self.cfg.zone_bonus1
            elif self.phase == 2:
                reward += self.cfg.zone_bonus2
        else:
            reward += self._distance_level_penalty(self.phase, zone_idx)

        if backtracked_zone:
            reward += self.cfg.backtrack_penalty

        reward += self._wrong_aisle_penalty_now(s)

        if self.phase == 0:
            corridor_pen, corridor_term = self._phase0_corridor_penalty_and_terminate(s)
            reward += corridor_pen
            reward -= self.cfg.phase0_x_dev_penalty_gain * phase0_x_dev
            if growth_phase0_x > 0.0:
                reward -= self.cfg.phase0_x_dev_growth_penalty_gain * growth_phase0_x

            # Positive reward is dominated by best-zone updates and y-progress toward lane_y.
            if entered_new_zone:
                reward += self.cfg.phase0_y_progress_gain * max(progress_phase0_y, 0.0)
                reward += 0.10 * self.cfg.seg_progress_gain0 * max(progress_route, 0.0)
            else:
                # No positive dense reward when not improving the best zone.
                if progress_phase0_y < 0.0:
                    reward += self.cfg.phase0_y_progress_gain * progress_phase0_y
                if progress_route < 0.0:
                    reward += 0.20 * self.cfg.seg_progress_gain0 * progress_route

            if corridor_term and not terminated:
                terminated = True
            if seg_dist <= self.cfg.map_cfg.wp0_tol:
                self.portal_reached = True
                self.phase = 1
                reward += self.cfg.portal_bonus

        elif self.phase == 1:
            if entered_new_zone:
                reward += 1.10 * self.cfg.seg_progress_gain1 * max(progress_seg, 0.0)
                reward += 0.55 * self.cfg.seg_progress_gain1 * max(progress_route, 0.0)
            else:
                reward += 0.10 * self.cfg.seg_progress_gain1 * min(progress_seg, 0.0)
                reward += 0.05 * self.cfg.seg_progress_gain1 * min(progress_route, 0.0)
            if seg_dist <= self.cfg.map_cfg.wp1_tol:
                self.aisle_reached = True
                self.phase = 2
                reward += self.cfg.aisle_bonus

        elif self.phase == 2:
            if xy_eval_dist <= self.cfg.xy_tol:
                self.xy_reached = True
                self.phase = 3
                reward += self.cfg.xy_bonus

        elif self.phase == 3:
            if abs(face_yaw_err) <= self.cfg.yaw_tol:
                # Important: PHASE3 reaches the target yaw while the robot may still
                # have angular velocity. Do NOT set phase=4 immediately. First send
                # cmd_vel=(0,0) through ROS2->Gazebo and wait briefly, then start lift.
                stop_before, stop_after = self.bridge.hard_stop_drive(
                    duration_sec=self.cfg.phase4_entry_stop_sec,
                    dt_sec=self.cfg.phase4_entry_stop_dt,
                )
                after_face_yaw_err = self._face_yaw_err(stop_after)
                print(
                    f"PHASE4_ENTRY_STOP ep={self.episode_idx} step={self.steps} "
                    f"stop_sec={self.cfg.phase4_entry_stop_sec:.3f} stop_dt={self.cfg.phase4_entry_stop_dt:.3f} "
                    f"before_lin_v={stop_before['linear_vel']:.3f} before_ang_v={stop_before['angular_vel']:.3f} "
                    f"after_lin_v={stop_after['linear_vel']:.3f} after_ang_v={stop_after['angular_vel']:.3f} "
                    f"before_yaw={stop_before['yaw']:.3f} after_yaw={stop_after['yaw']:.3f} "
                    f"before_face_yaw_err={face_yaw_err:.3f} after_face_yaw_err={after_face_yaw_err:.3f}",
                    flush=True,
                )
                # Refresh state for the transition log and the next PHASE4 step.
                s = stop_after
                s["lift_z"] = float(np.clip(s["lift_z"], self.cfg.lift_min, self.cfg.lift_max))
                seg_dist = self._segment_distance(s, self.phase)
                route_remaining = self._route_remaining(s)
                xy_eval_mode, xy_eval_dist, goal_ref_dists = self._xy_eval_dist(s)
                goal_dist = goal_ref_dists["base"]
                z_err = abs(self.plan["goal_z"] - s["lift_z"])
                head_err = self._head_err(s)
                face_yaw_err = after_face_yaw_err
                stopped = self._stopped(s)
                contact_collision = bool(s["collision"])
                geom_collision, geom_collision_kind = self._geometry_collision(s)
                collision = bool(contact_collision or geom_collision)
                collision_kind = "contact" if contact_collision else geom_collision_kind
                in_bounds = self._within_bounds(s)
                phase0_y_err = self._phase0_lane_y_err(s)
                phase0_x_dev = self._phase0_x_dev(s)
                phase1_lane_err = self._phase1_lane_err(s)

                self.rot_reached = True
                self.phase = 4
                reward += self.cfg.rotate_bonus

        else:
            signed_z_err = float(self.plan["goal_z"] - s["lift_z"])
            overshoot = max(0.0, -signed_z_err)

            # Dense z shaping: reward progress toward the current goal_z.
            reward += self.cfg.z_progress_gain * progress_z
            # Small absolute-error penalty keeps the policy focused even when progress stalls.
            reward -= self.cfg.phase4_abs_z_penalty_gain * z_err
            # Strong asymmetric penalty: being above the target is worse than being below,
            # because old level=1 policy tends to keep lifting after SDF upper was raised.
            reward -= self.cfg.phase4_overshoot_penalty_gain * overshoot
            reward -= self.cfg.phase4_big_overshoot_penalty_gain * max(0.0, overshoot - self.cfg.phase4_big_overshoot_margin)

            # Penalize asking for force in the wrong direction relative to goal_z.
            if signed_z_err > self.cfg.z_tol and raw_lift < 0.0:
                reward -= self.cfg.phase4_wrong_direction_penalty_gain * abs(raw_lift)
            elif signed_z_err < -self.cfg.z_tol and raw_lift > 0.0:
                reward -= self.cfg.phase4_wrong_direction_penalty_gain * abs(raw_lift)

            if (not self.z_reached) and (z_err < self.cfg.z_tol):
                self.z_reached = True
                reward += self.cfg.z_bonus

        if self.portal_reached and self.phase >= 1 and point_dist(s["x"], s["y"], *self.plan["wp0"]) > self.cfg.map_cfg.wp0_tol + 2.5:
            reward += self.cfg.backtrack_penalty
        if self.aisle_reached and self.phase >= 2 and xy_eval_dist > self.cfg.map_cfg.wp1_tol + 2.5:
            reward += self.cfg.backtrack_penalty

        if self.phase != old_phase:
            self._log_phase_change(old_phase, self.phase, s, seg_dist, z_err)

        if self.xy_reached and self.rot_reached and self.z_reached and stopped:
            reward += self.cfg.success_bonus
            terminated = True
            success = True

        if collision and self.steps > self.cfg.collision_grace_steps:
            reward += self.cfg.collision_penalty
            terminated = True

        if (not in_bounds) and (not terminated):
            reward += self.cfg.out_of_bounds_penalty
            terminated = True

        if self.phase == 0:
            progressed = entered_new_zone or (progress_phase0_y > self.cfg.min_portal_progress and phase0_x_dev <= self.cfg.phase0_corridor_soft_half_width)
        elif self.phase == 1:
            progressed = (
                (progress_seg > self.cfg.min_aisle_progress) or
                (progress_route > self.cfg.min_aisle_progress) or
                (progress_phase1_lane > 0.02)
            )
        elif self.phase == 2:
            progressed = progress_xy_eval > self.cfg.min_xy_progress
        elif self.phase == 3:
            progressed = face_yaw_progress > 0.002
        else:
            progressed = (progress_z > self.cfg.min_z_progress) or (z_err < self.cfg.z_tol)

        if progressed:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        if self.no_progress_steps >= self.cfg.no_progress_limit and not terminated:
            reward += self.cfg.stuck_penalty
            truncated = True
        if self.steps >= self.cfg.max_steps and not terminated:
            reward += self.cfg.timeout_penalty
            truncated = True

        self.prev_segment_dist = seg_dist
        self.prev_route_remaining = route_remaining
        self.prev_z_err = z_err
        self.prev_head_abs = abs(head_err)
        self.prev_x = s["x"]
        self.prev_y = s["y"]
        self.prev_phase0_y_err = phase0_y_err
        self.prev_phase0_x_dev = phase0_x_dev
        self.prev_phase1_lane_err = phase1_lane_err
        self.prev_xy_eval_dist = xy_eval_dist
        self.prev_face_yaw_abs = abs(face_yaw_err)

        info = {
            "is_success": success,
            "phase": self.phase,
            "portal_reached": self.portal_reached,
            "aisle_reached": self.aisle_reached,
            "xy_reached": self.xy_reached,
            "rot_reached": self.rot_reached,
            "z_reached": self.z_reached,
            "collision": collision,
            "stopped": stopped,
            "steps": self.steps,
            "x": s["x"], "y": s["y"], "base_z": s["base_z"], "yaw": s["yaw"], "lift_z": s["lift_z"],
            "goal_x": self.plan["goal_x"], "goal_y": self.plan["goal_y"], "goal_z": self.plan["goal_z"],
            "portal_x": self.plan["wp0"][0], "portal_y": self.plan["wp0"][1],
            "seg_dist": seg_dist, "goal_dist": goal_dist, "z_err": z_err,
            "front_goal_dist": goal_ref_dists["front"], "fork_goal_dist": goal_ref_dists["fork_mid"],
            "xy_eval_mode": xy_eval_mode, "xy_eval_dist": xy_eval_dist, "hold_time": total_dt,
            "route_remaining": route_remaining, "route_speed": route_speed, "head_err": head_err, "target_yaw": self.plan["target_yaw"], "face_yaw_err": face_yaw_err,
            "zone_idx": zone_idx, "zone_best0": self.zone_best0, "zone_best1": self.zone_best1,
            "v_cmd": v_cmd, "yaw_cmd": yaw_cmd, "lift_cmd": lift_cmd,
            "no_progress_steps": self.no_progress_steps,
            "idle": is_idle,
            "in_bounds": in_bounds,
            "reward": reward,
            "phase0_y_err": phase0_y_err,
            "phase0_x_dev": phase0_x_dev,
            "phase1_lane_err": phase1_lane_err,
        }

        if (self.steps <= 10) or (self.cfg.debug_every_steps > 0 and self.steps % self.cfg.debug_every_steps == 0):
            print(
                f"STEP_DBG ep={self.episode_idx} step={self.steps} phase={PHASE_NAMES[self.phase]} "
                f"x={s['x']:.3f} y={s['y']:.3f} base_z={s['base_z']:.3f} lift_z={s['lift_z']:.3f} yaw={s['yaw']:.3f} "
                f"goal=({self.plan['goal_x']:.3f},{self.plan['goal_y']:.3f},{self.plan['goal_z']:.3f}) "
                f"wp0=({self.plan['wp0'][0]:.3f},{self.plan['wp0'][1]:.3f}) seg_dist={seg_dist:.3f} route_rem={route_remaining:.3f} goal_dist={goal_dist:.3f} xy_eval={xy_eval_mode}:{xy_eval_dist:.3f} front_goal={goal_ref_dists['front']:.3f} fork_goal={goal_ref_dists['fork_mid']:.3f} z_err={z_err:.3f} "
                f"route_speed={route_speed:.3f} head_err={head_err:.3f} zone_idx={zone_idx} p0_yerr={phase0_y_err:.3f} p0_xdev={phase0_x_dev:.3f} p1_yerr={phase1_lane_err:.3f} face_yaw_err={face_yaw_err:.3f} hold={total_dt:.3f} "
                f"v_cmd={v_cmd:.3f} yaw_cmd={yaw_cmd:.3f} lift_cmd={lift_cmd:.3f} cmd_mask={command_mask} lin_v={s['linear_vel']:.3f} ang_v={s['angular_vel']:.3f} lift_v={s['lift_vel']:.3f} "
                f"collision={collision} contact={contact_collision} geom={geom_collision} kind={collision_kind or 'none'} in_bounds={in_bounds} reward={reward:.3f} idle={is_idle} np={self.no_progress_steps}",
                flush=True,
            )

        if terminated or truncated:
            end_reason = "success" if success else "terminated" if terminated else "truncated"
            print(
                f"EP_END ep={self.episode_idx} reason={end_reason} phase={PHASE_NAMES[self.phase]} "
                f"x={s['x']:.3f} y={s['y']:.3f} base_z={s['base_z']:.3f} lift_z={s['lift_z']:.3f} yaw={s['yaw']:.3f} "
                f"goal_dist={goal_dist:.3f} xy_eval={xy_eval_mode}:{xy_eval_dist:.3f} front_goal={goal_ref_dists['front']:.3f} fork_goal={goal_ref_dists['fork_mid']:.3f} seg_dist={seg_dist:.3f} z_err={z_err:.3f} route_speed={route_speed:.3f} head_err={head_err:.3f} p0_yerr={phase0_y_err:.3f} p0_xdev={phase0_x_dev:.3f} p1_yerr={phase1_lane_err:.3f} hold={total_dt:.3f} "
                f"portal_reached={self.portal_reached} aisle_reached={self.aisle_reached} xy_reached={self.xy_reached} rot_reached={self.rot_reached} z_reached={self.z_reached} "
                f"zone_best0={self.zone_best0} zone_best1={self.zone_best1} v_cmd={v_cmd:.3f} yaw_cmd={yaw_cmd:.3f} lift_cmd={lift_cmd:.3f} cmd_mask={command_mask} "
                f"collision={collision} contact={contact_collision} geom={geom_collision} kind={collision_kind or 'none'} in_bounds={in_bounds} stopped={stopped} steps={self.steps} reward={reward:.3f} np={self.no_progress_steps}",
                flush=True,
            )
            self.bridge.stop()

        return self._get_obs(s), reward, terminated, truncated, info

    def close(self):
        try:
            self.marker_mgr._remove_model(self.marker_mgr.model_name)
        except Exception:
            pass
        self.bridge.stop()
        self.bridge.close()


def parse_contact_topics(arg: str, cfg: EnvConfig) -> List[str]:
    if not arg:
        return list(cfg.contact_topics)
    return [s.strip() for s in arg.split(",") if s.strip()]


def build_env(args) -> gym.Env:
    cfg = EnvConfig(
        world_name=args.world_name,
        model_name=args.model_name,
        goal_x=args.goal_x,
        goal_y=args.goal_y,
        goal_z=args.goal_z,
        model_sdf=args.model_sdf,
        spawn_x=args.spawn_x,
        spawn_y=args.spawn_y,
        spawn_yaw=args.spawn_yaw,
        spawn_z=args.spawn_z,
        target_row=args.target_row,
        target_bin=args.target_bin,
        target_level=args.target_level,
        target_face=args.target_face,
        randomize_target=args.randomize_target,
        target_rows=parse_int_set(args.target_rows, 0, 2),
        target_bins=parse_int_set(args.target_bins, 0, 19),
        target_levels=parse_int_set(args.target_levels, 0, 3),
        use_target_map=(not args.disable_target_map),
        debug_every_steps=args.debug_every_steps,
        gz_pose_topic=args.gz_pose_topic,
        gz_pose_timeout=args.gz_pose_timeout,
        spawn_target_marker=(not getattr(args, "no_target_marker", False)),
        xy_reference_point=args.xy_reference_point,
        phase4_max_lift_force=args.phase4_max_lift_force,
        max_lift_force=args.max_lift_force,
        lift_max=args.lift_max,
        phase4_force_deadband=args.phase4_force_deadband,
        phase4_force_taper_band=args.phase4_force_taper_band,
        phase4_overshoot_penalty_gain=args.phase4_overshoot_penalty_gain,
        phase4_big_overshoot_penalty_gain=args.phase4_big_overshoot_penalty_gain,
        phase_pre_lift_hold_force=args.phase_pre_lift_hold_force,
        phase4_entry_stop_sec=args.phase4_entry_stop_sec,
        phase4_entry_stop_dt=args.phase4_entry_stop_dt,
        respawn_self_test=(not args.no_respawn_self_test),
        respawn_self_test_required=(not args.respawn_self_test_warning_only),
        respawn_self_test_require_lift_down=args.selftest_require_lift_down,
        respawn_self_test_drive_v=args.selftest_drive_v,
        respawn_self_test_drive_sec=args.selftest_drive_sec,
        respawn_self_test_lift_force=args.selftest_lift_force,
        respawn_self_test_lift_sec=args.selftest_lift_sec,
    )
    hit, kind = geometry_collision_for_pose(cfg, cfg.spawn_x, cfg.spawn_y, cfg.spawn_yaw)
    if hit:
        raise RuntimeError(
            f"Configured spawn pose collides before training starts: x={cfg.spawn_x:.3f} y={cfg.spawn_y:.3f} yaw={cfg.spawn_yaw:.3f} kind={kind}"
        )
    contact_topics = parse_contact_topics(args.contact_topics, cfg)
    bridge = GazeboRosBridge(
        cfg=cfg,
        odom_topic=args.odom_topic,
        cmd_vel_topic=args.cmd_vel_topic,
        lift_cmd_topic=args.lift_cmd_topic,
        joint_topic=args.joint_topic,
        contact_topics=contact_topics,
    )
    # In the split world+forklift setup, the world starts without a forklift model.
    # The first env.reset() will respawn the forklift and only then will odom/joint topics exist.
    env = ForkliftRouteEnv(bridge, cfg)
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-name", default="warehouse_world")
    parser.add_argument("--model-name", default="forklift_simple")
    parser.add_argument("--goal-x", type=float, default=0.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-z", type=float, default=2.5)
    parser.add_argument("--model-sdf", default="models/forklift_simple/model.sdf", help="Path to the forklift model.sdf used for respawn reset")
    parser.add_argument("--spawn-x", type=float, default=38.0)
    parser.add_argument("--spawn-y", type=float, default=14.0)
    parser.add_argument("--spawn-yaw", type=float, default=-1.571)
    parser.add_argument("--spawn-z", type=float, default=0.10)
    parser.add_argument("--target-row", type=int, default=2)
    parser.add_argument("--target-bin", type=int, default=10)
    parser.add_argument("--target-level", type=int, default=1)
    parser.add_argument("--target-face", choices=["auto", "low", "high"], default="auto")
    parser.add_argument("--randomize-target", action="store_true", help="Sample target row/bin/level at every episode reset for Goal-Conditioned SAC")
    parser.add_argument("--target-rows", default="1", help="Candidate rows for --randomize-target, e.g. '1' or '0,1,2' or '0-2'")
    parser.add_argument("--target-bins", default="10", help="Candidate bins for --randomize-target, e.g. '8-12' or '0-19'")
    parser.add_argument("--target-levels", default="1", help="Candidate levels for --randomize-target, e.g. '1' or '0,1'. Current forklift reaches only low levels unless lift limit is increased.")
    parser.add_argument("--disable-target-map", action="store_true")
    parser.add_argument("--odom-topic", default=None)
    parser.add_argument("--cmd-vel-topic", default=None)
    parser.add_argument("--lift-cmd-topic", default=None, help="Default is /model/<model>/lift_force_cmd in train9")
    parser.add_argument("--joint-topic", default=None)
    parser.add_argument("--contact-topics", default="")
    parser.add_argument("--gz-pose-topic", default="")
    parser.add_argument("--gz-pose-timeout", type=float, default=8.0)
    parser.add_argument("--timesteps", type=int, default=400000)
    parser.add_argument("--model-out", default="sac_forklift_train12_goal_conditioned")
    parser.add_argument("--save-dir", default="./models_ckpt12_goal_conditioned")
    parser.add_argument("--checkpoint-freq", type=int, default=5000)
    parser.add_argument("--xy-reference-point", choices=["base", "front", "fork_mid"], default="base")
    parser.add_argument("--debug-every-steps", type=int, default=50)
    parser.add_argument("--no-target-marker", action="store_true")
    parser.add_argument("--max-lift-force", type=float, default=150.0, help="Fallback lift force scale for idle checks")
    parser.add_argument("--phase4-max-lift-force", type=float, default=800.0, help="PHASE4 action[2] multiplier; force, not velocity. Visible-selftest default is 800N for gravity-enabled lift.")
    parser.add_argument("--lift-max", type=float, default=8.4, help="Clip/normalize lift joint position in observations/rewards. Set to SDF lift_joint upper, e.g. 8.4 for all levels.")
    parser.add_argument("--phase4-force-deadband", type=float, default=0.10, help="PHASE4: send zero lift force when |goal_z-lift_z| is within this distance.")
    parser.add_argument("--phase4-force-taper-band", type=float, default=0.55, help="PHASE4: taper lift force when near target z.")
    parser.add_argument("--phase4-overshoot-penalty-gain", type=float, default=45.0, help="PHASE4: penalty per meter above target z.")
    parser.add_argument("--phase4-big-overshoot-penalty-gain", type=float, default=120.0, help="PHASE4: extra penalty per meter above big overshoot margin.")
    parser.add_argument("--phase-pre-lift-hold-force", type=float, default=0.0, help="Deprecated/ignored in train11. PHASE0-3 do not publish lift_force_cmd at all.")
    parser.add_argument("--phase4-entry-stop-sec", type=float, default=0.80, help="Before setting phase=4, send cmd_vel=0,0 for this many seconds to remove PHASE3 rotation.")
    parser.add_argument("--phase4-entry-stop-dt", type=float, default=0.05, help="Interval for repeated cmd_vel=0,0 during PHASE4 entry stop.")
    parser.add_argument("--no-respawn-self-test", action="store_true", help="Disable drive/lift self-test after respawn")
    parser.add_argument("--respawn-self-test-warning-only", action="store_true", help="Log self-test failure but do not abort training")
    parser.add_argument("--selftest-require-lift-down", action="store_true", help="Make lift-down return test mandatory. Default train9_v2 only requires drive + lift-up and logs lift-down as diagnostic.")
    parser.add_argument("--selftest-drive-v", type=float, default=0.80)
    parser.add_argument("--selftest-drive-sec", type=float, default=1.20)
    parser.add_argument("--selftest-lift-force", type=float, default=800.0)
    parser.add_argument("--selftest-lift-sec", type=float, default=1.40)
    parser.add_argument("--resume-from", default="", help="Path to a Stable-Baselines3 .zip checkpoint to resume from")
    parser.add_argument("--reset-num-timesteps", action="store_true", help="When resuming, reset the displayed timestep counter to 0")
    args = parser.parse_args()

    if "DISPLAY" not in os.environ:
        print("WARNING: DISPLAY is not set. Use export DISPLAY=:99 if you want GUI.", flush=True)

    print(
        f"SPAWN_CONFIG model={args.model_name} model_sdf={args.model_sdf or '[auto]'} x={args.spawn_x:.3f} y={args.spawn_y:.3f} yaw={args.spawn_yaw:.3f} z={args.spawn_z:.3f}",
        flush=True,
    )
    print("RESPAWN_MODE split_world=true (world starts without forklift; env.reset() will spawn it)", flush=True)
    print(
        f"TRAIN12_GOAL_CONDITIONED_CONFIG default_lift_topic=/model/{args.model_name}/lift_force_cmd "
        f"phase4_max_lift_force={args.phase4_max_lift_force:.3f} lift_max={args.lift_max:.3f} "
        f"phase0_3_publish=cmd_vel_only phase4_publish=lift_only stop_before_phase4=True "
        f"phase4_entry_stop_sec={args.phase4_entry_stop_sec:.3f} "
        f"selftest_enabled={not args.no_respawn_self_test} "
        f"selftest_required={not args.respawn_self_test_warning_only} selftest_require_lift_down={args.selftest_require_lift_down} "
        f"randomize_target={args.randomize_target} target_rows={args.target_rows} target_bins={args.target_bins} target_levels={args.target_levels}",
        flush=True,
    )

    rclpy.init()
    env = build_env(args)

    if args.resume_from:
        print(f"RESUME_FROM path={args.resume_from}", flush=True)
        model = SAC.load(args.resume_from, env=env, device="auto")
    else:
        print("START_MODE scratch", flush=True)
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=300000,
            learning_starts=5000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            ent_coef="auto",
            verbose=1,
            tensorboard_log="./tb_forklift_simple/",
            device="auto",
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = CheckpointCallback(save_freq=args.checkpoint_freq, save_path=str(save_dir), name_prefix="sac_forklift_goal_ckpt")
    best_success_cb = SaveBestSuccessCallback(save_dir=str(save_dir), check_freq=1000, verbose=1)
    callbacks = CallbackList([checkpoint_cb, best_success_cb])

    print(
        f"TRAIN_CONFIG timesteps={args.timesteps} checkpoint_freq={args.checkpoint_freq} save_dir={save_dir} model_out={args.model_out} "
        f"reset_num_timesteps={bool(args.reset_num_timesteps)}",
        flush=True,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            log_interval=10,
            progress_bar=True,
            callback=callbacks,
            reset_num_timesteps=(args.reset_num_timesteps if args.resume_from else True),
        )
        model.save(args.model_out)
        print(f"saved latest -> {args.model_out}.zip", flush=True)
    finally:
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
