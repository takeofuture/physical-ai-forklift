#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export DISPLAY=${DISPLAY:-:99}

source /opt/ros/jazzy/setup.bash
source /opt/pyenv/gazebo_wh/bin/activate

export GZ_SIM_RESOURCE_PATH="$PROJECT_DIR/models:${GZ_SIM_RESOURCE_PATH:-}"

ROW="${1:-1}"
BIN="${2:-10}"
LEVEL="${3:-1}"

CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/checkpoints/best_success_model.zip}"

echo "============================================================"
echo "Forklift Goal-Conditioned Inference"
echo "============================================================"
echo "Project    : $PROJECT_DIR"
echo "Checkpoint : $CHECKPOINT"
echo "Target     : row=$ROW bin=$BIN level=$LEVEL"
echo "DISPLAY    : $DISPLAY"
echo "============================================================"

# ------------------------------------------------------------
# ROS 2 <-> Gazebo bridge
# ------------------------------------------------------------

echo
echo "Starting ROS-Gazebo bridge..."

ros2 run ros_gz_bridge parameter_bridge \
  /model/forklift_simple/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
  /model/forklift_simple/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry \
  /model/forklift_simple/lift_force_cmd@std_msgs/msg/Float64]gz.msgs.Double \
  /world/warehouse_world/model/forklift_simple/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model \
  /world/warehouse_world/model/forklift_simple/link/base_link/sensor/base_link_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts \
  /world/warehouse_world/model/forklift_simple/link/fork_carriage_link/sensor/fork_carriage_link_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts \
  /world/warehouse_world/model/forklift_simple/link/left_fork_link/sensor/left_fork_link_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts \
  /world/warehouse_world/model/forklift_simple/link/right_fork_link/sensor/right_fork_link_contact/contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts \
  > /tmp/forklift_ros_gz_bridge.log 2>&1 &

BRIDGE_PID=$!

cleanup() {
    echo
    echo "Stopping ROS-Gazebo bridge..."

    if kill -0 "$BRIDGE_PID" 2>/dev/null; then
        kill "$BRIDGE_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

sleep 3

if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "ERROR: ROS-Gazebo bridge failed to start."
    echo
    cat /tmp/forklift_ros_gz_bridge.log
    exit 1
fi

echo "Bridge PID: $BRIDGE_PID"

# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

echo
echo "Starting inference..."

python -u src/infer_goal_conditioned.py \
  --checkpoint "$CHECKPOINT" \
  --world-name warehouse_world \
  --model-name forklift_simple \
  --model-sdf "$PROJECT_DIR/models/forklift_simple/model.sdf" \
  --spawn-x 38.0 \
  --spawn-y 14.0 \
  --spawn-z 0.10 \
  --spawn-yaw -1.571 \
  --target-row "$ROW" \
  --target-bin "$BIN" \
  --target-level "$LEVEL" \
  --target-face auto \
  --device auto
