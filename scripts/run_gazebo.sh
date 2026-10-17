#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export DISPLAY="${DISPLAY:-:99}"
export GZ_SIM_RESOURCE_PATH="$PROJECT_DIR/models:${GZ_SIM_RESOURCE_PATH:-}"

WORLD="$PROJECT_DIR/worlds/warehouse_with_pallet.sdf"
LOG="/tmp/physical_ai_forklift_gazebo.log"
PID_FILE="/tmp/physical_ai_forklift_gazebo.pid"

echo "Starting Gazebo..."
echo "DISPLAY : $DISPLAY"
echo "WORLD   : $WORLD"

gz sim "$WORLD" > "$LOG" 2>&1 &
GZ_PID=$!

echo "$GZ_PID" > "$PID_FILE"

echo "Gazebo PID: $GZ_PID"

sleep 10

# Maximize Gazebo window if wmctrl can find it.
wmctrl -r "Gazebo Sim" -b add,maximized_vert,maximized_horz 2>/dev/null || true

echo "Unpausing simulation..."

gz service \
  -s /world/warehouse_world/control \
  --reqtype gz.msgs.WorldControl \
  --reptype gz.msgs.Boolean \
  --timeout 3000 \
  --req 'pause: false'

echo
echo "Gazebo started."
echo "Log: $LOG"
