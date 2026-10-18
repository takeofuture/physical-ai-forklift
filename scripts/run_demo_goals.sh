#!/bin/bash

set +u

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

source /opt/ros/jazzy/setup.bash
source /opt/pyenv/gazebo_wh/bin/activate

export DISPLAY=:99
export GZ_SIM_RESOURCE_PATH="$PROJECT_DIR/models:${GZ_SIM_RESOURCE_PATH:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
VIDEO_NAME="${1:-goal20_${TIMESTAMP}}"

CAMERA_WAIT_SEC="${CAMERA_WAIT_SEC:-20}"
BETWEEN_RUNS_SEC="${BETWEEN_RUNS_SEC:-3}"

# ------------------------------------------------------------
# 20 different Goal patterns
#
# Format:
#   "ROW BIN LEVEL"
#
# Valid:
#   ROW   = 0-2
#   BIN   = 0-19
#   LEVEL = 0-3
# ------------------------------------------------------------

GOALS=(
    "0 17 1"

    "1 15 0"

    "2 10 2"
)

TOTAL=${#GOALS[@]}

RESULT_DIR="$PROJECT_DIR/videos"
mkdir -p "$RESULT_DIR"

RESULT_CSV="$RESULT_DIR/${VIDEO_NAME}_results.csv"

echo "run,row,bin,level,result,return_code" > "$RESULT_CSV"

RECORDING=0


cleanup()
{
    echo
    echo "Cleaning up..."

    if [ "$RECORDING" -eq 1 ]; then
        ./scripts/record.sh stop || true
        RECORDING=0
    fi
}

trap cleanup EXIT INT TERM


echo
echo "============================================================"
echo " Goal-Conditioned Forklift 20-Goal Demo"
echo "============================================================"
echo " Total goals : $TOTAL"
echo " Video       : videos/${VIDEO_NAME}.mp4"
echo " Results     : videos/${VIDEO_NAME}_results.csv"
echo " DISPLAY     : $DISPLAY"
echo "============================================================"
echo


# ------------------------------------------------------------
# 1. Start Xvfb / Openbox / VNC
# ------------------------------------------------------------

if pgrep -f "Xvfb :99" >/dev/null 2>&1; then
    echo "[1/5] DISPLAY :99 already running."
else
    echo "[1/5] Starting display / VNC..."
    ./scripts/setup_display.sh
    sleep 2
fi


# ------------------------------------------------------------
# 2. Start Gazebo
# ------------------------------------------------------------

if pgrep -f "gz sim" >/dev/null 2>&1; then
    echo "[2/5] Gazebo already running."
else
    echo "[2/5] Starting Gazebo..."
    ./scripts/run_gazebo.sh
fi


# Make sure simulation is unpaused

gz service \
    -s /world/warehouse_world/control \
    --reqtype gz.msgs.WorldControl \
    --reptype gz.msgs.Boolean \
    --timeout 3000 \
    --req 'pause: false' \
    >/dev/null 2>&1 || true


# ------------------------------------------------------------
# 3. Give time for VNC / camera adjustment
# ------------------------------------------------------------

echo
echo "[3/5] Gazebo ready."
echo
echo "Connect with VNC now and adjust the camera."
echo "Recording starts in ${CAMERA_WAIT_SEC} seconds."
echo

for ((i=CAMERA_WAIT_SEC; i>0; i--)); do
    printf "\rRecording starts in %2d sec... " "$i"
    sleep 1
done

echo
echo


# ------------------------------------------------------------
# 4. Start ONE continuous recording
# ------------------------------------------------------------

echo "[4/5] Starting recording..."

./scripts/record.sh start "$VIDEO_NAME"

RECORDING=1

sleep 2


# ------------------------------------------------------------
# 5. Run all Goal patterns
# ------------------------------------------------------------

SUCCESS_COUNT=0
FAIL_COUNT=0

for ((i=0; i<TOTAL; i++)); do

    RUN=$((i + 1))

    read -r ROW BIN LEVEL <<< "${GOALS[$i]}"

    echo
    echo "============================================================"
    echo " GOAL $RUN / $TOTAL"
    echo
    echo " ROW   = $ROW"
    echo " BIN   = $BIN"
    echo " LEVEL = $LEVEL"
    echo "============================================================"
    echo

    ./scripts/infer.sh "$ROW" "$BIN" "$LEVEL"

    RC=$?

    if [ "$RC" -eq 0 ]; then

        RESULT="SUCCESS"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        echo
        echo ">>> GOAL $RUN SUCCESS"

    else

        RESULT="FAILED"
        FAIL_COUNT=$((FAIL_COUNT + 1))

        echo
        echo ">>> GOAL $RUN FAILED (rc=$RC)"

    fi

    echo "$RUN,$ROW,$BIN,$LEVEL,$RESULT,$RC" >> "$RESULT_CSV"

    if [ "$RUN" -lt "$TOTAL" ]; then
        echo
        echo "Waiting ${BETWEEN_RUNS_SEC}s before next Goal..."
        sleep "$BETWEEN_RUNS_SEC"
    fi

done


# ------------------------------------------------------------
# Stop recording
# ------------------------------------------------------------

echo
echo "Stopping recording..."

./scripts/record.sh stop
RECORDING=0


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

VIDEO_FILE="$RESULT_DIR/${VIDEO_NAME}.mp4"

echo
echo "============================================================"
echo " 20-GOAL DEMO FINISHED"
echo "============================================================"
echo " Total    : $TOTAL"
echo " Success  : $SUCCESS_COUNT"
echo " Failed   : $FAIL_COUNT"
echo
echo " Success rate:"
python - <<PY
total = $TOTAL
success = $SUCCESS_COUNT
print(f"{success}/{total} = {100.0 * success / total:.1f}%")
PY

echo
echo "Video:"
echo "  $VIDEO_FILE"
echo
echo "Results:"
echo "  $RESULT_CSV"
echo "============================================================"


if command -v ffprobe >/dev/null 2>&1 && [ -f "$VIDEO_FILE" ]; then
    echo
    ffprobe \
        -v error \
        -show_entries format=duration,size \
        -of default=noprint_wrappers=1 \
        "$VIDEO_FILE"
fi

