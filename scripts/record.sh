#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VIDEO_DIR="$PROJECT_DIR/videos"

DISPLAY_NUM=":99.0"
RESOLUTION="1920x1080"
FPS=30

PID_FILE="/tmp/physical_ai_forklift_ffmpeg.pid"
NAME_FILE="/tmp/physical_ai_forklift_ffmpeg_name"

mkdir -p "$VIDEO_DIR"

case "$1" in

  start)

    NAME="${2:-forklift_demo}"

    if [ -f "$PID_FILE" ]; then
      OLD_PID=$(cat "$PID_FILE")

      if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Recording is already running."
        echo "PID: $OLD_PID"
        exit 1
      else
        rm -f "$PID_FILE"
        rm -f "$NAME_FILE"
      fi
    fi

    OUTPUT="$VIDEO_DIR/${NAME}.mp4"
    LOG="$VIDEO_DIR/${NAME}_ffmpeg.log"

    echo "Starting recording..."
    echo "Display : $DISPLAY_NUM"
    echo "Size    : $RESOLUTION"
    echo "FPS     : $FPS"
    echo "Output  : $OUTPUT"

    ffmpeg \
      -y \
      -f x11grab \
      -video_size "$RESOLUTION" \
      -framerate "$FPS" \
      -i "$DISPLAY_NUM" \
      -c:v libx264 \
      -preset veryfast \
      -crf 23 \
      -pix_fmt yuv420p \
      "$OUTPUT" \
      > "$LOG" 2>&1 &

    PID=$!

    echo "$PID" > "$PID_FILE"
    echo "$NAME" > "$NAME_FILE"

    sleep 2

    if kill -0 "$PID" 2>/dev/null; then
      echo
      echo "Recording started successfully."
      echo "PID: $PID"
      echo
      echo "Stop with:"
      echo "  $0 stop"
    else
      echo "ERROR: ffmpeg stopped unexpectedly."
      echo "Check:"
      echo "  $LOG"
      rm -f "$PID_FILE" "$NAME_FILE"
      exit 1
    fi
    ;;


  stop)

    if [ ! -f "$PID_FILE" ]; then
      echo "No active recording found."
      exit 1
    fi

    PID=$(cat "$PID_FILE")

    if [ -f "$NAME_FILE" ]; then
      NAME=$(cat "$NAME_FILE")
    else
      NAME="unknown"
    fi

    echo "Stopping recording..."
    echo "PID: $PID"

    if kill -0 "$PID" 2>/dev/null; then
      # Ctrl+C相当をffmpegへ送ってMP4を正常終了させる
      kill -INT "$PID"

      for i in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then
          break
        fi
        sleep 1
      done

      if kill -0 "$PID" 2>/dev/null; then
        echo "ffmpeg did not stop cleanly; terminating."
        kill "$PID" 2>/dev/null || true
      fi
    fi

    rm -f "$PID_FILE"
    rm -f "$NAME_FILE"

    echo
    echo "Recording stopped."
    echo "Video:"
    echo "  $VIDEO_DIR/${NAME}.mp4"
    ;;


  status)

    if [ -f "$PID_FILE" ]; then
      PID=$(cat "$PID_FILE")

      if kill -0 "$PID" 2>/dev/null; then
        echo "Recording is running."
        echo "PID: $PID"

        if [ -f "$NAME_FILE" ]; then
          NAME=$(cat "$NAME_FILE")
          echo "Video: $VIDEO_DIR/${NAME}.mp4"
        fi

        exit 0
      fi
    fi

    echo "Recording is not running."
    ;;


  *)

    echo "Usage:"
    echo
    echo "  $0 start [video_name]"
    echo "  $0 stop"
    echo "  $0 status"
    echo
    echo "Examples:"
    echo "  $0 start inference_row1_bin10_level3"
    echo "  $0 status"
    echo "  $0 stop"
    exit 1
    ;;

esac
