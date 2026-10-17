#!/bin/bash
set -e

export DISPLAY=:99
export XDG_RUNTIME_DIR=/tmp/runtime-root

mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

echo "DISPLAY=$DISPLAY"

# 以前のプロセスが残っていたら終了
pkill -f "Xvfb :99" 2>/dev/null || true
pkill -f "x11vnc -display :99" 2>/dev/null || true
pkill -f "openbox" 2>/dev/null || true

sleep 1

echo "Starting Xvfb..."
Xvfb :99 -screen 0 1920x1080x24 \
  > /tmp/forklift_xvfb.log 2>&1 &

sleep 1

echo "Starting Openbox..."
DISPLAY=:99 openbox \
  > /tmp/forklift_openbox.log 2>&1 &

sleep 1

echo "Starting x11vnc..."
DISPLAY=:99 x11vnc \
  -display :99 \
  -forever \
  -nopw \
  -listen 0.0.0.0 \
  -xkb \
  > /tmp/forklift_x11vnc.log 2>&1 &

sleep 2

echo
echo "Display environment ready."
echo "DISPLAY=:99"
echo "Resolution: 1920x1080"
echo
echo "Processes:"
pgrep -af "Xvfb :99|x11vnc|openbox" || true
