#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/venv3.10/bin/python"
CARLA_BIN="/mnt/HDD2TB/carla/Dist/CARLA_Development_0.9.14-dirty/LinuxNoEditor/CarlaUE4.sh"
CARLA_LOG="/mnt/HDD2TB/carla/carla_server.log"
CARLA_LOG_MAX_BYTES=$((200 * 1024 * 1024))  # 200 MB cap

if [ ! -x "$VENV_PY" ]; then
	echo "Python venv not found or not executable: $VENV_PY"
	exit 1
fi

echo "Using Python: $VENV_PY"

WATCHDOG_PID=""

if pgrep -f "CarlaUE4" >/dev/null 2>&1; then
	echo "CARLA already running."
else
	echo "Starting CARLA..."
	# -RenderOffScreen -quality-level=Low
	"$CARLA_BIN" -RenderOffScreen >"$CARLA_LOG" 2>&1 &

	# Background watchdog: truncate log in-place if it exceeds the cap.
	# truncate -s 0 is safe even while CARLA holds the fd open.
	(
		while true; do
			sleep 30
			if [ -f "$CARLA_LOG" ]; then
				size=$(stat -c%s "$CARLA_LOG" 2>/dev/null || echo 0)
				if [ "$size" -gt "$CARLA_LOG_MAX_BYTES" ]; then
					echo "--- log truncated at $(date) (was ${size} bytes) ---" > "$CARLA_LOG"
				fi
			fi
		done
	) &
	WATCHDOG_PID=$!
fi

echo "Waiting for CARLA RPC on localhost:2000 ..."
for i in $(seq 1 60); do
	if "$VENV_PY" - <<'PY'
import carla
try:
	c = carla.Client("localhost", 2000)
	c.set_timeout(1.0)
	c.get_world()
	print("ready")
except Exception:
	raise SystemExit(1)
PY
	then
		echo "CARLA is ready."
		break
	fi

	if [ "$i" -eq 60 ]; then
		echo "CARLA did not become ready in time. Check $CARLA_LOG"
		exit 1
	fi
	sleep 1
done

if [ -f "$SCRIPT_DIR/train_td3.py" ]; then
	TRAIN_SCRIPT="$SCRIPT_DIR/train_td3.py"
elif [ -f "$SCRIPT_DIR/train_ddpg.py" ]; then
	echo "train_td3.py not found, falling back to train_ddpg.py"
	TRAIN_SCRIPT="$SCRIPT_DIR/train_ddpg.py"
else
	echo "No training script found (expected train_td3.py or train_ddpg.py)."
	exit 1
fi

exec "$VENV_PY" "$TRAIN_SCRIPT"
