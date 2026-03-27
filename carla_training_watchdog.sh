#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARLA_BIN="/mnt/HDD2TB/carla/Dist/CARLA_Development_0.9.14-dirty/LinuxNoEditor/CarlaUE4.sh"
CARLA_LOG="$SCRIPT_DIR/carla_watchdog.log"
PYTHON_SCRIPT="$SCRIPT_DIR/train_td3.py"
VENV_PY="$SCRIPT_DIR/venv3.10/bin/python"

MEMORY_THRESHOLD_GB=12
MEMORY_CHECK_INTERVAL=5  # seconds

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

log_info() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*"
}

cleanup() {
    log_warn "Cleanup triggered, killing child processes..."
    # Kill background monitor
    if [ -n "${MONITOR_PID:-}" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
    # Interrupt Python training gracefully (SIGINT)
    if pgrep -f "train_td3.py" >/dev/null 2>&1; then
        pkill -INT -f "train_td3.py" || true
        sleep 5
    fi
    # Kill CARLA
    if pgrep -f "CarlaUE4" >/dev/null 2>&1; then
        pkill -9 -f "CarlaUE4" || true
    fi
    log_info "Cleanup complete"
}

trap cleanup EXIT INT TERM

find_carla_pids() {
    # Supports packaged CARLA (CarlaUE4) and source builds (UE4Editor with CarlaUE4 project).
    pgrep -x "CarlaUE4"
}

get_carla_memory_gb() {
    local pid
    pid="$(find_carla_pids)"

    if [ -z "$pid" ]; then
        echo "0"
        return
    fi

    # Normalize PID list (ps -p accepts comma-separated list)
    local pid_list
    pid_list=$(echo "$pid" | tr '\n' ',' | sed 's/,$//')

    # Sum RSS (in KB) for all CARLA PIDs and convert to GB (float with 2 decimals)
    local rss_kb
    rss_kb=$(ps -o rss= -p "$pid_list" 2>/dev/null | awk '{sum += $1} END {print sum+0}')
    if [ -z "$rss_kb" ] || [ "$rss_kb" = "0" ]; then
        echo "0"
        return
    fi

    local rss_gb
    rss_gb=$(echo "scale=2; $rss_kb/1024/1024" | bc -l)
    echo "$rss_gb"
}

monitor_memory() {
    # log_info "Memory monitor started (threshold: ${MEMORY_THRESHOLD_GB}GB, check interval: ${MEMORY_CHECK_INTERVAL}s)"
    
    while true; do
        sleep "$MEMORY_CHECK_INTERVAL"
        
        # Check if CARLA is still alive
        if [ -z "$(find_carla_pids)" ]; then
            log_info "CARLA process no longer running, exiting monitor"
            break
        fi
        
        local mem_gb=$(get_carla_memory_gb)
        
        # Show all running processes for debugging (first time)
        if [ "$mem_gb" = "0" ]; then
            log_warn "Memory detection returned 0GB. Running processes:"
            ps aux | grep -i carla | grep -v grep || true
            ps aux | grep -i unreal | grep -v grep || true
        fi
        
        log_info "CARLA memory: ${mem_gb}GB"
        
        # Compare as floats using bc
        if (( $(echo "$mem_gb >= $MEMORY_THRESHOLD_GB" | bc -l) )); then
            log_error "CARLA memory (${mem_gb}GB) exceeded threshold (${MEMORY_THRESHOLD_GB}GB), triggering restart"
            # Signal parent to restart by killing CARLA
            pkill -9 -f "CarlaUE4" || true
            break
        fi
    done
}

run_iteration() {
    local iteration=$1
    log_info "========== ITERATION $iteration START =========="
    
    # Start CARLA with memory cap
    log_info "Starting CARLA..."
    systemd-run --user --scope -p MemoryMax=16G \
        "$CARLA_BIN" -RenderOffScreen -quality-level=Medium \
        >"$CARLA_LOG" 2>&1 &
    
    local carla_pid=$!
    log_info "CARLA started with PID $carla_pid"
    
    # Wait for CARLA to be ready
    log_info "Waiting for CARLA to be ready..."
    local max_retries=60
    local retry=0
    while [ $retry -lt $max_retries ]; do
        if "$VENV_PY" - <<'PY' 2>/dev/null
import carla
try:
    c = carla.Client("localhost", 2000)
    c.set_timeout(1.0)
    c.get_world()
    print("ready")
except:
    raise SystemExit(1)
PY
        then
            log_info "CARLA is ready"
            break
        fi
        retry=$((retry + 1))
        if [ $retry -eq $max_retries ]; then
            log_error "CARLA did not become ready in time"
            return 1
        fi
        sleep 1
    done
    
    # Start memory monitor in background
    monitor_memory &
    MONITOR_PID=$!
    log_info "Memory monitor started with PID $MONITOR_PID"
    
    # Start training
    log_info "Starting training script..."
    "$VENV_PY" "$PYTHON_SCRIPT" &
    local train_pid=$!
    log_info "Training started with PID $train_pid"
    
    # Wait for training or memory limit trigger
    wait $train_pid 2>/dev/null || true
    
    # Kill monitor if still running
    if [ -n "${MONITOR_PID:-}" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
    
    # Send graceful interrupt and wait for Python process to fully stop
    if pgrep -f "train_td3.py" >/dev/null 2>&1; then
        log_info "Sending SIGINT to training script for graceful shutdown..."
        pkill -INT -f "train_td3.py" || true
        
        # Wait for process to actually terminate
        local max_wait=15
        local waited=0
        while pgrep -f "train_td3.py" >/dev/null 2>&1; do
            if [ $waited -ge $max_wait ]; then
                log_warn "Python script did not terminate gracefully in ${max_wait}s, force killing..."
                pkill -9 -f "train_td3.py" || true
                break
            fi
            log_info "Waiting for Python script to stop... (${waited}s/${max_wait}s)"
            sleep 1
            waited=$((waited + 1))
        done
        log_info "Python script fully stopped"
    fi
    
    # Kill CARLA
    pkill -9 -f "CarlaUE4" 2>/dev/null || true
    
    sleep 1
    log_info "========== ITERATION $iteration END =========="
}

main() {
    log_info "Starting CARLA training watchdog"
    log_info "Memory threshold: ${MEMORY_THRESHOLD_GB}GB"
    log_info "Memory check interval: ${MEMORY_CHECK_INTERVAL}s"
    log_info "Press Ctrl+C to stop"
    log_info ""
    
    local iteration=1
    while true; do
        run_iteration "$iteration"
        iteration=$((iteration + 1))
        sleep 2  # Brief pause before restarting
    done
}

main "$@"
