#!/usr/bin/env bash
# Monitor a long-running training process and emit periodic status snapshots.
#
# Polls the given PID every INTERVAL seconds. Writes one structured line per
# tick to LOGFILE (also echoes to stdout). Exits cleanly when the PID is gone.
# Designed to be run in background alongside any long training job:
#
#     nohup bash scripts/monitor_training.sh <PID> [INTERVAL] [LOGFILE] &
#
# Each line captures: ISO timestamp | process etime/CPU%/MEM%/RSS | system RAM
# used/free/avail | swap used | disk used/total | cumulative OOM-kill count.
# Also writes a final summary line on exit.
#
# Trip-wire warnings (printed to stderr in red) fire if:
#   - process RSS exceeds 80% of physical RAM (OOM imminent)
#   - swap usage exceeds 50% of swap total
#   - new OOM-kill events appear in dmesg
#
# v1 written 2026-05-11 after the SPM 1M-docs/source run OOM'd at +50 min.

set -u

PID="${1:?usage: $0 <PID> [INTERVAL_SECONDS] [LOGFILE]}"
INTERVAL="${2:-300}"
LOGFILE="${3:-artifacts/monitor_$(date -u +%Y%m%d-%H%M%S)_pid${PID}.log}"

if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: PID $PID is not alive" >&2
    exit 1
fi

mkdir -p "$(dirname "$LOGFILE")"
PHYS_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
SWAP_KB=$(awk '/SwapTotal/ {print $2}' /proc/meminfo)
START_OOM=$(dmesg -T 2>/dev/null | grep -c "Out of memory:" || true)

header="ts=$(date -u --iso-8601=seconds) pid=$PID interval=${INTERVAL}s phys_gb=$((PHYS_KB/1024/1024)) swap_gb=$((SWAP_KB/1024/1024)) logfile=$LOGFILE"
echo "MONITOR START $header" | tee -a "$LOGFILE"

red() { printf "\033[1;31m%s\033[0m\n" "$*" >&2; }

while kill -0 "$PID" 2>/dev/null; do
    ts=$(date -u --iso-8601=seconds)

    # Process snapshot.
    proc=$(ps -p "$PID" -o etime=,pcpu=,pmem=,rss=,stat= 2>/dev/null)
    if [ -z "$proc" ]; then
        break
    fi
    etime=$(echo "$proc" | awk '{print $1}')
    pcpu=$(echo "$proc" | awk '{print $2}')
    pmem=$(echo "$proc" | awk '{print $3}')
    rss_kb=$(echo "$proc" | awk '{print $4}')
    pstat=$(echo "$proc" | awk '{print $5}')
    rss_gb=$(awk "BEGIN{printf \"%.1f\", $rss_kb/1024/1024}")

    # System.
    mem_used_gb=$(free -g | awk 'NR==2{print $3}')
    mem_free_gb=$(free -g | awk 'NR==2{print $4}')
    mem_avail_gb=$(free -g | awk 'NR==2{print $7}')
    swap_used_gb=$(free -g | awk 'NR==3{print $3}')

    # Disk.
    disk=$(df -h / | awk 'NR==2{printf "%s/%s(%s)", $3, $2, $5}')

    # OOM history.
    cur_oom=$(dmesg -T 2>/dev/null | grep -c "Out of memory:" || true)
    oom_delta=$((cur_oom - START_OOM))

    line="ts=$ts pid=$PID etime=$etime cpu=${pcpu}% mem=${pmem}% rss_gb=$rss_gb stat=$pstat sys_used_gb=$mem_used_gb sys_avail_gb=$mem_avail_gb swap_used_gb=$swap_used_gb disk=$disk new_oom_events=$oom_delta"
    echo "$line" | tee -a "$LOGFILE"

    # Trip-wires.
    pct_phys=$(awk "BEGIN{printf \"%.0f\", ($rss_kb*100)/$PHYS_KB}")
    if [ "$pct_phys" -ge 80 ]; then
        red "WARN: process RSS=${rss_gb}GB is ${pct_phys}% of physical RAM — OOM risk imminent"
    fi
    if [ "$SWAP_KB" -gt 0 ]; then
        pct_swap=$(awk "BEGIN{printf \"%.0f\", ($swap_used_gb*1024*1024*100)/$SWAP_KB}")
        if [ "$pct_swap" -ge 50 ]; then
            red "WARN: swap usage=${swap_used_gb}GB is ${pct_swap}% of total swap"
        fi
    fi
    if [ "$oom_delta" -gt 0 ]; then
        red "ALERT: $oom_delta new OOM-kill events in dmesg since monitor started"
    fi

    sleep "$INTERVAL"
done

end_ts=$(date -u --iso-8601=seconds)
final_oom=$(dmesg -T 2>/dev/null | grep -c "Out of memory:" || true)
oom_total=$((final_oom - START_OOM))

# Post-exit forensics: did the process die naturally or get killed?
last_dmesg_kill=$(dmesg -T 2>/dev/null | grep "Killed process $PID" | tail -1)
exit_reason="natural-exit"
if [ -n "$last_dmesg_kill" ]; then
    exit_reason="oom-killed"
fi

summary="MONITOR END ts=$end_ts pid=$PID exit_reason=$exit_reason oom_events_during_monitor=$oom_total"
echo "$summary" | tee -a "$LOGFILE"
if [ "$exit_reason" = "oom-killed" ]; then
    red "$last_dmesg_kill"
fi
