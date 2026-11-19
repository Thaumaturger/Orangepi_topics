#!/bin/bash
# usage: wd.sh <pid> <swap_baseline_kb> <log_file>
# Kills <pid> if swap grows > 1 GB over baseline, or PSI memory full avg10 > 30 for 4 consecutive samples (~60 s).
P=$1; S0=$2; LOG=$3
HI=0
while kill -0 "$P" 2>/dev/null; do
  SW=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
  AV=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  PF=$(awk '/^full/{split($2,a,"="); print a[2]}' /proc/pressure/memory)
  CT=$(ps -o time= -p "$P")
  if [ "${PF%.*}" -gt 30 ]; then HI=$((HI + 1)); else HI=0; fi
  TM=$(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1)
  echo "$(date +%T) swap_kb=$SW avail_kb=$AV psi_full10=$PF cpu=$CT hi=$HI tmax=$TM"; sync
  if [ $((SW - S0)) -gt 1048576 ] || [ "$HI" -ge 4 ]; then
    echo "WATCHDOG KILL"
    kill "$P"
    echo "# killed by watchdog $(date -Is) swap_kb=$SW psi_full10=$PF hi=$HI" >> "$LOG"
  fi
  sleep 15
done
echo "$(date +%T) target exited"
