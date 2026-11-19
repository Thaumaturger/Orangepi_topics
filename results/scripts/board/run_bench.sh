#!/bin/bash
# usage: run_bench.sh <name> <command...>
# Runs <command> in ~/llama.cpp with a header log, stdout -> results/<name>_<ts>.out, stderr -> .log,
# and a memory watchdog writing .mem (kills on +512MB swap or PSI full avg10 > 30).
cd ~/llama.cpp || exit 1
mkdir -p results
NAME=$1; shift
CMD="$*"
BASE=results/${NAME}_$(date +%m%d_%H%M)
{
  echo "# node: $(hostname)"
  echo "# commit: $(git rev-parse --short HEAD)"
  echo "# start: $(date -Is)"
  echo "# governor: $(cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | sort -u | tr '\n' ' ')"
  echo "# mem: $(free -m | awk 'NR==2{print "used="$3"M cache="$6"M avail="$7"M"} NR==3{print "swap_used="$3"M"}' | tr '\n' ' ')"
  echo "# cmd: $CMD > $BASE.out"
} > "$BASE.log"
S0=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
bash -c "$CMD" > "$BASE.out" 2>> "$BASE.log" < /dev/null &
P=$!
# $P is the bash -c wrapper; find the real benchmark child if any
sleep 1
C=$(pgrep -P "$P" | head -1); [ -n "$C" ] && T=$C || T=$P
bash ~/wd_mem.sh "$T" "$S0" "$BASE.log" > "$BASE.mem" 2>&1 &
wait "$P"; RC=$?
echo "# exit=$RC end: $(date -Is)" >> "$BASE.log"
echo "$BASE"
