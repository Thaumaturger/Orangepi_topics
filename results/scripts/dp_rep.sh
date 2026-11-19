#!/bin/bash
# Data parallel, short context, both boards at once: 8 and 16 users per board, 3 runs each (one batched-bench run).
# -c 4096 as in 6.11: with -c 8192 and 16 non-unified streams the per-stream sliding-window cache doubles and the load runs out of memory.
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
MODEL=models/gemma-4-31B-it-Q4_0.gguf
busy() { $SSH $1 "pgrep -f '^bash /home/orangepi/run_bench.sh $2' >/dev/null"; local rc=$?; [ $rc -eq 0 ] || [ $rc -eq 255 ]; }
for n in node1 node2; do
    $SSH $n 'pgrep -x llama-batched-b >/dev/null || pgrep -x llama-pp-stage >/dev/null || pgrep -x llama-cap-test >/dev/null' && { echo "[dp] $n busy, abort"; exit 1; }
    $SSH $n "cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | sort -u" | grep -qx performance || { echo "[dp] $n governor not performance, abort"; exit 2; }
done
DP="taskset -c 0,1,6-11 ./build-cpu-opt/bin/llama-batched-bench -m $MODEL -c 4096 -npp 128 -ntg 32 -npl 1,8,8,8,16,16,16 -lm dio -t 8"
echo "[dp] start $(date -u +%T) UTC"
for n in node1 node2; do
    $SSH $n "cd ~/llama.cpp; dd if=$MODEL iflag=nocache count=0 status=none; setsid nohup bash ~/run_bench.sh dprep2_${n} '$DP' > /dev/null 2>&1 < /dev/null & disown"
done
sleep 30
for i in $(seq 1 180); do b=0; for n in node1 node2; do busy $n dprep2_ && b=1; done; [ $b -eq 0 ] && break; sleep 20; done
for n in node1 node2; do
    echo "===== $n"
    $SSH $n 'cd ~/llama.cpp/results; f=$(ls -t dprep2_*.log | head -1); echo $f; grep -E "^# (start|exit)|killed|min avail" $f; tail -3 ${f%.log}.mem 2>/dev/null; grep -E "^\|" ${f%.log}.out'
done
echo "[dp] done $(date -u +%T) UTC"
