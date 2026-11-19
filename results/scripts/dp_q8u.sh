#!/bin/bash
# Data parallel, q8_0 KV, 4K per user, unified KV with each user prefilled separately (no -pps), like the pipeline: both boards at once,
# llama-batched-bench with 1, 4 and 8 users per board. Waits for any running pipeline stage to finish first.
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
MODEL=models/gemma-4-31B-it-Q4_0.gguf
DP="taskset -c 0,1,6-11 ./build-cpu-opt/bin/llama-batched-bench -m $MODEL -c 32768 -npp 4000 -ntg 32 -npl 4,8 -kvu -ctk q8_0 -ctv q8_0 -fa on -lm dio -t 8"
busy() { $SSH $1 "pgrep -f '^bash /home/orangepi/run_bench.sh $2' >/dev/null"; local rc=$?; [ $rc -eq 0 ] || [ $rc -eq 255 ]; }
for i in $(seq 1 240); do   # wait up to 80 min for the pipeline run to end
    b=0; for n in node1 node2; do $SSH $n 'pgrep -x llama-pp-stage >/dev/null'; rc=$?; [ $rc -ne 1 ] && b=1; done
    [ $b -eq 0 ] && break; sleep 20
done
for n in node1 node2; do
    $SSH $n 'pgrep -x llama-batched-b >/dev/null || pgrep -x llama-pp-stage >/dev/null || pgrep -x llama-cap-test >/dev/null' && { echo "[dpqu] $n busy, abort"; exit 1; }
    $SSH $n "cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | sort -u" | grep -qx performance || { echo "[dpqu] $n governor not performance, abort"; exit 2; }
done
echo "[dpqu] === data parallel, q8_0 KV, 4K own KV ($(date -u +%T) UTC)"
for n in node1 node2; do
    $SSH $n "cd ~/llama.cpp; dd if=$MODEL iflag=nocache count=0 status=none; setsid nohup bash ~/run_bench.sh dpq8u_${n} '$DP' > /dev/null 2>&1 < /dev/null & disown"
done
sleep 30
for i in $(seq 1 360); do   # up to 120 min
    b=0; for n in node1 node2; do busy $n dpq8u_ && b=1; done
    [ $b -eq 0 ] && break; sleep 20
done
for n in node1 node2; do
    echo "===== $n"
    $SSH $n 'cd ~/llama.cpp/results; f=$(ls -t dpq8u_*.log | head -1); echo $f; grep -E "^# (start|exit)|killed" $f; grep -E "^\|" ${f%.log}.out; m=${f%.log}.mem; echo "min avail MiB: $(grep -o "avail_kb=[0-9]*" $m | cut -d= -f2 | sort -n | head -1 | awk "{print int(\$1/1024)}")"'
done
echo "[dpqu] done ($(date -u +%T) UTC)"
