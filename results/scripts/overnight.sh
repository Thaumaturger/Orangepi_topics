#!/bin/bash
# Overnight queue:
#  1) data parallel, short context, both boards at once: 8 and 16 users per board, 3 runs each (one batched-bench run)
#  2) pipeline split 33, short context, f16: 16x2 and 32x2, 3 runs each (one load)
#  3) pipeline split 33, own 4K prefix per user, q8_0 KV, up to 32 users: 8x2 x3, 16x2 x3, 24x2, 32x2 (prefixes are
#     filled once, incrementally; repeated configs reuse them)
D=$(dirname "$0")
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
MODEL=models/gemma-4-31B-it-Q4_0.gguf
busy() { $SSH $1 "pgrep -f '^bash /home/orangepi/run_bench.sh $2' >/dev/null"; local rc=$?; [ $rc -eq 0 ] || [ $rc -eq 255 ]; }
for n in node1 node2; do
    $SSH $n 'pgrep -x llama-batched-b >/dev/null || pgrep -x llama-pp-stage >/dev/null || pgrep -x llama-cap-test >/dev/null' && { echo "[on] $n busy, abort"; exit 1; }
    $SSH $n "cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | sort -u" | grep -qx performance || { echo "[on] $n governor not performance, abort"; exit 2; }
done

echo "[on] === 1) data parallel short repeats ($(date -u +%T) UTC)"
DP="taskset -c 0,1,6-11 ./build-cpu-opt/bin/llama-batched-bench -m $MODEL -c 8192 -npp 128 -ntg 32 -npl 1,8,8,8,16,16,16 -lm dio -t 8"
for n in node1 node2; do
    $SSH $n "cd ~/llama.cpp; dd if=$MODEL iflag=nocache count=0 status=none; setsid nohup bash ~/run_bench.sh dprep_${n} '$DP' > /dev/null 2>&1 < /dev/null & disown"
done
sleep 30
for i in $(seq 1 180); do b=0; for n in node1 node2; do busy $n dprep_ && b=1; done; [ $b -eq 0 ] && break; sleep 20; done
for n in node1 node2; do
    echo "===== $n"
    $SSH $n 'cd ~/llama.cpp/results; f=$(ls -t dprep_*.log | head -1); echo $f; grep -E "^# (start|exit)|killed" $f; grep -E "^\|" ${f%.log}.out'
done

echo "[on] === 2) pipeline short repeats"
bash $D/pp_run.sh pprep "--n-seq 32 --ctx-seq 256" "--configs 16x2,16x2,16x2,32x2,32x2,32x2" 33 2400

echo "[on] === 3) pipeline long q8_0, up to 32 users"
bash $D/pp_run.sh pp4kq8L "--n-seq 32 --ctx-seq 4096 --kv-type q8_0" "--prefix 4000 --distinct --configs 8x2,8x2,8x2,16x2,16x2,16x2,24x2,32x2" 33 16200
echo "[on] all done ($(date -u +%T) UTC)"
