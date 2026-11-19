#!/bin/bash
# Pipeline (split 33) long-context throughput with a distinct 4000-token prefix per sequence (own KV cells),
# filled incrementally for configs 1x1, 4x2, 8x2, 16x2. An ssh failure counts as busy.
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
MODEL=models/gemma-4-31B-it-Q4_0.gguf
BIN=/home/orangepi/llama.cpp-pp/build-pp/bin/llama-pp-stage
REF=/home/orangepi/stage_ref_31b.bin
busy() { $SSH $1 "pgrep -f '^bash /home/orangepi/run_bench.sh $2' >/dev/null"; local rc=$?; [ $rc -eq 0 ] || [ $rc -eq 255 ]; }
# wait for the reference run (llama-stage-test) on node1 to finish
for i in $(seq 1 90); do $SSH node1 "pgrep -x llama-stage-test >/dev/null"; rc=$?; [ $rc -eq 1 ] && break; sleep 20; done
$SSH node1 'cd ~/llama.cpp/results; f=$(ls -t stage_ref31b_*.log | head -1); grep -E "^# exit" $f'
for i in $(seq 1 120); do ok=1; for n in node1 node2; do $SSH $n "cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | sort -u" | grep -qx performance || ok=0; done; [ $ok -eq 1 ] && break; [ $i -eq 1 ] && echo "[lcq] waiting for governor = performance"; sleep 15; done; [ $ok -eq 1 ] || { echo "[lcq] governor not performance, abort"; exit 2; }
for n in node1 node2; do
    $SSH $n 'pgrep -x llama-batched-b >/dev/null || pgrep -x llama-pp-stage >/dev/null || pgrep -x llama-cap-test >/dev/null' && { echo "[lcq] $n busy, abort"; exit 1; }
done
$SSH node1 "test -s $REF" || { echo "[lcq] $REF missing on node1"; exit 3; }
echo "[lcq] === pipeline, split 33, distinct 4K prefix, q8_0 KV ($(date -u +%T) UTC)"
for n in node1 node2; do $SSH $n "cd ~/llama.cpp; dd if=$MODEL iflag=nocache count=0 status=none"; done
$SSH node2 "cd ~/llama.cpp; setsid nohup bash ~/run_bench.sh pp4kq8b_last 'taskset -c 0,1,6-11 $BIN --role last -m $MODEL -s 33 -N 60 --listen 192.168.137.31:9000 --n-seq 16 --ctx-seq 4096 --kv-type q8_0 -t 8' > /dev/null 2>&1 < /dev/null & disown"
sleep 2
$SSH node1 "cd ~/llama.cpp; setsid nohup bash ~/run_bench.sh pp4kq8b_first 'timeout 10800 taskset -c 0,1,6-11 $BIN --role first -m $MODEL -s 33 --next 192.168.137.31:9000 --ref $REF --n-seq 16 --ctx-seq 4096 --prefix 4000 --distinct --kv-type q8_0 -n 32 -t 8 --configs 8x2,16x2' > /dev/null 2>&1 < /dev/null & disown"
sleep 30
for i in $(seq 1 600); do   # up to 200 min
    busy node1 pp4kq8b_first || break; sleep 20
done
$SSH node2 'p=$(pgrep -x llama-pp-stage); [ -n "$p" ] && kill $p'
$SSH node1 'cd ~/llama.cpp/results; f=$(ls -t pp4kq8b_first_*.log | head -1); echo $f; grep -E "^# (start|exit)|killed" $f; cat ${f%.log}.out'
$SSH node2 'cd ~/llama.cpp/results; f=$(ls -t pp4kq8b_last_*.log | head -1); echo $f; grep -E "^# (start|exit)|killed|tmax" $f | tail -3; tail -5 ${f%.log}.out'
echo "[lcq] all done ($(date -u +%T) UTC)"
