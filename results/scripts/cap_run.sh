#!/bin/bash
# Capacity test: max users per context length for the full model (node1, data-parallel board) and for the
# two pipeline stages split at layer 33 (node1 [0,33), node2 [33,60)). Waits for all runs, prints results.
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30"
BIN=/home/orangepi/llama.cpp-pp/build-pp/bin/llama-cap-test
MODEL=models/gemma-4-31B-it-Q4_0.gguf
CTX="--ctx 1024,4096,8192,32768 --reserve-mib 1024 -t 8"
FULL="taskset -c 0,1,6-11 $BIN -m $MODEL $CTX --users 1,2,3,4,5,6,8,10,12,16"
FRONT="taskset -c 0,1,6-11 $BIN -m $MODEL -a 0 -b 33 $CTX --users 1,2,4,8,12,16,24,32,40,48,64,80"
BACK="taskset -c 0,1,6-11 $BIN -m $MODEL -a 33 -b 60 $CTX --users 1,2,4,8,12,16,24,32,40,48,64,80"

for n in node1 node2; do
    $SSH $n 'pgrep -x llama-batched-b >/dev/null || pgrep -x llama-pp-stage >/dev/null || pgrep -x llama-cap-test >/dev/null' && { echo "[cap] $n busy, abort"; exit 1; }
    $SSH $n "cd ~/llama.cpp; dd if=$MODEL iflag=nocache count=0 status=none"
done
echo "[cap] start ($(date -u +%T) UTC)"
$SSH node1 "cd ~/llama.cpp; setsid nohup bash -c \"bash ~/run_bench.sh cap_full '$FULL'; dd if=$MODEL iflag=nocache count=0 status=none; bash ~/run_bench.sh cap_front33 '$FRONT'\" > /dev/null 2>&1 < /dev/null & disown"
$SSH node2 "cd ~/llama.cpp; setsid nohup bash ~/run_bench.sh cap_back33 '$BACK' > /dev/null 2>&1 < /dev/null & disown"
sleep 30
for i in $(seq 1 180); do   # up to 60 min
    busy=0
    for n in node1 node2; do
        $SSH $n 'pgrep -f "^bash /home/orangepi/run_bench.sh cap_" >/dev/null' && busy=1
    done
    [ $busy -eq 0 ] && break
    sleep 20
done
echo "[cap] finished ($(date -u +%T) UTC)"
for n in node1 node2; do
    $SSH $n 'cd ~/llama.cpp/results; for f in $(ls -tr cap_*_*.log | tail -2); do case $f in cap_*) echo "===== $(hostname) $f"; grep -E "^# (start|exit)|killed" $f; cat ${f%.log}.out;; esac; done'
done
