#!/bin/bash
# Starts the same llama-batched-bench on node1 and node2 at the same moment (data-parallel baseline),
# waits until both finish, then prints each board's results and exit line.
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30"
MODEL=models/gemma-4-31B-it-Q4_0.gguf
CMD="taskset -c 0,1,6-11 ./build-cpu-opt/bin/llama-batched-bench -m $MODEL -c 4096 -npp 128 -ntg 32 -npl 1,1,2,4,8,16 -lm dio -t 8"

for n in node1 node2; do
    $SSH $n 'pgrep -x llama-batched-b >/dev/null || pgrep -x llama-pp-stage >/dev/null || pgrep -x llama-stage-tes >/dev/null' && { echo "[dp] $n busy, abort"; exit 1; }
    $SSH $n "cd ~/llama.cpp; dd if=$MODEL iflag=nocache count=0 status=none"
done
echo "[dp] starting on both boards ($(date -u +%T) UTC)"
for n in node1 node2; do
    $SSH $n "cd ~/llama.cpp; setsid nohup bash ~/run_bench.sh dp_${n} '$CMD' > /dev/null 2>&1 < /dev/null & disown"
done
sleep 20
for i in $(seq 1 180); do   # up to 60 min
    busy=0
    for n in node1 node2; do
        $SSH $n 'pgrep -f "^bash /home/orangepi/run_bench.sh dp_" >/dev/null' && busy=1
    done
    [ $busy -eq 0 ] && break
    sleep 20
done
echo "[dp] finished ($(date -u +%T) UTC)"
for n in node1 node2; do
    echo "===== $n"
    $SSH $n 'cd ~/llama.cpp/results; f=$(ls -t dp_*_*.log | head -1); echo $f; grep -E "^# (start|exit)" $f; grep -E "^\|" ${f%.log}.out'
done
