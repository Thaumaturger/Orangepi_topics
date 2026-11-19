#!/bin/bash
# Correctness check of a 3-stage chain on 2 boards: node1 first [0,20) -> node2 mid [20,40) -> node2 last [40,60).
# mid and last share node2 (4 big cores each), so only the verify pass and the agreement with the reference matter.
# Needs the mid-capable llama-pp-stage in build-pp on both boards.
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
MODEL=models/gemma-4-31B-it-Q4_0.gguf
BIN=/home/orangepi/llama.cpp-pp/build-pp/bin/llama-pp-stage
REF=/tmp/stage_ref_31b.bin
busy() { $SSH $1 "pgrep -f '^bash /home/orangepi/run_bench.sh $2' >/dev/null"; local rc=$?; [ $rc -eq 0 ] || [ $rc -eq 255 ]; }
for n in node1 node2; do
    $SSH $n 'pgrep -x llama-batched-b >/dev/null || pgrep -x llama-pp-stage >/dev/null || pgrep -x llama-cap-test >/dev/null' && { echo "[3s] $n busy, abort"; exit 1; }
done
echo "[3s] start $(date -u +%T)"
$SSH node2 "cd ~/llama.cpp; setsid nohup bash ~/run_bench.sh pp3s_last 'taskset -c 8-11 $BIN --role last -m $MODEL -s 40 -N 60 --listen 192.168.137.31:9000 --n-seq 8 -t 4' > /dev/null 2>&1 < /dev/null & disown"
sleep 2
$SSH node2 "cd ~/llama.cpp; setsid nohup bash ~/run_bench.sh pp3s_mid 'taskset -c 0,1,6,7 $BIN --role mid -m $MODEL -s 20 -e 40 --listen 192.168.137.31:9001 --next 192.168.137.31:9000 --n-seq 8 -t 4' > /dev/null 2>&1 < /dev/null & disown"
sleep 2
$SSH node1 "cd ~/llama.cpp; setsid nohup bash ~/run_bench.sh pp3s_first 'timeout 2400 taskset -c 0,1,6-11 $BIN --role first -m $MODEL -s 20 --next 192.168.137.31:9001 --ref $REF --n-seq 8 -n 32 -t 8 --configs 1x1,3x3,6x3' > /dev/null 2>&1 < /dev/null & disown"
sleep 30
for i in $(seq 1 200); do busy node1 pp3s_first || break; sleep 15; done
$SSH node2 'for p in $(pgrep -x llama-pp-stage); do kill $p; done'
sleep 3
$SSH node1 'cd ~/llama.cpp/results; f=$(ls -t pp3s_first_*.log | head -1); echo $f; grep -E "^# (start|exit)|killed" $f; grep -E "^\[(load|verify)\]|^\[config" ${f%.log}.out'
$SSH node2 'cd ~/llama.cpp/results; for k in mid last; do f=$(ls -t pp3s_${k}_*.log | head -1); echo $f; grep -E "^# exit|killed" $f | tail -1; grep -E "^\[(load|mid)\]" ${f%.log}.out; grep -iE "error|assert" $f | head -3; done'
echo "[3s] done $(date -u +%T)"
