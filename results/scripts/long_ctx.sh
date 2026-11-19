#!/bin/bash
# Long-context throughput (about 4K tokens per user):
#  1) data parallel: both boards at once, llama-batched-bench with a shared 4000-token prompt copied into each
#     sequence's own KV stream (-pps, non-unified KV), 1 and 4 users per board
#  2) pipeline split at layer 33: llama-pp-stage with a shared 4000-token prefix, 1x1, 8x2, 16x2
# An ssh failure counts as busy, so a slow board is not mistaken for a finished one.
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
MODEL=models/gemma-4-31B-it-Q4_0.gguf
BIN=/home/orangepi/llama.cpp-pp/build-pp/bin/llama-pp-stage
REF=/tmp/stage_ref_31b.bin
DP="taskset -c 0,1,6-11 ./build-cpu-opt/bin/llama-batched-bench -m $MODEL -c 16384 -npp 4000 -ntg 32 -npl 1,4 -pps -lm dio -t 8"

busy() {   # $1 node, $2 run_bench name prefix; returns 0 while running or unreachable
    $SSH $1 "pgrep -f '^bash /home/orangepi/run_bench.sh $2' >/dev/null"
    local rc=$?
    [ $rc -eq 0 ] || [ $rc -eq 255 ]
}

for n in node1 node2; do
    $SSH $n 'pgrep -x llama-batched-b >/dev/null || pgrep -x llama-pp-stage >/dev/null || pgrep -x llama-cap-test >/dev/null' && { echo "[lc] $n busy, abort"; exit 1; }
done
$SSH node1 "test -s $REF" || { echo "[lc] $REF missing on node1"; exit 3; }

echo "[lc] === data parallel, 4K context ($(date -u +%T) UTC)"
for n in node1 node2; do
    $SSH $n "cd ~/llama.cpp; dd if=$MODEL iflag=nocache count=0 status=none; setsid nohup bash ~/run_bench.sh dp4k_${n} '$DP' > /dev/null 2>&1 < /dev/null & disown"
done
sleep 30
for i in $(seq 1 240); do   # up to 80 min
    b=0; for n in node1 node2; do busy $n dp4k_ && b=1; done
    [ $b -eq 0 ] && break; sleep 20
done
for n in node1 node2; do
    echo "===== $n"
    $SSH $n 'cd ~/llama.cpp/results; f=$(ls -t dp4k_*.log | head -1); echo $f; grep -E "^# (start|exit)|killed" $f; grep -E "^\|" ${f%.log}.out'
done

echo "[lc] === pipeline, split 33, 4K context ($(date -u +%T) UTC)"
for n in node1 node2; do $SSH $n "cd ~/llama.cpp; dd if=$MODEL iflag=nocache count=0 status=none"; done
$SSH node2 "cd ~/llama.cpp; setsid nohup bash ~/run_bench.sh pp4k_last 'taskset -c 0,1,6-11 $BIN --role last -m $MODEL -s 33 -N 60 --listen 192.168.137.31:9000 --n-seq 16 --ctx-seq 4096 -t 8' > /dev/null 2>&1 < /dev/null & disown"
sleep 2
$SSH node1 "cd ~/llama.cpp; setsid nohup bash ~/run_bench.sh pp4k_first 'timeout 3600 taskset -c 0,1,6-11 $BIN --role first -m $MODEL -s 33 --next 192.168.137.31:9000 --ref $REF --n-seq 16 --ctx-seq 4096 --prefix 4000 -n 32 -t 8 --configs 1x1,8x2,16x2' > /dev/null 2>&1 < /dev/null & disown"
sleep 30
for i in $(seq 1 240); do
    busy node1 pp4k_first || break; sleep 20
done
$SSH node2 'p=$(pgrep -x llama-pp-stage); [ -n "$p" ] && kill $p'
$SSH node1 'cd ~/llama.cpp/results; f=$(ls -t pp4k_first_*.log | head -1); echo $f; grep -E "^# (start|exit)|killed" $f; cat ${f%.log}.out'
echo "[lc] all done ($(date -u +%T) UTC)"
