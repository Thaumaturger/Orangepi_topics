#!/bin/bash
# Regenerates the reference on node1 if missing, waits for both boards to be on the performance governor,
# then runs the pipeline experiments back to back. Prints one status line per step; full logs stay on the boards.

SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30"
BIN=/home/orangepi/llama.cpp-pp/build-pp/bin/llama-pp-stage
MODEL=models/gemma-4-31B-it-Q4_0.gguf
REF=/home/orangepi/stage_ref_31b.bin

# the reference lives in node1:/tmp and is lost on reboot; its content does not depend on CPU frequency
if ! $SSH node1 "test -s $REF"; then
    echo "[orch] $REF missing on node1, regenerating ($(date -u +%T) UTC)"
    $SSH node1 "cd ~/llama.cpp; bash ~/run_bench.sh stage_ref31b 'taskset -c 0,1,6-11 /home/orangepi/llama.cpp-pp/build-pp/bin/llama-stage-test -m $MODEL -n 32 -t 8 --mode ref --ref $REF'" > /dev/null 2>&1
    $SSH node1 "test -s $REF" || { echo "[orch] failed to regenerate $REF"; exit 3; }
    echo "[orch] reference regenerated ($(date -u +%T) UTC)"
fi

gov_ok() {
    for n in node1 node2; do
        g=$($SSH $n 'cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | sort -u | tr "\n" " "' 2>/dev/null)
        [ "$g" = "performance " ] || return 1
    done
    return 0
}

echo "[orch] waiting for performance governor on both boards ($(date -u +%T) UTC)"
for i in $(seq 1 240); do   # up to 60 min
    if gov_ok; then break; fi
    sleep 15
done
if ! gov_ok; then echo "[orch] governor still not performance after 60 min, giving up"; exit 2; fi
echo "[orch] governor OK ($(date -u +%T) UTC)"

run_exp() {   # $1 name, $2 split, $3 configs
    local name=$1 split=$2 configs=$3
    echo "[orch] === $name: split $split, configs $configs ($(date -u +%T) UTC)"
    $SSH node1 'pgrep -x llama-pp-stage >/dev/null' && { echo "[orch] node1 busy, abort"; return 1; }
    $SSH node2 'p=$(pgrep -x llama-pp-stage); [ -n "$p" ] && kill $p; sleep 1' 2>/dev/null
    for n in node1 node2; do $SSH $n "cd ~/llama.cpp; dd if=$MODEL iflag=nocache count=0 status=none" 2>/dev/null; done
    $SSH node2 "cd ~/llama.cpp; setsid nohup bash ~/run_bench.sh ${name}_last 'taskset -c 0,1,6-11 $BIN --role last -m $MODEL -s $split -N 60 --listen 192.168.137.31:9000 --n-seq 32 -t 8' > /dev/null 2>&1 < /dev/null & disown"
    sleep 2
    # run_bench.sh blocks until the first stage exits; timeout stops it if node2 stops answering
    $SSH node1 "cd ~/llama.cpp; bash ~/run_bench.sh ${name}_first 'timeout 2400 taskset -c 0,1,6-11 $BIN --role first -m $MODEL -s $split --next 192.168.137.31:9000 --ref $REF --n-seq 32 -n 32 -t 8 --configs $configs'" 2>/dev/null
    local base
    base=$($SSH node1 "ls -t ~/llama.cpp/results/${name}_first_*.out | head -1")
    $SSH node2 'p=$(pgrep -x llama-pp-stage); [ -n "$p" ] && kill $p' 2>/dev/null
    echo "[orch] $name done: $base"
    $SSH node1 "cat $base; grep '^# exit' ${base%.out}.log"
}

run_exp pp6 30 "32x2"
run_exp pp5 33 "1x1,2x2,16x2,32x2"
echo "[orch] all done ($(date -u +%T) UTC)"
