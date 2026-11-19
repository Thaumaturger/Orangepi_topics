#!/bin/bash
# Inference tuning, single board (build-cpu-opt, full 31B model).
#  node1: batched decode (16 users) with OpenMP thread binding: none / GOMP_CPU_AFFINITY / affinity + OMP_WAIT_POLICY=ACTIVE
#  node2: prompt processing (pp1024) vs n_ubatch 256 / 512 / 1024
# The two boards run different tests at the same time; each board runs one test at a time.
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
MODEL=models/gemma-4-31B-it-Q4_0.gguf
BB="taskset -c 0,1,6-11 ./build-cpu-opt/bin/llama-batched-bench -m $MODEL -c 4096 -npp 32 -ntg 32 -npl 1,16,16 -lm dio -t 8"
PB="taskset -c 0,1,6-11 ./build-cpu-opt/bin/llama-bench -m $MODEL -p 1024 -n 0 -b 2048 -ub 256,512,1024 -r 2 -lm dio -t 8 -o json"
busy() { $SSH $1 "pgrep -f '^bash /home/orangepi/run_bench.sh' >/dev/null"; local rc=$?; [ $rc -eq 0 ] || [ $rc -eq 255 ]; }
for n in node1 node2; do
    $SSH $n 'pgrep -f "llama-(batched-bench|bench|pp-stage|cap-test|stage-test)" >/dev/null' && { echo "[tune] $n busy, abort"; exit 1; }
    $SSH $n "cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | sort -u" | grep -qx performance || { echo "[tune] $n governor not performance"; exit 2; }
done
run() {  # node name cmd
    $SSH $1 "cd ~/llama.cpp; dd if=$MODEL iflag=nocache count=0 status=none; setsid nohup bash ~/run_bench.sh $2 '$3' > /dev/null 2>&1 < /dev/null & disown"
}
wait_node() { sleep 20; for i in $(seq 1 240); do busy $1 || return; sleep 15; done; }
show() { $SSH $1 "cd ~/llama.cpp/results; f=\$(ls -t $2_*.log | head -1); echo \"== $1 \$f\"; grep -E '^# (exit|cmd)|killed' \$f | cut -c1-260; cat \${f%.log}.out | grep -E '^\|' ; m=\${f%.log}.mem; echo \"tmax \$(grep -o 'tmax=[0-9]*' \$m | cut -d= -f2 | sort -n | tail -1)\""; }
echo "[tune] start $(date -u +%T)"
run node2 tune_ub "$PB"
run node1 tune_base "$BB"
wait_node node1; show node1 tune_base
run node1 tune_aff "GOMP_CPU_AFFINITY=0,1,6-11 $BB"
wait_node node1; show node1 tune_aff
run node1 tune_affact "GOMP_CPU_AFFINITY=0,1,6-11 OMP_WAIT_POLICY=ACTIVE $BB"
wait_node node1; show node1 tune_affact
wait_node node2
$SSH node2 "cd ~/llama.cpp/results; f=\$(ls -t tune_ub_*.log | head -1); echo \"== node2 \$f\"; grep -E '^# exit' \$f; python3 -c \"import json,sys; [print('ub', r['n_ubatch'], 'pp', r['n_prompt'], 'avg %.2f' % r['avg_ts'], 'samples', ['%.2f' % x for x in r['samples_ts']]) for r in json.load(open(sys.argv[1]))]\" \${f%.log}.out"
echo "[tune] done $(date -u +%T)"
