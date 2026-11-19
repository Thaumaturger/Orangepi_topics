#!/bin/bash
# One two-stage pipeline run (split at layer S): starts the last stage on node2, the first stage on node1,
# waits for the first stage to finish, stops the last stage, prints the results.
# usage: pp_run.sh <name> "<common args: --n-seq N --ctx-seq C [--kv-type T]>" "<first-only args: --configs ... [--prefix P --distinct]>" [S] [timeout_s]
# optional: PP_ENV="VAR=value ..." is put in front of both stage commands (e.g. OpenMP settings)
NAME=$1; COMMON=$2; FIRST=$3; S=${4:-33}; TMO=${5:-5400}; ENVP=${PP_ENV:-}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
MODEL=models/gemma-4-31B-it-Q4_0.gguf
BIN=/home/orangepi/llama.cpp-pp/build-pp/bin/llama-pp-stage
REF=/home/orangepi/stage_ref_31b.bin
busy() { $SSH $1 "pgrep -f '^bash /home/orangepi/run_bench.sh $2' >/dev/null"; local rc=$?; [ $rc -eq 0 ] || [ $rc -eq 255 ]; }
for n in node1 node2; do
    $SSH $n 'pgrep -x llama-batched-b >/dev/null || pgrep -x llama-pp-stage >/dev/null || pgrep -x llama-cap-test >/dev/null || pgrep -x llama-stage-tes >/dev/null' && { echo "[pp] $n busy, abort"; return 1 2>/dev/null || exit 1; }
    $SSH $n "cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | sort -u" | grep -qx performance || { echo "[pp] $n governor not performance, abort"; return 2 2>/dev/null || exit 2; }
done
$SSH node1 "test -s $REF" || { echo "[pp] $REF missing on node1"; return 3 2>/dev/null || exit 3; }
echo "[pp] === $NAME: ${ENVP:+env $ENVP }-s $S $COMMON $FIRST ($(date -u +%T) UTC)"
for n in node1 node2; do $SSH $n "cd ~/llama.cpp; dd if=$MODEL iflag=nocache count=0 status=none"; done
$SSH node2 "cd ~/llama.cpp; setsid nohup bash ~/run_bench.sh ${NAME}_last '$ENVP taskset -c 0,1,6-11 $BIN --role last -m $MODEL -s $S -N 60 --listen 192.168.137.31:9000 $COMMON -t 8' > /dev/null 2>&1 < /dev/null & disown"
sleep 2
$SSH node1 "cd ~/llama.cpp; setsid nohup bash ~/run_bench.sh ${NAME}_first '$ENVP timeout $TMO taskset -c 0,1,6-11 $BIN --role first -m $MODEL -s $S --next 192.168.137.31:9000 --ref $REF $COMMON $FIRST -n 32 -t 8' > /dev/null 2>&1 < /dev/null & disown"
sleep 30
for i in $(seq 1 $((TMO / 20 + 60))); do busy node1 ${NAME}_first || break; sleep 20; done
$SSH node2 'p=$(pgrep -x llama-pp-stage); [ -n "$p" ] && kill $p'
sleep 5
$SSH node1 "cd ~/llama.cpp/results; f=\$(ls -t ${NAME}_first_*.log | head -1); echo \$f; grep -E '^# (start|exit)|killed' \$f; grep -E '^\[(ctx|verify)\]|^\[config' \${f%.log}.out; m=\${f%.log}.mem; echo \"min avail MiB \$(grep -o 'avail_kb=[0-9]*' \$m | cut -d= -f2 | sort -n | head -1 | awk '{print int(\$1/1024)}') tmax \$(grep -o 'tmax=[0-9]*' \$m | cut -d= -f2 | sort -n | tail -1)\""
$SSH node2 "cd ~/llama.cpp/results; f=\$(ls -t ${NAME}_last_*.log | head -1); echo \$f; grep -E '^# (start|exit)|killed' \$f | tail -2; m=\${f%.log}.mem; echo \"min avail MiB \$(grep -o 'avail_kb=[0-9]*' \$m | cut -d= -f2 | sort -n | head -1 | awk '{print int(\$1/1024)}') tmax \$(grep -o 'tmax=[0-9]*' \$m | cut -d= -f2 | sort -n | tail -1)\""
echo "[pp] $NAME done ($(date -u +%T) UTC)"
