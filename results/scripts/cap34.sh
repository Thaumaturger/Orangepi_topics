#!/bin/bash
# Capacity at split 34 (front [0,34) on node1, back [34,60) on node2): q8_0 KV, then f16 KV.
# Same parameters as cap_q8.sh / cap_run.sh so the result compares directly with split 33.
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
MODEL=models/gemma-4-31B-it-Q4_0.gguf
CAP="taskset -c 0,1,6-11 /home/orangepi/llama.cpp-pp/build-pp/bin/llama-cap-test -m $MODEL --ctx 1024,4096,8192,32768 --reserve-mib 1024 -t 8"
UP="--users 1,2,4,8,12,16,24,32,40,48,64,80,96,128"
busy() { $SSH $1 "pgrep -f '^bash /home/orangepi/run_bench.sh' >/dev/null"; local rc=$?; [ $rc -eq 0 ] || [ $rc -eq 255 ]; }
for n in node1 node2; do
    $SSH $n 'pgrep -x llama-batched-b >/dev/null || pgrep -x llama-pp-stage >/dev/null || pgrep -x llama-cap-test >/dev/null' && { echo "[cap] $n busy, abort"; exit 1; }
done
run() {  # node name cmd
    $SSH $1 "cd ~/llama.cpp; dd if=$MODEL iflag=nocache count=0 status=none; setsid nohup bash ~/run_bench.sh $2 '$3' > /dev/null 2>&1 < /dev/null & disown"
}
wait_both() { sleep 20; for i in $(seq 1 120); do b=0; for n in node1 node2; do busy $n && b=1; done; [ $b -eq 0 ] && return; sleep 15; done; }
show() { $SSH $1 "cd ~/llama.cpp/results; f=\$(ls -t $2_*.log | head -1); echo \"== $1 \$f\"; grep -E '^# (exit)|killed' \$f; grep -E '^\[(load|result)\]|per user' \${f%.log}.out"; }
echo "[cap] start $(date -u +%T)"
run node1 capq8_front34 "$CAP -a 0 -b 34 --kv-type q8_0 $UP"
run node2 capq8_back34  "$CAP -a 34 -b 60 --kv-type q8_0 $UP"
wait_both; show node1 capq8_front34; show node2 capq8_back34
run node1 capf16_front34 "$CAP -a 0 -b 34 $UP"
run node2 capf16_back34  "$CAP -a 34 -b 60 $UP"
wait_both; show node1 capf16_front34; show node2 capf16_back34
echo "[cap] done $(date -u +%T)"
