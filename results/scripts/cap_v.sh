#!/bin/bash
# Capacity, directions 1 and 2 (6.18): mmap on the first stage (swappiness 10, avail = MemAvailable - RssFile)
# and q4_0 KV. Split 33. Same parameters as cap_q8.sh so the results compare with 6.13.
#  round A: node1 front q8_0 mmap    | node2 back q4_0
#  round B: node1 front q4_0 dio     | node2 full model q4_0 (data-parallel board)
#  round C: node1 front q4_0 mmap
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
MODEL=models/gemma-4-31B-it-Q4_0.gguf
CAP="taskset -c 0,1,6-11 /home/orangepi/llama.cpp-pp/build-pp/bin/llama-cap-test -m $MODEL --ctx 1024,4096,8192,32768 --reserve-mib 1024 -t 8"
UP="--users 1,2,4,8,12,16,24,32,40,48,64,80,96,128,160,192"
UF="--users 1,2,3,4,5,6,8,10,12,16,20,24,32"
busy() { $SSH $1 "pgrep -f '^bash /home/orangepi/run_bench.sh' >/dev/null"; local rc=$?; [ $rc -eq 0 ] || [ $rc -eq 255 ]; }
for n in node1 node2; do
    $SSH $n 'pgrep -f "llama-(batched-bench|bench|pp-stage|cap-test|stage-test)" >/dev/null' && { echo "[cap] $n busy, abort"; exit 1; }
    $SSH $n "cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | sort -u" | grep -qx performance || { echo "[cap] $n governor not performance"; exit 2; }
done
run() {  # node name cmd
    $SSH $1 "cd ~/llama.cpp; dd if=$MODEL iflag=nocache count=0 status=none; setsid nohup bash ~/run_bench.sh $2 '$3' > /dev/null 2>&1 < /dev/null & disown"
}
wait_both() { sleep 20; for i in $(seq 1 160); do b=0; for n in node1 node2; do busy $n && b=1; done; [ $b -eq 0 ] && return; sleep 15; done; }
show() { $SSH $1 "cd ~/llama.cpp/results; f=\$(ls -t $2_*.log | head -1); echo \"== $1 \$f\"; grep -E '^# (exit)|killed' \$f; grep -E '^\[(load|result)\]|per user' \${f%.log}.out; m=\${f%.log}.mem; echo \"swap_kb max \$(grep -o 'swap_kb=[0-9]*' \$m | cut -d= -f2 | sort -n | tail -1) first \$(grep -o 'swap_kb=[0-9]*' \$m | head -1 | cut -d= -f2)\""; }
echo "[cap] start $(date -u +%T)"
run node1 capq8mm2_front33 "$CAP -a 0 -b 33 --kv-type q8_0 --mmap $UP"
run node2 capq4_back33     "$CAP -a 33 -b 60 --kv-type q4_0 $UP"
wait_both; show node1 capq8mm2_front33; show node2 capq4_back33
run node1 capq4_front33 "$CAP -a 0 -b 33 --kv-type q4_0 $UP"
run node2 capq4_full    "$CAP --kv-type q4_0 $UF"
wait_both; show node1 capq4_front33; show node2 capq4_full
run node1 capq4mm_front33 "$CAP -a 0 -b 33 --kv-type q4_0 --mmap $UP"
wait_both; show node1 capq4mm_front33
echo "[cap] done $(date -u +%T)"
