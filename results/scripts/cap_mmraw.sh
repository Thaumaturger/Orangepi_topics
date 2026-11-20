#!/bin/bash
# Direction 1 (6.18): first stage loaded with mmap, capacity judged by MemAvailable as is (--avail-raw), swappiness 10.
# Weight pages that were copied into the repack buffers are never read again, so the kernel can drop them;
# the run fails if the kernel swaps instead (watchdog stops it at +512 MB swap). node1 only, q8_0 then q4_0 KV.
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
MODEL=models/gemma-4-31B-it-Q4_0.gguf
CAP="taskset -c 0,1,6-11 /home/orangepi/llama.cpp-pp/build-pp/bin/llama-cap-test -m $MODEL --ctx 1024,4096,8192,32768 --reserve-mib 1024 -t 8 -a 0 -b 33 --mmap --avail-raw"
UP="--users 1,2,4,8,12,16,24,32,40,48,64,80,96,128,160,192"
busy() { $SSH node1 "pgrep -f '^bash /home/orangepi/run_bench.sh' >/dev/null"; local rc=$?; [ $rc -eq 0 ] || [ $rc -eq 255 ]; }
$SSH node1 'pgrep -f "llama-(batched-bench|bench|pp-stage|cap-test|stage-test)" >/dev/null' && { echo "[cap] node1 busy, abort"; exit 1; }
show() { $SSH node1 "cd ~/llama.cpp/results; f=\$(ls -t $1_*.log | head -1); echo \"== node1 \$f\"; grep -E '^# (exit)|killed' \$f; grep -E '^\[(load|result)\]|per user' \${f%.log}.out; m=\${f%.log}.mem; echo \"swap_kb first \$(grep -o 'swap_kb=[0-9]*' \$m | head -1 | cut -d= -f2) max \$(grep -o 'swap_kb=[0-9]*' \$m | cut -d= -f2 | sort -n | tail -1); psi_full max \$(grep -o 'psi_full10=[0-9.]*' \$m | cut -d= -f2 | sort -n | tail -1)\""; }
for kv in q8_0 q4_0; do
    n=capmmraw_${kv}_front33
    $SSH node1 "cd ~/llama.cpp; dd if=$MODEL iflag=nocache count=0 status=none; setsid nohup bash ~/run_bench.sh $n '$CAP --kv-type $kv $UP' > /dev/null 2>&1 < /dev/null & disown"
    sleep 20; for i in $(seq 1 160); do busy || break; sleep 15; done
    show $n
done
echo "[cap] done $(date -u +%T)"
