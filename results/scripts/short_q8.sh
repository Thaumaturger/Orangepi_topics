#!/bin/bash
# Short context (29-token prompt + 32 generated), split 33: f16 vs q8_0 KV at 32, 48 and 64 users (M=2).
# Waits for the data-parallel run to finish first.
D=$(dirname "$0")
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30"
for i in $(seq 1 3); do
    b=0; for n in node1 node2; do $SSH $n 'pgrep -x llama-batched-b >/dev/null'; rc=$?; [ $rc -ne 1 ] && b=1; done
    [ $b -eq 0 ] && break; sleep 20
done
bash $D/pp_run.sh ppsf16 "--n-seq 64 --ctx-seq 256" "--configs 32x2,48x2,64x2" 33 2400
bash $D/pp_run.sh ppsq8  "--n-seq 64 --ctx-seq 256 --kv-type q8_0" "--configs 32x2,48x2,64x2" 33 2400
