#!/bin/bash
# One-time board setup. Run on each board with sudo:  sudo bash ~/board_setup.sh node1   (or node2)
#  1. performance governor at every boot (systemd oneshot; replaces running ~/set_perf.sh after each boot)
#  2. vm.swappiness 100 -> 10 (node1 has a 4 GB swapfile; with 100 the kernel swaps out llama.cpp's weights
#     instead of dropping page cache). Persistent in /etc/sysctl.d/99-sysctl.conf, applied now as well.
#  3. cluster time: node1 serves NTP to 192.168.137.0/24 (also without internet); node2 syncs from node1.
# Safe to run again. Undo: systemctl disable --now cpu-performance; rm /etc/systemd/system/cpu-performance.service;
#   set vm.swappiness=100 back in /etc/sysctl.d/99-sysctl.conf; rm /etc/chrony/conf.d/cluster.conf; systemctl restart chrony
set -e
ROLE=$1
[ "$ROLE" = node1 ] || [ "$ROLE" = node2 ] || { echo "usage: sudo bash $0 node1|node2"; exit 1; }
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }

# 1. performance governor at boot
cat > /etc/systemd/system/cpu-performance.service <<'EOF'
[Unit]
Description=Set all CPU frequency policies to the performance governor

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo performance > "$f"; done'

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now cpu-performance.service
echo "[1] governor: $(cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | sort | uniq -c | xargs)"

# 2. swappiness
if grep -q '^vm.swappiness=' /etc/sysctl.d/99-sysctl.conf; then
    sed -i 's/^vm.swappiness=.*/vm.swappiness=10/' /etc/sysctl.d/99-sysctl.conf
else
    echo 'vm.swappiness=10' >> /etc/sysctl.d/99-sysctl.conf
fi
sysctl -q -w vm.swappiness=10
echo "[2] swappiness: $(cat /proc/sys/vm/swappiness)"

# 3. time sync
if [ "$ROLE" = node1 ]; then
    cat > /etc/chrony/conf.d/cluster.conf <<'EOF'
# Serve time to the cluster; keep serving the local clock when there is no internet.
allow 192.168.137.0/24
local stratum 10
EOF
else
    cat > /etc/chrony/conf.d/cluster.conf <<'EOF'
# Sync from node1 (works without internet); the Ubuntu pools in chrony.conf are still used when reachable.
server 192.168.137.30 iburst prefer
EOF
fi
systemctl enable chrony > /dev/null 2>&1
systemctl restart chrony
echo "[3] chrony ($ROLE): $(systemctl is-active chrony), config /etc/chrony/conf.d/cluster.conf"
echo "done"
