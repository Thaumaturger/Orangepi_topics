#!/bin/bash
# One-time board setup. Run on each board with sudo:  sudo bash ~/board_setup.sh node1   (or node2)
#  1. performance governor at every boot (systemd oneshot; replaces running ~/set_perf.sh after each boot)
#  2. vm.swappiness 100 -> 10 (node1 has a 4 GB swapfile; with 100 the kernel swaps out llama.cpp's weights
#     instead of dropping page cache). Persistent in /etc/sysctl.d/99-sysctl.conf, applied now as well.
#  3. cluster time: node1 serves NTP to 192.168.137.0/24 (also without internet); node2 syncs from node1 and
#     steps its clock whenever it is more than 1 s off (makestep 1 -1 in /etc/chrony/chrony.conf).
# Safe to run again. Undo: systemctl disable --now cpu-performance; rm /etc/systemd/system/cpu-performance.service;
#   set vm.swappiness=100 back in /etc/sysctl.d/99-sysctl.conf; rm /etc/chrony/conf.d/cluster.conf;
#   (node2) set "makestep 1 3" back in /etc/chrony/chrony.conf; systemctl restart chrony
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
    # Step the clock whenever it is more than 1 s off, not only in the first 3 updates, so node2 follows node1
    # after node1's clock is corrected by hand. chrony.conf is edited because its makestep line comes after
    # confdir and would override a makestep in conf.d.
    sed -i 's/^makestep .*/makestep 1 -1/' /etc/chrony/chrony.conf
    grep -q '^makestep 1 -1' /etc/chrony/chrony.conf || echo 'makestep 1 -1' >> /etc/chrony/chrony.conf
fi
systemctl enable chrony > /dev/null 2>&1
systemctl restart chrony
echo "[3] chrony ($ROLE): $(systemctl is-active chrony), config /etc/chrony/conf.d/cluster.conf"
echo "done"
