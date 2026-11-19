#!/bin/bash
# Sets every CPU frequency policy to the performance governor (needs sudo). Resets on reboot.
echo performance | sudo tee /sys/devices/system/cpu/cpufreq/policy*/scaling_governor > /dev/null
cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | sort | uniq -c
