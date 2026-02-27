"""
skills/system_info.py
Get Raspberry Pi 5 system information: CPU, RAM, disk, temperature, processes.
"""

DESCRIPTION = (
    "Get Pi 5 system information. "
    "Args: info_type (all|cpu|ram|disk|temp|processes|network), default 'all'"
)

import subprocess
import os
import re


def _run(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except Exception:
        return ""


def run(info_type: str = "all") -> str:
    sections = []

    if info_type in ("all", "cpu"):
        cpu = _run("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'")
        load = _run("uptime | awk -F'load average:' '{print $2}'")
        cores = _run("nproc")
        freq = _run("vcgencmd measure_clock arm 2>/dev/null | cut -d= -f2")
        if freq:
            freq_mhz = int(freq) // 1_000_000
            freq_str = f"{freq_mhz} MHz"
        else:
            freq_str = _run("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null")
            freq_str = f"{int(freq_str or 0)//1000} MHz"
        sections.append(
            f"CPU: {cores} cores @ {freq_str}\n"
            f"  Usage: {cpu}%  Load: {load}"
        )

    if info_type in ("all", "temp"):
        temp = _run("vcgencmd measure_temp 2>/dev/null | cut -d= -f2")
        if not temp:
            temp_raw = _run("cat /sys/class/thermal/thermal_zone0/temp")
            temp = f"{int(temp_raw or 0)/1000:.1f}'C" if temp_raw else "unavailable"
        sections.append(f"Temperature: {temp}")

    if info_type in ("all", "ram"):
        mem = _run("free -h | grep Mem")
        swap = _run("free -h | grep Swap")
        sections.append(f"RAM:\n  {mem}\nSwap:\n  {swap}")

    if info_type in ("all", "disk"):
        disk = _run("df -h | grep -E '/$|/mnt/nvme'")
        sections.append(f"Disk:\n{disk}")

    if info_type in ("all", "processes"):
        procs = _run("ps aux --sort=-%cpu | head -8")
        sections.append(f"Top Processes:\n{procs}")

    if info_type in ("all", "network"):
        net = _run("ip addr show | grep 'inet ' | grep -v 127.0.0.1")
        wifi = _run("iwconfig 2>/dev/null | grep ESSID")
        sections.append(f"Network:\n{net}\n{wifi}")

    if info_type == "ollama":
        models = _run("ollama list")
        ps = _run("ollama ps")
        sections.append(f"Ollama models:\n{models}\n\nRunning:\n{ps}")

    return "\n\n".join(sections) if sections else f"Unknown info_type: {info_type}"
