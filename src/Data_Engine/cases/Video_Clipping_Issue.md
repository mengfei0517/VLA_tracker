# 📝 Post-Mortem: Video Clipping Issue in LeRobot Data Visualization

## 🔍 Problem Description

During the visualization process of **LeRobot** datasets, it was observed that approximately the **last 2 seconds** of video content were consistently clipped/missing, despite the recording appearing to finish normally.

* **Observation**: The issue reproduced across all datasets collected today, while historical datasets remained unaffected.
* **Initial Check**: Verified `rosbag` data and the `rosbags` conversion documentation.
* **Hypothesis**: Data alignment logic during the conversion (aligning Joint states with Camera frames) was dropping frames due to padding or timestamp gaps.

---

## 🛠️ Root Cause Analysis: Clock Desynchronization

The root cause was identified as a **Clock Skew** between two machines in a distributed setup:

* **Machine A (Controller)**: Running the launch file, sending control signals, and publishing ROS 2 nodes (including Joint states).
* **Machine B (Recorder)**: Recording the `rosbag` containing both Joint states and Camera streams.

**The Conflict:** Machine A's system clock was approximately **2 seconds slower** than Machine B's. When Machine B recorded the data:

1. The **Camera frames** were timestamped by Machine B's "current" time.
2. The **Joint states** arrived with Machine A's "past" timestamps.
3. During the **LeRobot alignment process**, the software attempted to match images to joints. Since the joint data "ended" 2 seconds earlier (according to the timestamps), the remaining 2 seconds of video had no matching joint data and were consequently **discarded**.

---

## 💻 Diagnostic Toolkit

To diagnose and verify the synchronization, the following commands were used:

### 1. Check Unix Epoch Time

Used to compare the raw timestamp (seconds/microseconds) between the two machines.

```bash
# Display the precise system time
echo $EPOCHREALTIME

# Alternative using date command
date +%s.%N

```

### 2. Check System Time Status

Used to verify if NTP (Network Time Protocol) is active and synchronized.

```bash
# Check detailed synchronization info
timedatectl status

# Key indicators to look for:
# System clock synchronized: yes
# NTP service: active

```

### 3. Check Network Connectivity

Used to ensure the physical link between machines is stable.

```bash
# Ping the other machine to check latency
ping <TARGET_IP_ADDRESS>

```

---

## 🚀 Solutions & Prevention

### Immediate Fix

* **Unified Network**: Connected both machines to the same local network with internet access.
* **Auto-Sync**: Enabled automatic time synchronization on both systems to ensure they fetch time from the same NTP server.

### Best Practices for Distributed ROS 2

To prevent future drift in a multi-machine setup:

1. **Install Chrony**: A more robust alternative to standard NTP for robotics.
```bash
sudo apt install chrony
sudo chronyc -a makestep

```


2. **Pre-flight Check**: Always run a timestamp comparison script before starting long data collection sessions.

---

**Would you like me to write a short Bash or Python script that automatically checks the time offset between your two machines before you start recording?**