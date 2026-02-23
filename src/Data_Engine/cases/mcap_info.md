Here is a concise Markdown summary to help you identify the contents of an `.mcap` file.

---

# 🛠️ Inspecting `.mcap` Data Content

If you have downloaded an `.mcap` file and need to verify whether it contains **Joint**, **Cartesian**, or **Image** data without writing code, use the following methods.

## 1. Fast Inspection (Command Line)

The most direct way to check recorded topics and message types is using the ROS 2 CLI:

```bash
ros2 bag info your_data.mcap

```

### How to Interpret the Output:

Look at the **Topic** and **Type** columns to determine the recorded data:

| If you see Topic/Type... | Content Identified |
| --- | --- |
| `/joint_states` | **Joint Space Info** (`sensor_msgs/msg/JointState`) |
| `/tf` or `/tf_static` | **Coordinate Transforms** (Can derive Cartesian) |
| `/end_effector_pose` | **Cartesian Space Info** (`geometry_msgs/msg/PoseStamped`) |
| `/camera/...` | **Visual Data** (`sensor_msgs/msg/Image` or `CompressedImage`) |

---

## 2. Verify Data Values (Live Echo)

To see the actual numerical values (e.g., to check if Cartesian coordinates are populated):

1. **Start Playback**:
```bash
ros2 bag play your_data.mcap

```


2. **Echo Topic (In a second terminal)**:
```bash
# Check Joint values
ros2 topic echo /joint_states --once

# Check Cartesian values
ros2 topic echo /target_pose --once

```



---

## 3. Visual Confirmation (GUI)

For the most intuitive view, use **Foxglove Studio** (the native viewer for `.mcap`):

1. Open [Foxglove Studio](https://studio.foxglove.dev/).
2. **Drag and drop** your `.mcap` file into the browser.
3. Check the **Schema/Topic list** on the left sidebar.
4. **Pro-tip**: Add a "3D Panel" to visualize Cartesian TFs or a "Plot Panel" to see Joint curves over time.

---

## 🔍 Summary Logic

* **Cartesian Only**: High frequency of `geometry_msgs/Pose` topics; no `JointState` topics.
* **Joint Only**: `JointState` topics present; no specific end-effector pose topics.
* **Full Dataset**: Both topic types appear in the `ros2 bag info` summary.
