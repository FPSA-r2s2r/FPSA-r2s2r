# Franka Standalone — 最简通信控制 & 数据采集

从 UMI 项目精简提取，**零依赖**于 UMI 框架本身，可独立运行。

## 文件结构

```
franka_standalone/
├── config.py                # 统一配置（IP、端口、增益、Home位姿等）
├── launch_polymetis.sh      # 下位机 Step 1 — 启动 polymetis + Franka 硬件
├── franka_server.py         # 下位机 Step 2 — ZeroRPC 暴露机械臂+夹爪
├── franka_control.py        # 上位机 — 纯键盘遥操作（无相机）
├── franka_collect.py        # 上位机 — 键盘遥操作 + 多相机数据采集
├── requirements_server.txt  # 下位机依赖
├── requirements_client.txt  # 上位机依赖
└── README.md
```

## 架构

```
┌──────────────────────┐          ZeroRPC           ┌──────────────────────────────┐
│  上位机 (Workstation) │ ◄═══════════════════════► │  下位机 (NUC)                 │
│                      │   tcp://<NUC_IP>:4242      │                              │
│  franka_control.py   │                            │  Terminal 1:                 │
│  franka_collect.py   │                            │    launch_polymetis.sh       │
│    ├─ 键盘遥操       │                            │    (polymetis gRPC :50051)   │
│    ├─ L515 相机 ×N   │                            │                              │
│    └─ Fisheye 相机   │                            │  Terminal 2:                 │
│                      │                            │    franka_server.py          │
│                      │                            │    (ZeroRPC :4242)           │
│                      │                            │    ├─ polymetis client       │
│                      │                            │    └─ libfranka gripper      │
└──────────────────────┘                            └──────────────────────────────┘
```

## 快速开始

### 1. 下位机 (NUC) — 需要两个终端

```bash
# 确保 polymetis 和 panda-python 已安装
pip install -r requirements_server.txt
```

**Terminal 1** — 启动 polymetis + Franka 硬件驱动（需要 sudo）:

```bash
bash launch_polymetis.sh 10.168.1.200
# 等价于: launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=10.168.1.200
```

**Terminal 2** — 等 Terminal 1 就绪后，启动 ZeroRPC server:

```bash
python franka_server.py --gripper_robot_ip 10.168.1.200
# 会自动等待 polymetis 就绪（最多重试 30 次，每次 2s）
# 加 --no_retry 则立即失败不等待
```

### 2. 上位机 — 纯控制（无相机）

```bash
pip install -r requirements_client.txt

# 键盘遥操作，终端显示位姿
python franka_control.py --robot_ip 192.168.3.2
```

### 3. 上位机 — 数据采集（有相机）

```bash
# 键盘遥操作 + L515 RGB-D + Fisheye 相机 + 录制
python franka_collect.py -o ./collected_data --robot_ip 192.168.3.2
```

## 键盘控制

| 键 | 功能 |
|----|------|
| W/S | 前进/后退 (X) |
| A/D | 左/右 (Y) |
| Q/E | 上/下 (Z) |
| J/L | Yaw 左/右 |
| I/K | Pitch 上/下 |
| U/O | Roll 左/右 |
| Shift | 按住 3 倍速 |
| Z | 闭合夹爪 |
| X | 打开夹爪 |
| H | 回到 Home 位 |
| C | 开始录制 (仅 collect) |
| V | 停止录制并保存 (仅 collect) |
| B | 丢弃当前 episode (仅 collect) |
| Esc | 退出 |

## 数据格式

每个 episode 保存在 `output_dir/episode_XXXX/`：

```
episode_0000/
├── robot_data.npz          # timestamps, actions(T,7), robot_states(T,7), joint_positions(T,7)
├── l515_0/
│   ├── color_00000.jpg     # RGB 图像 (960×540)
│   ├── color_00001.jpg
│   └── depth.npz           # 所有帧的深度图 (uint16, 640×480 aligned to color)
├── l515_1/
│   └── ...
└── fisheye/
    ├── color_00000.jpg
    └── ...
```

## 配置

修改 `config.py` 中的参数以匹配你的硬件：

- `ROBOT_IP` / `ROBOT_PORT` — NUC 网络地址
- `L515_SERIALS` — L515 相机序列号列表
- `FISHEYE_USB_ID` — 鱼眼相机 USB vendor:product ID
- `EE_HOME_POSE` — 末端执行器 Home 位姿
- `KX_DEFAULT` / `KXD_DEFAULT` — 阻抗控制增益
