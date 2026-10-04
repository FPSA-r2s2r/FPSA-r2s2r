#!/bin/bash
# ============================================================
#  Step 1 (NUC Terminal 1):  Start polymetis + Franka hardware
#
#  This launches the real-time gRPC server on port 50051.
#  Must be running BEFORE franka_server.py.
#
#  Usage:
#    bash launch_polymetis.sh                    # default robot IP
#    bash launch_polymetis.sh 10.168.1.200       # custom robot IP
# ============================================================

ROBOT_IP="${1:-10.168.1.200}"

echo "=========================================="
echo "  Polymetis + Franka Hardware (Step 1)"
echo "  Robot IP: $ROBOT_IP"
echo "=========================================="
echo ""
echo "After this is running, open another terminal and run:"
echo "  python franka_server.py"
echo ""

launch_robot.py \
    robot_client=franka_hardware \
    robot_client.executable_cfg.robot_ip="$ROBOT_IP"
