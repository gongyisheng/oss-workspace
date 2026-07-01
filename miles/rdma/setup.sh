#!/usr/bin/env bash
# One-time setup: install RDMA userspace libs and bring up Soft-RoCE so the
# demo can run on a machine with no InfiniBand/RoCE hardware.
set -euo pipefail

# 1. Userspace verbs + connection-manager libraries and CLI tools.
sudo apt-get update
sudo apt-get install -y rdma-core libibverbs-dev librdmacm-dev ibverbs-utils

# 2. Load the software RDMA transport (RDMA over an ordinary Ethernet NIC).
sudo modprobe rdma_rxe

# 3. Bind a virtual rxe device to your primary network interface. Loopback
#    (lo) does not carry RoCE well, so use the default-route interface.
IFACE=$(ip -o -4 route show default | awk '{print $5}' | head -1)
IFACE=${IFACE:-eth0}
echo "binding rxe0 to netdev: $IFACE"
sudo rdma link add rxe0 type rxe netdev "$IFACE" 2>/dev/null || true

# 4. Confirm the device shows up (state should be ACTIVE).
rdma link show
ibv_devices
