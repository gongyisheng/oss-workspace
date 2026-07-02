#!/usr/bin/env bash
# Build the demo and run server + client in loopback on this host.
set -euo pipefail

mkdir -p build
cc -Wall -O2 -o build/rdma_demo rdma_demo.c -lrdmacm -libverbs

# Client connects to this host's real IP (not 127.0.0.1 — that bypasses rxe).
IP=$(ip -o -4 addr show scope global | awk '{print $4}' | cut -d/ -f1 | head -1)

./build/rdma_demo &          # server
SRV=$!
sleep 1
./build/rdma_demo "$IP"      # client
wait "$SRV"
