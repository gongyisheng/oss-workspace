#!/usr/bin/env bash
# Build the demo and run server + client in loopback on this host.
set -euo pipefail

cc -Wall -O2 -o rdma_demo rdma_demo.c -lrdmacm -libverbs

# Client connects to this host's real IP (not 127.0.0.1 — that bypasses rxe).
IP=$(ip -o -4 addr show scope global | awk '{print $4}' | cut -d/ -f1 | head -1)
IP=${IP:-127.0.0.1}

./rdma_demo &          # server
SRV=$!
sleep 1
./rdma_demo "$IP"      # client
wait "$SRV"
