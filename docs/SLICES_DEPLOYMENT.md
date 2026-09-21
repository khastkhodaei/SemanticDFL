# EU SLICES Testbed Deployment Guide for SemanticDFL

## Overview
SemanticDFL includes a production-grade distributed transport layer powered by gRPC and Protocol Buffers, specifically designed for real-world deployment on the **EU SLICES (Scientific Large-Scale Infrastructure for Computing/Communication Experimental Studies)** testbed and edge computing environments.

---

## Architecture

```
                                +---------------------------+
                                |  Bootstrap / Super-Peer   |
                                |     (Node 0 / Port 50050) |
                                +-------------+-------------+
                                              |
                +-----------------------------+-----------------------------+
                |                             |                             |
  +-------------v-------------+ +-------------v-------------+ +-------------v-------------+
  |        Peer Node 1        | |        Peer Node 2        | |        Peer Node 3        |
  |    (Node 1 / Port 50051)  | |    (Node 2 / Port 50052)  | |    (Node 3 / Port 50053)  |
  +---------------------------+ +---------------------------+ +---------------------------+
```

### Key Capabilities:
- **Zero-Copy Tensor Streaming**: PyTorch parameter and gradient tensors are serialized into raw memory buffer payloads without intermediate JSON or string conversions.
- **Proximity Zone Alignment**: Nodes organize into bounded physical proximity zones with leader election and hierarchical super-peer trees.
- **Adaptive Top-P Signature Exchange**: Transmits coordinate-wise sparsified gradient signatures over TCP to drastically reduce WAN bandwidth consumption.
- **Failover & Churn Resiliency**: Automatic replica super-peer promotion if the primary zone exemplar experiences network disconnection.

---

## Configuration Profiles

SemanticDFL offers three standalone, decoupled configuration profiles located in `configs/`:

| Profile | File | Target Environment | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **Research Simulation** | [`configs/simulation.yaml`](file:///d:/Dev/SemanticDFL/configs/simulation.yaml) | Centralized Research & Benchmarking | In-Memory broker, 100 clients, 50 rounds, full academic plotting & JSON metrics |
| **Distributed Local** | [`configs/distributed_local.yaml`](file:///d:/Dev/SemanticDFL/configs/distributed_local.yaml) | Local Machine Socket Verification | Multi-process gRPC sockets on `127.0.0.1:50050+`, 3 clients, 2 rounds for rapid testbed smoke testing |
| **SLICES Production** | [`configs/slices_production.yaml`](file:///d:/Dev/SemanticDFL/configs/slices_production.yaml) | EU SLICES Large-Scale Testbed | 100 distributed edge nodes across EU sites, multi-super-peer hierarchy, gRPC over WAN |

### Quick-Select CLI Execution Commands

Run in-memory algorithm simulation:
```bash
python run_simulation.py --config configs/simulation.yaml
```

Run local multi-process gRPC test:
```bash
python run_simulation.py --config configs/distributed_local.yaml
```

Run SLICES multi-node production deployment orchestration:
```bash
python run_simulation.py --config configs/slices_production.yaml
```

---

## Deployment Modes

### 1. Docker Compose (Local & Single-Host SLICES Node)
Spin up the complete 5-node testbed network:
```bash
docker compose -f docker-compose.slices.yml up --build
```

### 2. Multi-Host Bare-Metal / Kubernetes SLICES Deployment

#### Start Bootstrap Super-Peer on Edge Gateway:
```bash
python run_node.py \
  --node_id 0 \
  --host 0.0.0.0 \
  --port 50050 \
  --is_super_peer \
  --config configs/slices_production.yaml
```

#### Start Distributed Worker Nodes across SLICES Sites:
```bash
# Node 1 (Paris Site)
python run_node.py --node_id 1 --host 0.0.0.0 --port 50051 --bootstrap 192.168.1.10:50050 --config configs/slices_production.yaml

# Node 2 (Madrid Site)
python run_node.py --node_id 2 --host 0.0.0.0 --port 50052 --bootstrap 192.168.1.10:50050 --config configs/slices_production.yaml

# Node 3 (Athens Site)
python run_node.py --node_id 3 --host 0.0.0.0 --port 50053 --bootstrap 192.168.1.10:50050 --config configs/slices_production.yaml
```

---

## Transport Verification
Run the integration test suite:
```bash
pytest tests/test_transport_grpc.py -v
```
