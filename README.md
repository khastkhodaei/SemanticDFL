# SemanticDFL: Similarity-Aware Pull-based Personalized Decentralized Federated Learning

[![Publication: ACM POMACS 2026](https://img.shields.io/badge/Publication-ACM%20POMACS%202026-b31b1b.svg)](#paper-publication--citation)
[![Conference: ACM SIGMETRICS 2026](https://img.shields.io/badge/Conference-ACM%20SIGMETRICS%202026-1f425f.svg)](#paper-publication--citation)
[![Test Suite](https://img.shields.io/badge/pytest-97%20passed%20(100%25)-brightgreen.svg)](#test-suite--verification)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**SemanticDFL** is a high-performance framework for Personalized Decentralized Federated Learning (P-DFL) across heterogeneous peer-to-peer (P2P) edge networks. Rather than relying on rigid static overlays or bandwidth-prohibitive full-model gossip, SemanticDFL dynamically self-organizes peers into a 2-Tier hierarchical **Semantic Overlay Network (SON)** using sparse parameter importance signatures ($\text{Top-}P$).

Peers autonomously discover semantically aligned collaborators with logarithmic search complexity ($O(\log N)$), prune wire bandwidth by **$80.0\% - 87.3\%$**, accelerate convergence by **$2.5\times$**, and retain high personalization accuracy even under severe network churn ($40\%$).

---

## Paper Publication & Citation

This repository contains the reference implementation of the research paper published in **ACM POMACS / ACM SIGMETRICS 2026**:

> **Javad Dogani, Mostafa Khastkhodaei, Farshad Khunjush, Nikolaos Laoutaris.**  
> *"SemanticDFL: Similarity-Aware Pull-based Personalized Decentralized Federated Learning."*  
> **Proceedings of the ACM on Measurement and Analysis of Computing Systems (POMACS)**, Vol. 10, No. 2, Article 51, June 2026.  
> Presented at **ACM SIGMETRICS 2026**.

If you use SemanticDFL in your research or academic benchmarks, please cite the paper:

```bibtex
@article{dogani2026semanticdfl,
  title     = {SemanticDFL: Similarity-Aware Pull-based Personalized Decentralized Federated Learning},
  author    = {Dogani, Javad and Khastkhodaei, Mostafa and Khunjush, Farshad and Laoutaris, Nikolaos},
  journal   = {Proceedings of the ACM on Measurement and Analysis of Computing Systems (POMACS)},
  volume    = {10},
  number    = {2},
  articleno = {51},
  pages     = {1--35},
  year      = {2026},
  publisher = {ACM New York, NY, USA},
  doi       = {10.1145/3711718}
}
```

---

## Key Empirical Results from the Paper

### 1. Test Accuracy Across 16 Non-IID Scenarios (Paper Table 2)
Test accuracy (mean over 3 seeds) evaluated across four diverse modalities under both Dirichlet label skew ($\alpha = 0.3, 0.1$) and Pathological class-quota partitioning ($30\%, 20\%$). **SemanticDFL consistently outperforms all 7 state-of-the-art baselines in every scenario**:

| Dataset | Split Type | FedAvg | FedProx | Per-FedAvg | Ditto | pFedGraph | DPFL | DFedPGP | **SemanticDFL** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fashion-MNIST** | $\text{Dir}(0.3)$ | 66.4% | 69.9% | 73.4% | 76.1% | 75.4% | 76.8% | 77.9% | **79.6%** |
| *(2-Layer CNN)* | $\text{Dir}(0.1)$ | 62.0% | 66.0% | 69.7% | 70.8% | 70.4% | 71.0% | 71.3% | **73.1%** |
| | $\text{Path}(30\%)$ | 57.0% | 61.5% | 65.0% | 67.2% | 66.7% | 68.4% | 69.8% | **71.5%** |
| | $\text{Path}(20\%)$ | 53.5% | 58.0% | 61.5% | 63.8% | 63.2% | 65.7% | 67.4% | **68.4%** |
| **Tiny-ImageNet** | $\text{Dir}(0.3)$ | 38.3% | 42.3% | 46.3% | 47.6% | 48.2% | 47.0% | 48.6% | **48.8%** |
| *(ResNet-18)* | $\text{Dir}(0.1)$ | 36.0% | 40.0% | 44.0% | 46.1% | 46.8% | 45.2% | 47.1% | **47.3%** |
| | $\text{Path}(30\%)$ | 36.7% | 41.0% | 44.7% | 46.6% | 47.2% | 45.8% | 47.8% | **48.0%** |
| | $\text{Path}(20\%)$ | 36.2% | 40.5% | 44.2% | 45.3% | 45.6% | 44.9% | 46.5% | **46.7%** |
| **Google Speech** | $\text{Dir}(0.3)$ | 78.4% | 81.4% | 85.2% | 86.8% | 87.0% | 87.4% | 88.2% | **89.1%** |
| *(ResNet-18)* | $\text{Dir}(0.1)$ | 76.6% | 79.6% | 83.2% | 84.8% | 85.2% | 86.2% | 86.6% | **86.8%** |
| | $\text{Path}(30\%)$ | 77.0% | 80.2% | 84.0% | 85.8% | 86.1% | 86.8% | 87.4% | **88.2%** |
| | $\text{Path}(20\%)$ | 76.2% | 79.4% | 83.1% | 85.0% | 85.5% | 86.1% | 86.9% | **87.5%** |
| **20Newsgroups** | $\text{Dir}(0.3)$ | 50.8% | 53.6% | 57.8% | 60.1% | 60.5% | 61.2% | 61.8% | **64.2%** |
| *(ALBERT-Base)* | $\text{Dir}(0.1)$ | 49.4% | 52.0% | 55.4% | 58.0% | 58.6% | 59.1% | 58.2% | **63.3%** |
| | $\text{Path}(30\%)$ | 48.0% | 51.0% | 54.2% | 56.8% | 57.3% | 58.0% | 57.1% | **62.0%** |
| | $\text{Path}(20\%)$ | 46.5% | 49.2% | 52.6% | 55.0% | 55.4% | 56.2% | 55.0% | **60.1%** |

### 2. Wire Sizing & Communication Compression (Paper §3.4, §5.1)
Each Top-$P$ signature coordinate consumes **6 Bytes** ($4\text{ B}$ integer index + $2\text{ B}$ float16 value) vs **4 Bytes** per dense FP32 parameter. The analytical wire reduction factor is $\frac{4M}{6P} = \frac{2}{3 \cdot (P/M)}$:

| Benchmark Model | Architecture | Parameters ($M$) | Tuned $P/M$ | Signature Size ($6P$) | Dense FP32 | Wire Reduction | Bandwidth Saved |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fashion-MNIST** | 2-Layer CNN | 62,000 | 12.3% | 45.7 KB | 248.0 KB | **5.42×** | **81.5%** |
| **Tiny-ImageNet** | ResNet-18 | 11,700,000 | 9.6% | 6.74 MB (6.43 MiB) | 46.8 MB (44.6 MiB) | **6.94×** | **85.6%** |
| **Google Speech** | ResNet-18 | 11,700,000 | 11.2% | 7.86 MB (7.50 MiB) | 46.8 MB (44.6 MiB) | **5.95×** | **83.2%** |
| **20Newsgroups** | ALBERT-Base | 11,683,000 | 12.8% | 8.97 MB (8.55 MiB) | 46.7 MB (44.6 MiB) | **5.21×** | **80.8%** |

### 3. Convergence Speedup (Paper Table 3)
Rounds required to reach target test accuracy under Dirichlet $\alpha = 0.3$:
- **Fashion-MNIST** ($75\%$ target): SemanticDFL reaches target in **28 rounds** ($2.46\times$ faster than FedAvg @ 69 rounds; $1.29\times$ faster than DPFL @ 36 rounds).
- **Google Speech** ($85\%$ target): SemanticDFL reaches target in **31 rounds** ($2.55\times$ faster than FedAvg @ 79 rounds; $1.26\times$ faster than DPFL @ 39 rounds).

### 4. Robustness Under Severe Dynamic Churn (Paper §5.5, Table 7)
Accuracy degradation under increasing peer dropout rates ($0\% \to 40\%$):
- At **$40\%$ churn**, SemanticDFL retains **$67.1\%$** on FMNIST, **$40.3\%$** on Tiny-ImageNet, **$80.8\%$** on Speech, and **$56.3\%$** on 20Newsgroup.
- SemanticDFL outperforms FedAvg by **$+16.0$ to $+20.9\text{ pp}$** under severe churn due to $r=3$ replica super-peers, lease failover, and local pointer patching.

---

## System Architecture & Protocol Lifecycle

SemanticDFL executes a structured 10-stage decentralized protocol that balances local computation and network communication:

```
                  +-------------------------------------------------------------+
                  |  Level H: Root Super-Peer (Compact Overlay Tree Descriptor) |
                  +-------------------------------------------------------------+
                                     /                               \
                                    /                                 \
            +------------------------------+             +------------------------------+
            | Level 2 Super-Peer (AP Repr) |             | Level 2 Super-Peer (AP Repr) |
            +------------------------------+             +------------------------------+
                   /                \                           /                \
      +-----------------+   +-----------------+   +-----------------+   +-----------------+
      | Level 1 Zone A  |   | Level 1 Zone B  |   | Level 1 Zone C  |   | Level 1 Zone D  |
      | (VRF Initiator) |   | (VRF Initiator) |   | (VRF Initiator) |   | (VRF Initiator) |
      +-----------------+   +-----------------+   +-----------------+   +-----------------+
           /         \           /         \           /         \           /         \
         Peer1     Peer2       Peer3     Peer4       Peer5     Peer6       Peer7     Peer8
```

### Protocol Stages:
1. **S1: Bootstrap & VRF Initiator Election** (§3.3.1) — Verifiable Random Function score $\sigma_p = \text{VRF}_p(\text{seed})$. Nodes exchange $(\sigma, \text{id})$ within radius $R$.
2. **S2: Proximity Zone Formation** (§3.3.1) — Initiators flood PROBE with $\text{TTL} = R$. First-wins rule bounds zone size to cap $S_Z = 0.05N$.
3. **S3: Top-$P$ Selection & Signature Extraction** (§3.2) — Local binary search finds minimal $P_{\min}$ satisfying $\cos(w, w_{[P]}) \ge \tau_{\min} (0.80)$. Zone aggregates $q$-quantile to set global $P$.
4. **S4: Intra-Zone AP Clustering** (§3.3.2) — Affinity Propagation creates exemplar clusters without requiring a fixed $k$. Replicates cluster descriptors to $r=3$ super-peers.
5. **S5: Inter-Zone Recursive SON Build** (§3.3.3) — Recursive AP builds overlay tree of depth $H = \lceil \log_{S_Z} N \rceil = O(\log N)$. Total network control cost is $O(N)$.
6. **S6: Overlay Maintenance & Leases** (§3.7) — Short leases with deterministic successors $(\text{VRF}, \text{id})$ prevent orphan subtrees upon peer churn.
7. **S7: Similarity Search Per Round** (§3.4, §3.5) — Top-down beam search ($O(PFBH)$ ops, $O(BH)$ messages) with online adaptive threshold $\tau^{(t)}$ controller.
8. **S8: Model Pull & Aggregation** (§2.1, §3.4) — Peers pull $K = 0.10N$ full models from selected partners.
9. **S9: Personalization Update** (§5.1) — Softmax similarity weighting with dynamic regularization $\alpha_i = \frac{\eta \psi_i}{1 + \eta \psi_i}$ ($\psi = 2$).
10. **S10: Asynchronous Freshness Guard** (§3.6) — Filters delayed responses from round $t' \in [t - \Delta_{\max}, t)$.

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/khastkhodaei/SemanticDFL.git
cd SemanticDFL
```

### 2. Set Up Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Run: Execution Modes

SemanticDFL supports four distinct execution modes tailored for research simulations, local debugging, distributed networking, and containerized deployments:

### Mode 1: High-Speed In-Memory Simulation (Recommended for Benchmarks)
Executes multi-client decentralized federated learning with precise communication accounting and zero socket overhead:

```bash
# Run default simulation (10 clients, Fashion-MNIST, Dirichlet alpha=0.3, 20 rounds)
python run_simulation.py --config configs/simulation.yaml --rounds 20

# Run with customized client count and dataset
python run_simulation.py --num_clients 20 --rounds 15 --dataset fmnist --partitioner dirichlet --alpha 0.1

# Run on GPU with custom seed
python run_simulation.py --device cuda --seed 42 --rounds 25
```

### Mode 2: Distributed gRPC Network Execution
Orchestrates autonomous peer processes communicating over real TCP/IP sockets via gRPC:

```bash
# Launch a local distributed multi-process cluster
python run_simulation.py --config configs/distributed_local.yaml --mode distributed
```

### Mode 3: Standalone Autonomous Peer Node
Spins up an individual peer process that connects to or bootstraps a P2P network:

```bash
# Terminal 1: Start bootstrap super-peer (Node 0) on port 50050
python run_node.py --node_id 0 --host 0.0.0.0 --port 50050 --is_super_peer --rounds 10

# Terminal 2: Start worker peer (Node 1) connecting to Node 0
python run_node.py --node_id 1 --host 0.0.0.0 --port 50051 --bootstrap 127.0.0.1:50050 --rounds 10

# Terminal 3: Start worker peer (Node 2) connecting to Node 0
python run_node.py --node_id 2 --host 0.0.0.0 --port 50052 --bootstrap 127.0.0.1:50050 --rounds 10
```

### Mode 4: Docker & SLICES WAN Testbed Deployment
Run a 10-node containerized decentralized topology modeling cross-region European testbeds (Paris, Madrid, Bologna):

```bash
# Build and orchestrate the 10-node SLICES cluster
docker-compose -f docker-compose.slices.yml up --build

# Follow logs of a specific peer node
docker-compose -f docker-compose.slices.yml logs -f node-0
```

---

## Interactive Companion Dashboard (`SemanticVisual`)

For interactive visual inspections, real-time Top-$P$ coordinate slicing, and animated protocol step walkthroughs, run the companion visualization dashboard:

```bash
# Clone the visualization repository
git clone https://github.com/khastkhodaei/SemanticVisual.git
cd SemanticVisual

# Launch dashboard (starts FastAPI server on http://127.0.0.1:8000 and opens browser)
# Windows:
start_server.bat
# Linux / macOS:
python run.py
```

Dashboard features:
- **Tab 1: Top-P Benchmark Inspector**: Direct parameter introspection across all 5 benchmark models.
- **Tab 2: Step-by-Step Worked Example**: Complete deterministic numerical walkthrough of Section 3.2.
- **Tab 3: Paper Models & Savings**: Interactive Analytical Cost Calculator (§3.4 & §5.1), Round 100 Utopia Pareto Frontier projection, and Dynamic Churn Evaluator (Table 7).
- **Tab 4: Q&A**: Searchable technical documentation and theoretical derivations.

---

## Configuration Reference (`configs/default_config.yaml`)

```yaml
simulation:
  rounds: 20
  seed: 42
  device: "auto"              # "auto", "cpu", or "cuda"

federated:
  num_clients: 10
  k_neighbors: 3              # K = 0.10 * N
  zone_size: 5                # Zone cap S_Z = 0.05 * N
  alpha_personalization: 0.67 # Mixing coupling parameter
  p_cardinality: 100          # Top-P signature dimension
  son_refresh_interval: 1     # Periodic AP clustering refresh epoch

network:
  topology: "small_world"     # Options: ring, small_world, geometric, grid, slices_wan
  degree: 4

data:
  dataset: "fmnist"           # Options: fmnist, cifar10, tiny_imagenet, google_speech, 20newsgroups
  partitioner: "dirichlet"    # Options: dirichlet, pathological, iid
  alpha: 0.3                  # Dirichlet concentration parameter
  batch_size: 32

model:
  architecture: "FashionCNN"
  learning_rate: 0.01
  local_epochs: 1

logging:
  output_dir: "results"
  generate_plots: true        # Automatically generates publication figures
  save_results_json: true
```

---

## Test Suite & Verification

The framework includes 97 unit and integration tests covering the core mathematical algorithms, overlay formation, search pruning, gRPC networking, and plotting:

```bash
# Run pytest test suite
pytest -v
```

Expected output:
```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Dev\SemanticDFL
collected 97 items

tests/test_core_and_data.py ...........................                  [ 27%]
tests/test_execution_mode.py ............                                [ 40%]
tests/test_overlay_and_clustering.py ........                            [ 48%]
tests/test_search_and_aggregation.py ....................                [ 69%]
tests/test_signature_and_metrics.py ..........                           [ 79%]
tests/test_topology_and_roles.py ............                            [ 91%]
tests/test_transport_grpc.py ......                                      [ 97%]
tests/test_visualization.py ..                                           [100%]

============================= 97 passed in 12.23s =============================
```

---

## Repository Directory Structure

```
SemanticDFL/
├── configs/                     # YAML configuration profiles
│   ├── default_config.yaml      # Baseline configuration
│   ├── simulation.yaml          # Fast in-memory simulation profile
│   ├── distributed_local.yaml   # Distributed gRPC multi-process profile
│   └── slices_production.yaml   # 10-node SLICES WAN testbed profile
├── semantic_dfl/                # Core Python package
│   ├── aggregation/             # Softmax pull aggregator & Freshness Guard
│   ├── core/                    # Neural architectures & parameter flattening
│   ├── data/                    # Dataset loaders & non-IID partitioners
│   ├── federated/               # Client trainer & simulation coordinator
│   ├── overlay/                 # SON tree, VRF election, & network topologies
│   ├── search/                  # Beam search & online threshold controller
│   ├── signature/               # Top-P sparse signatures & cosine metrics
│   ├── transport/               # In-memory broker & gRPC network transport
│   └── visualization/           # Academic figure replication engine
├── tests/                       # 97 pytest unit & integration tests
├── results/                     # Output metrics, logs, and generated figures
├── run_simulation.py            # Primary simulation entrypoint
├── run_node.py                  # Standalone distributed peer process entrypoint
├── docker-compose.slices.yml    # Docker cluster configuration
├── requirements.txt             # Pip dependency manifest
├── LICENSE                      # MIT License
└── README.md                    # Project documentation
```

---

## Author & Contact

**Mostafa Khastkhodaei**  
- **Email**: [mostafakhastkhodaei@gmail.com](mailto:mostafakhastkhodaei@gmail.com)  
- **LinkedIn**: [linkedin.com/in/mostafakhastkhodaei](https://www.linkedin.com/in/mostafakhastkhodaei/)  
- **GitHub**: [@khastkhodaei](https://github.com/khastkhodaei)

Feel free to open an issue or reach out directly for research collaborations, questions regarding the ACM POMACS 2026 paper, or extending SemanticDFL to new architectures.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.
