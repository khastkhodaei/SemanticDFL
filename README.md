# SemanticDFL: Scalable and Efficient Semantic Overlay Network for Decentralized Federated Learning

[![CI / Test Suite](https://img.shields.io/badge/pytest-97%20passed%20(100%25)-brightgreen.svg)](#test-suite--verification)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Publication](https://img.shields.io/badge/ACM%20POMACS-2026-red.svg)](#citation)

**SemanticDFL** is a decentralized federated learning (DFL) framework that addresses statistical data heterogeneity (Non-IID label skew) across peer-to-peer (P2P) edge networks. By self-organizing peers into a hierarchical **Semantic Overlay Network (SON)** based on sparse model parameter signatures, SemanticDFL achieves near-optimal model personalization and peer discovery with logarithmic control-plane message overhead and bounded communication complexity.

---

## Academic Citation

If you use this repository or framework in your research, please cite:

```bibtex
@article{dogani2026semanticdfl,
  title     = {SemanticDFL: Scalable and Efficient Semantic Overlay Network for Decentralized Federated Learning},
  author    = {Dogani, Jafar and Khastkhodaei, Mostafa and Khunjush, Farshad and Laoutaris, Nikolaos},
  journal   = {Proceedings of the ACM on Measurement and Analysis of Computing Systems (POMACS) / ACM SIGMETRICS},
  year      = {2026},
  publisher = {ACM New York, NY, USA}
}
```

---

## Key Highlights & System Architecture

```
                                 [Root Super-Peer] (Level H)
                                      /              \
                                     /                \
                       [Super-Peer A] (Level 2)      [Super-Peer B] (Level 2)
                            /         \                    /         \
                 [Zone 1 (L1)]     [Zone 2 (L1)]     [Zone 3 (L1)]   [Zone 4 (L1)]
                    /   \             /   \             /   \           /   \
                 (p1)   (p2)       (p3)   (p4)       (p5)   (p6)     (p7)   (p8)
                  ^
                  |-- VRF Leader Election -> Zone Initiator (Algorithm 2)
                  |-- Sparse Cosine Signature u_i in R^P (Top-P salience)
                  |-- Hierarchical Bounded-Fanout Search (Recall@K >= 0.85)
                  |-- Row-Stochastic Softmax Pull Aggregator (Temperature-scaled)
                  |-- Dynamic Personalization Coupling alpha_i = (eta*psi_i) / (1 + eta*psi_i)
                  |-- Asynchronous Freshness Guard T_fresh
```

### 1. Physical Underlay Network Modeling ($G = (V, E)$)
Realistic edge and testbed communication constraints in `semantic_dfl/overlay/topology.py`:
- **Regular Ring Lattice** (`generate_ring_topology`): $k$-regular ring with configurable degree.
- **Watts-Strogatz Small-World** (`generate_small_world_topology`): High clustering and short characteristic path lengths with connectivity assurance.
- **Random Geometric Graph (RGG)** (`generate_random_geometric_topology`): Models Euclidean spatial proximity of mobile/IoT edge devices.
- **2D Torus / Grid** (`generate_grid_topology`): 4-connected spatial grid mesh.
- **EU SLICES Testbed Multi-Region WAN** (`generate_slices_wan_topology`): High-density intra-site links (Paris, Madrid, Bologna) with bounded inter-site cross-border gateway links.

### 2. Role Lifecycle & VRF-Based Leader Election
- **Roles**: Explicit lifecycle state tracking (`ZONE_INITIATOR`, `SUPER_PEER`, `REPLICA_SUPER_PEER`, `LEAF_PEER`).
- **VRF Election** (Algorithm 2): Verifiable Random Function score $V_i = \text{Hash}(i \parallel \text{epoch\_seed}) \bmod 10^6$. Peers with the top local VRF scores within radius $R$ self-nominate as Zone Initiators.

### 3. Sparse Model Signatures & Top-$P$ Selection
- **Dimension Reduction**: Extracts Top-$P$ salient parameter coordinates:
  $$u_i^{(t)} = \text{Top-P}(w_i^{(t)} - w_{\text{init}})$$
- **Cosine Fidelity Bound**: Guarantees $\cos(u_i, u_j) \approx \cos(w_i, w_j)$ while reducing control transmission to only $6 \times P$ bytes ($4$-byte coordinate index + $2$-byte float16 weight).

### 4. Online Adaptive Threshold Controller
- **Dynamic Online Tuning** (Section 3.5):
  $$\tau^{(t+1)} \leftarrow \text{clip}_{[-1, 1]} \left( \tau^{(t)} + \eta_\tau (\hat{a}(t) - \phi) - \gamma(1 - \rho_i^{(t)}) \right)$$
- Balances peer selectivity ($\phi = 0.5$) against local drift ($\rho_i^{(t)} = \cos(u_i^{(t)}, u_i^{(t-1)})$).
- Stepwise threshold relaxation ($\delta = 0.02$) prevents search starvation when discovered peers $< K$.

### 5. Hierarchical Bounded-Fanout Search Engine
- Top-down beam search descending from tree root level $H \to 1$.
- Prunes divergent branches via prototype comparison while tracking control-plane bytes and Recall@K relative to ideal centralized oracle.

### 6. Personalized Pull Aggregator & Freshness Guard
- **Softmax Pull Weighting**:
  $$\bar{w}_i^{(t)} = \sum_{j \in \mathcal{C}_i^{(t)}} \beta_{ij}^{(t)} w_j^{(t)}, \quad \beta_{ij}^{(t)} = \frac{\exp(s_{ij}^{(t)} / \tau_{\text{soft}})}{\sum_{k} \exp(s_{ik}^{(t)} / \tau_{\text{soft}})}$$
- **Dynamic Personalization Mixing**:
  $$w_i^{(t+1)} = (1 - \alpha_i) \tilde{w}_i^{(t)} + \alpha_i \bar{w}_i^{(t)}, \quad \alpha_i = \frac{\eta \psi_i}{1 + \eta \psi_i}$$
- **Freshness Guard**: Filters delayed model replies from round $t' \in [t - \Delta_{\max}, t)$ enforcing $\text{sim}(u_i^{(t)}, u_j^{(t')}) \ge T_{\text{fresh}}$, safely discarding stale models $> \Delta_{\max}$.

---

## Multi-Modal Benchmarks & Heterogeneous Partitions

SemanticDFL includes full multi-modal model architectures and non-IID partitioners:

| Modality | Benchmark Dataset | Architecture | Classes | Input Format |
| :--- | :--- | :--- | :--- | :--- |
| **Vision (Fashion)** | Fashion-MNIST | `FashionCNN` | 10 | $1 \times 28 \times 28$ |
| **Vision (Complex)** | Tiny-ImageNet / CIFAR | `ResNet18Small` | 200 / 10 | $3 \times 64 \times 64$ / $3 \times 32 \times 32$ |
| **Audio (Speech)** | Google Speech Commands | `GoogleSpeechResNet` | 35 | $1 \times 64 \times 64$ Log-Mel |
| **Text (NLP)** | 20Newsgroups | `AlbertTextClassifier` | 20 | Sequence Token IDs |
| **Baseline** | Synthetic / MNIST | `MLP2NN` | 10 | Flattened Vectors |

### Heterogeneity Partitions
- **Dirichlet Skew**: $\text{Dir}(\alpha)$ with $\alpha \in \{0.1, 0.3, 0.5\}$ (lower $\alpha \implies$ extreme non-IID skew).
- **Pathological Skew**: Fixed class-quota or fractional ratio ($20\%$ and $30\%$ of total classes per client).

---

## Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/khastkhodaei/SemanticDFL.git
cd SemanticDFL

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Execution Modes

### Mode 1: High-Speed In-Memory Simulation
Runs zero-socket multi-client decentralized learning with exact communication accounting:

```bash
# Run simulation using default config (10 clients, Fashion-MNIST, Dirichlet alpha=0.3)
python run_simulation.py --config configs/simulation.yaml --rounds 20

# Override parameters from CLI
python run_simulation.py --rounds 10 --num_clients 20
```

### Mode 2: Distributed gRPC Network Execution
Orchestrates autonomous peer processes communicating over real TCP/IP sockets:

```bash
# Local multi-process cluster
python run_simulation.py --config configs/distributed_local.yaml --mode distributed
```

### Mode 3: Standalone Autonomous Peer Node
Spins up a single autonomous peer process joining a cluster:

```bash
# Start bootstrap super-peer (Node 0)
python run_node.py --node_id 0 --host 0.0.0.0 --port 50050 --is_super_peer --rounds 10

# Start worker peer (Node 1) connecting to bootstrap
python run_node.py --node_id 1 --host 0.0.0.0 --port 50051 --bootstrap 127.0.0.1:50050 --rounds 10
```

### Mode 4: Docker & SLICES Testbed Deployment
```bash
# Build and start 10-node SLICES testbed cluster via docker-compose
docker-compose -f docker-compose.slices.yml up --build
```

---

## Configuration Reference (`configs/default_config.yaml`)

```yaml
simulation:
  rounds: 20
  seed: 42
  device: "auto"

federated:
  num_clients: 10
  k_neighbors: 3
  zone_size: 5
  alpha_personalization: 0.67
  p_cardinality: 100
  son_refresh_interval: 1

network:
  topology: "small_world"  # Options: ring, small_world, geometric, grid, slices_wan
  degree: 4

data:
  dataset: "fmnist"        # Options: fmnist, cifar10, tiny_imagenet, google_speech, 20newsgroups
  partitioner: "dirichlet" # Options: dirichlet, pathological, iid
  alpha: 0.3
  batch_size: 32

model:
  architecture: "FashionCNN"
  learning_rate: 0.01
  local_epochs: 1

logging:
  output_dir: "results"
  generate_plots: true
  save_results_json: true
```

---

## Test Suite & Verification

The codebase includes 97 comprehensive unit and integration tests:

```bash
# Run the entire test suite
pytest -v
```

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

## Directory Structure

```
SemanticDFL/
├── configs/                   # Experiment and deployment configuration profiles
│   ├── default_config.yaml
│   ├── simulation.yaml
│   ├── distributed_local.yaml
│   └── slices_production.yaml
├── semantic_dfl/              # Core framework package
│   ├── aggregation/           # Softmax Pull Aggregator & Freshness Guard
│   ├── core/                  # Neural network models & parameter flattening
│   ├── data/                  # Multi-modal dataset loaders & Non-IID partitioners
│   ├── federated/             # Client trainer & centralized simulation coordinator
│   ├── overlay/               # SON hierarchical tree, VRF election, & physical topologies
│   ├── search/                # Bounded-fanout Top-K search & adaptive threshold controller
│   ├── signature/             # Sparse model signatures & cosine metrics
│   ├── transport/             # In-memory broker & gRPC network transport
│   └── visualization/         # Academic publication plotter
├── tests/                     # 97 comprehensive pytest unit & integration tests
├── results/                   # Simulation logs, JSON records, & publication figures
├── run_simulation.py          # Unified CLI simulation runner
├── run_node.py                # Standalone distributed peer process runner
├── docker-compose.slices.yml  # Dockerized 10-node SLICES testbed orchestration
├── requirements.txt           # Python dependencies
└── README.md                  # Comprehensive documentation
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
