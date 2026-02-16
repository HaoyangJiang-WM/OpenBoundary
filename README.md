dataset from: https://github.com/nkirschi/neural-flood-forecasting


# Physics-Refined Spatiotemporal Forecasting on Open-Boundary Hydrologic Graphs (PRSOBG)

This repository implements the **PRSOBG** framework, designed to solve instability issues in spatiotemporal forecasting for open-boundary systems (such as river networks and coastal meshes).

## 📄 Paper Summary

[cite_start]Standard Spatiotemporal Graph Neural Networks (STGNNs) often fail in open-boundary systems because external forcing (e.g., upstream inflows or tides) is unobserved at the boundary nodes[cite: 66, 140]. This leads to error accumulation during autoregressive rollouts.

[cite_start]To address this, this paper proposes a **"What-How" decomposition** framework[cite: 254]:
1.  **What (Ghost Proxies):** Learns missing boundary forcing using **Ghost Node Proxies**. [cite_start]These are auxiliary nodes that approximate external inputs based on the dynamics between boundary nodes and their interior neighbors[cite: 73, 404].
2.  **How (Physics Refiners):** Stabilizes the propagation of these forces using two physics-informed components:
    * [cite_start]**Local Boundary Refiner:** Enforces consistency between ghost and boundary nodes using a Robin-type boundary condition[cite: 75, 409].
    * [cite_start]**Global Physics Refiner:** A post-processing layer (Explicit or Convexified) that regularizes the flow over the entire graph to prevent long-term numerical drift[cite: 76, 248].

---

## 🛠️ Data Preparation

* **`dataset_tt.py`**
    This script handles data preprocessing. [cite_start]It loads the raw hydrological data (e.g., discharge, water levels), constructs the graph structure (adjacency matrices), and prepares the training/testing splits for the model[cite: 968].

---

## 🌊 River System Implementation

[cite_start]The following files represent the incremental build-up of the framework, specifically configured for the **Directed River System** (e.g., Danube River)[cite: 271, 969].

### 1. Baseline Model
* **`main_river_stgnn.py`** (STGNN Backbone)
    * The vanilla Spatiotemporal GNN model **without** any ghost nodes or physics refinements.
    * This serves as the baseline to demonstrate the instability and error drift caused by missing boundary information[cite: 983].

### 2. Ghost Node Integration
* **`main_river_ghost.py`** (Ghost-Only)
    * Builds on the backbone by adding **Explicit Upstream Ghost Nodes**[cite: 495].
    * [cite_start]Uses an MLP proxy to learn the unobserved boundary forcing from the immediate downstream neighbors[cite: 404].
    * *Note:* No physics refinement is applied here; it only addresses the input deficit.

### 3. Boundary Stabilization
* **`main_river_boundary_refiner.py`** (Ghost + Boundary Refiner)
    * Adds the **Local Boundary Refiner**[cite: 658].
    * [cite_start]Solves a strictly convex optimization problem at each step to align the ghost proxies with a discretized Robin boundary condition, reducing local noise before it propagates inward[cite: 660].

### 4. Global Explicit Refinement
* **`main_river_explicit_refiner.py`** (Ghost + Boundary Refiner + Explicit)
    * Adds the **Global Explicit Refiner**[cite: 751].
    * [cite_start]Applies a structured "Upwind" update layer based on the advective flow equation (Eq. 10 in the paper)[cite: 830].
    * This acts as a lightweight regularization step during rollout.

### 5. Global Convexified Refinement (Full Model)
* **`main_river_convex_refiner.py`** (Ghost + Boundary Refiner + Convexified)
    * The complete PRSOBG model using the **Convexified (Semi-Implicit) Refiner**[cite: 805].
    * [cite_start]Instead of a simple explicit step, this solves a symmetric positive definite (SPD) system using a differentiable Preconditioned Conjugate Gradient (PCG) layer[cite: 809].
    * [cite_start]This provides the highest robustness and stability for long-horizon forecasting[cite: 1292].

---

## 🔗 Citation

If you use this code or framework, please refer to the original paper:
> **Physics-Refined Spatiotemporal Forecasting on Open-Boundary Hydrologic Graphs**
> *Jiang et al., KDD 2026*
