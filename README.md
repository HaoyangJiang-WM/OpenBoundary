# Physics-Refined Spatiotemporal Forecasting on Open-Boundary Hydrologic Graphs (PRSOBG)

This repository contains the implementation of **PRSOBG**, a framework designed to tackle instability in spatiotemporal forecasting for open-boundary systems (such as river networks and coastal meshes).

## 📄 Paper Summary

Standard Spatiotemporal Graph Neural Networks (STGNNs) often struggle in open-boundary systems because external forcing (e.g., upstream inflows or tidal signals) is unobserved at the boundary nodes. This "missing information" leads to error accumulation during autoregressive rollouts.

To solve this, the paper proposes a **"What-How" decomposition** framework:
1.  **What (Ghost Proxies):** The model learns the missing boundary forcing using **Ghost Node Proxies**. These auxiliary nodes approximate external inputs based on the observed dynamics between boundary nodes and their interior neighbors.
2.  **How (Physics Refiners):** The framework stabilizes the propagation of these forces using physics-informed components:
    * **Local Boundary Refiner:** Enforces consistency between ghost and boundary nodes using a Robin-type boundary condition.
    * **Global Physics Refiner:** A post-processing layer (either Explicit or Convexified) that regularizes the flow over the entire graph to prevent long-term numerical drift.

---

## 🛠️ Data Processing

* **`dataset_tt.py`**
    This script handles data preprocessing. It loads the raw hydrological data (discharge), constructs the graph topology (adjacency matrices), and prepares the training/testing splits for the model.
    dataset from: https://github.com/nkirschi/neural-flood-forecasting
---

## 🌊 River System Implementation

The following files represent the incremental build-up of the framework, specifically configured for the **River System** (Directed Graph).

### 1. Baseline Model
* **`main_river_stgnn.py`** (STGNN Backbone)
    * The vanilla Spatiotemporal GNN model **without** any ghost nodes or physics refinements.
    * This serves as the raw baseline to demonstrate the instability caused by missing boundary information.

### 2. Ghost Node Integration
* **`main_river_ghost.py`** (Ghost Proxy)
    * Builds on the backbone by adding **Explicit Upstream Ghost Nodes**.
    * Uses an MLP proxy to learn the unobserved boundary forcing from the immediate downstream neighbors.
    * *Note:* No physics refinement is applied here; it only addresses the input deficit.

### 3. Boundary Stabilization
* **`main_river_boundary_refiner.py`** (Ghost + Boundary Refiner)
    * Adds the **Local Boundary Refiner**.
    * Solves a strictly convex optimization problem at each step to align the ghost proxies with a discretized Robin boundary condition, reducing local noise before it propagates inward.

### 4. Global Explicit Refinement
* **`main_river_explicit_refiner.py`** (Ghost + Boundary Refiner + Upwind Explicit)
    * Adds the **Global Explicit Refiner**.
    * Applies a structured **Upwind** update layer based on the advective flow equation.
    * This acts as a lightweight regularization step during rollout.

### 5. Global Convexified Refinement (Full Model)
* **`main_river_convex_refiner.py`** (Ghost + Boundary Refiner + Convexified)
    * The complete PRSOBG model using the **Convexified (Semi-Implicit) Refiner**.
    * Instead of a simple explicit step, this solves a symmetric positive definite (SPD) system using a differentiable Preconditioned Conjugate Gradient (PCG) layer.
    * This provides the highest robustness and stability for long-horizon forecasting.

---

## 🔗 Citation

This paper is currently in submission to the **KDD AI4Science Track**.
