# 🚀 Project Findings & Architecture Summary

## 🏆 Current Best Configuration
**Status**: Stable & Optimized
**Performance**: ~22-24 PSNR (Car Dataset)

| Component | Choice | Reasoning |
| :--- | :--- | :--- |
| **Architecture** | `TwoStageModel` | Splits geometry (Stage 1) and appearance (Stage 2) learning. |
| **Position Decoder** | **Absolute + Zero Init** | Residual learning failed due to topology differences. Zero initialization prevents exploding gradients. |
| **Graph Structure** | **Static k-NN (k=6)** | Dynamic graphs (`EdgeConv`) were 6.5x slower with minimal gain. |
| **GNN Layers** | `PointGNN` & `LINKX` | `PointGNN` for geometric offsets, `LINKX` (w/ `SAGEConv`) for attribute mixing. |
| **Regularization** | **Anisotropy (0.01)** | Essential. Prevents "pancaking" (anisotropy > 100k) without hurting PSNR. |

---

## 🧠 Key Insights & Experiments

### 1. Position Prediction: The "Teleportation" Problem
*   **Hypothesis**: Residual prediction (`pos + delta`) should be easier than absolute prediction.
*   **Finding**: **False for this task.** The topological gap between the initial sphere and the target car is too large. Residual models got stuck in local minima.
*   **Solution**: **Absolute Position Prediction**. Allowing the model to predict coordinates directly works best, *provided* the final layer is initialized to zero (starting as a sphere) to avoid initial chaos.
*   **Failed Attempt**: Bounding positions with `Tanh` caused saturation and vanishing gradients.

### 2. Graph Dynamics: Speed vs. Flexibility
*   **Experiment**: Replaced static k-NN with `EdgeConv` (Dynamic Graph).
*   **Result**:
    *   **Quality**: Comparable PSNR (~19.6).
    *   **Cost**: **6.5x slowdown** (5.4 it/s -> 0.83 it/s) and 3x memory usage.
*   **Decision**: Stick to **Static Graph**. The computational cost of recomputing neighbors every forward pass outweighs the topological flexibility for this dataset.

### 3. Anisotropy & Geometry Control
*   **Issue**: The base model tends to flatten Gaussians into infinite planes (Anisotropy > 200,000) to cover the surface.
*   **Fix**: Added `anisotropy_penalty = 0.01`.
*   **Result**: Anisotropy dropped to ~1.3 (healthy 3D shapes) while **improving** PSNR. This proves extreme flattening is an artifact, not a requirement.

### 4. Layer Optimization (Refined Architecture)
*   **LINKX Layer**:
    *   Replaced custom `SparseLinear` with standard `SAGEConv`.
    *   **Benefit**: Makes the model **inductive** (works with any number of points) and removes custom sparse matrix handling.
    *   **Fix**: Added `LeakyReLU` between the convolution and the next linear layer to restore non-linearity.
*   **PointGNNConv**:
    *   Removed unused `mlp_f` branch (dead code).
    *   **Fix**: Removed the final activation in the residual branch to allow symmetric updates.
*   **RMSNorm**:
    *   **Status**: **Adopted**.
    *   **Insight**: Initially caused divergence, but works perfectly when combined with **Zero-Initialization** of the position decoder. It stabilizes feature distributions in the deeper MLP layers.

### 5. Failed Experiments (What didn't work)
*   **Opacity Regularization**: Penalizing intermediate opacity values pushed the model to make all Gaussians invisible (Opacity ~0.0).

---

## 📏 Evaluation Metrics & Diagnostics
To ensure the model isn't just "cheating" (e.g., predicting mean color) or failing silently, we track:

1.  **PSNR (Peak Signal-to-Noise Ratio)**:
    *   *Goal*: > 20.0.
    *   *Meaning*: Primary measure of visual reconstruction quality.
2.  **Anisotropy (Max Scale / Min Scale)**:
    *   *Goal*: ~1.0 - 5.0 (Healthy 3D shapes).
    *   *Failure Mode*: > 100,000 (Infinite planes/pancakes).
3.  **Displacement (Mean distance from origin)**:
    *   *Goal*: Stable growth from ~1.0.
    *   *Failure Mode*: Explodes to > 3.0 (Points flying away) or stays exactly 1.0 (No learning).
4.  **Opacity & Scale Histograms**:
    *   *Goal*: Balanced distribution.
    *   *Failure Mode*: All opacity ~0.0 (Invisible) or Scale ~0.0 (Degenerate points).
5.  **Gradient Norm**:
    *   *Goal*: ~1.0.
    *   *Failure Mode*: 0.0 (Dead neurons) or NaN (Explosion).

---

## 📊 Performance Benchmarks (20 Epochs)

| Model Variant | PSNR | Anisotropy | Speed | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Base (Unregularized)** | 23.72 | ~3,650 (Peak >130k) | ~4.5 it/s | Unstable geometry. |
| **Regularized (0.01)** | **23.88** | **~1.33** | ~4.5 it/s | **Best balance.** |
| **Dynamic Graph** | ~19.6 | ~92,000 | 0.83 it/s | Too slow. |
| **Residual Pos** | ~10.4 | N/A | ~4.5 it/s | Failed convergence. |
| **Optimized (SAGE)** | ~21.9 | ~2.1 | ~4.4 it/s | Cleaner code, good quality. |

## 🔬 General Experiment Setup (Baseline for GAN)
This experiment serves as the **Generator architecture validation** phase. The current model is designed to function as the **Generator (G)** in a future 3D-GAN framework.

*   **Task**: Single-Object 3D Reconstruction (Overfitting).
*   **Dataset**: Synthetic Car (Multi-view RGB images, White background).
*   **Input Representation (Latent Space)**:
    *   **Topology**: Fibonacci Sphere (8192 points, radius 1.0).
    *   **Features**: Fourier Positional Embeddings of the sphere coordinates.
    *   *GAN Note*: For generation, this input can be conditioned on a latent vector $z$ (global feature) injected into the `GlobalConv` layer.
*   **Model Architecture (The Generator)**:
    *   **Stage 1 (Geometry)**: `PointGNNConv` layers to deform the sphere into the coarse object shape.
    *   **Stage 2 (Appearance)**: `LINKX` layers (SAGEConv) to refine details and predict Gaussian attributes (Color, Opacity, Scale, Rotation).
    *   **Output**: Set of 3D Gaussians ready for rasterization.
*   **Rendering**:
    *   **Method**: Differentiable Gaussian Rasterization.
    *   **Resolution**: 128x128.
*   **Training Configuration**:
    *   **Optimizer**: AdamW (`lr=0.001`, `weight_decay=0.01`).
    *   **Loss Function**: $\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{aniso} \cdot \mathcal{L}_{aniso}$.
    *   **Reconstruction Loss**: $\mathcal{L}_{recon} = (1 - \alpha) \cdot \mathcal{L}_{1} + \alpha \cdot (1 - \text{SSIM})$, with $\alpha=0.1$.
    *   **Anisotropy Regularization**: $\mathcal{L}_{aniso} = \text{mean}(\log(\frac{\max(s)}{\min(s)}))$. Penalizes extreme stretching to ensure volumetric consistency.
    *   *GAN Transition*: In the GAN setup, the reconstruction loss ($\mathcal{L}_{1}$) will be replaced or weighted against an Adversarial Loss ($\mathcal{L}_{adv}$) from a Discriminator.
