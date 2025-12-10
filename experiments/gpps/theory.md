# Active Learning with K-means Clustering for Gaussian Process PDE Solutions

## 1. Introduction

Summary of the project and its objectives.

$$
-\Delta u(x) = f(x) \quad \text{in } \Omega \\
u(x) = g(x) \quad \text{on } \partial\Omega
$$

Where $\Omega$ is a unit disk in $\mathbb{R}^2$, $f(x) $ is the forcing function, and $g(x)$ is the boundary condition.

---

## 2. Gaussian Process Framework for PDEs

### 2.1 Kernel and Covariance Functions

We use a Radial Basis Function (RBF) kernel defined as:

$$
k(x, x') = \sigma^2 \exp\left(-\frac{|x - x'|^2}{2\ell^2}\right)
$$

Where:
- $\sigma^2$ is the kernel variance
- $\ell$ is the lengthscale
- $x, x' \in \mathbb{R}^2$ are 2D input points

### 2.2 PDE and Boundary Operators

What am i doing in this. Let's try to understand this using one example. For the Poisson equation, the differential operator $\mathcal{L}$ is the Laplacian:

$$
\mathcal{L}u = -\Delta u = -\left(\frac{\partial^2 u}{\partial x_1^2} + \frac{\partial^2 u}{\partial x_2^2}\right)
$$

The boundary operator $\mathcal{B}$ is the identity operator for Dirichlet boundary conditions:

$$
\mathcal{B}u = u|_{\partial\Omega}
$$

### 2.3 GP Prior and Posterior

The Gaussian Process framework constructs a prior over functions, which is then conditioned on observed data. The full covariance matrix $C$ is:

$$
C = \begin{bmatrix} 
C_{ii} & C_{ib} \\
C_{bi} & C_{bb} 
\end{bmatrix}
$$

Where:
- $C_{ii} = \mathcal{L}_x \mathcal{L}_{x'} k(x, x')$ for interior-interior points
- $C_{ib} = \mathcal{L}_x k(x, x')$ for interior-boundary interactions
- $C_{bi} = C_{ib}^T$
- $C_{bb} = k(x, x')$ for boundary-boundary points

The posterior mean for a new point $x_*$ is:

$$
\mu(x_*) = c(x_*)^T C^{-1} y
$$

Where:
- $c(x_*)$ is the covariance vector between $x_*$ and all observation points
- $y$ is the vector of observations ($f(x)$ for interior, $g(x)$ for boundary)

The posterior variance is:

$$
\sigma^2(x_*) = k(x_*, x_*) - c(x_*)^T C^{-1} c(x_*)
$$

---

## 3. Active Learning for PDEs

### 3.1 Acquisition Functions

To maximize information gain, we use uncertainty-based acquisition functions. Some simple choice of acquisiton funciton are:

- **Pure Variance** -- Suggested by sir: 
  $$
  a(x) = \sigma^2(x)
  $$

- **Upper Confidence Bound (UCB)**: 
  $$
  a(x) = \mu(x) + \kappa \sigma(x)
  $$
  where $\kappa$ is the exploration parameter.

### 3.2 K-means Clustering for Diversity

To avoid oversampling high-uncertainty regions, we apply K-means clustering:

**Algorithm: K-means Clustering for Active Point Selection**
1. Generate a large candidate pool $X_{pool}$ within the domain.
2. Filter out candidates too close to existing points using distance threshold $r_{excl}$.
3. Compute acquisition values $a(x)$ for all remaining candidates.
4. Sort candidates by acquisition value (descending).
5. Select top $p\%$ candidates by uncertainty.
6. Apply K-means clustering to these, generating $K$ clusters.
7. Choose the point closest to each cluster centroid.

This ensures:
- High-uncertainty points are chosen
- Spatial diversity is preserved

### 3.3 Results Storage

All experimental results are saved in `.npz` or `.pkl` format for reproducibility and comparison, including:
- Training points and convergence metrics
- Comparison baselines (SDD with fixed 80 collocation points)
- Error evolution and uncertainty reduction tracking

### 3.4 Algorithm I followed

Let:
- $X_i$: interior training points
- $X_b$: boundary points
- $X_{pool}$: candidate pool
- $n_a$: number of points to select
- $\alpha_r$: active ratio

**Algorithm:**
1. Compute covariance matrix $C$ and inverse $C^{-1}$
2. For each $x \in X_{pool}$:
   $$
   \sigma^2(x) = k(x, x) - c(x)^T C^{-1} c(x)
   $$
3. Filter:
   $$
   X_{filtered} = \{x \in X_{pool} : \min_{x' \in X_i} ||x - x'|| \geq r_{excl} \}
   $$
4. Sort by variance: $X_{sorted} = \text{sort}(X_{filtered}, \sigma^2(x))$
5. Select top candidates:
   $$
   X_{top} = X_{sorted}[1:\max(n_a \cdot 2, \alpha_r \cdot |X_{filtered}|)]
   $$
6. Cluster $X_{top}$ into $n_a$ groups via K-means:
   $$
   \{C_1, C_2, ..., C_{n_a}\} = \text{KMeans}(X_{top}, n_a)
   $$
7. For each cluster $C_j$, pick:
   $$
   x_j^* = \arg\min_{x \in C_j} ||x - \text{centroid}(C_j)||
   $$
8. Return selected points:
   $$
   X_{new} = \{x_1^*, x_2^*, ..., x_{n_a}^*\}
   $$

---

## 4. Training Process with Active Learning

**Algorithm: Active Learning Loop**
1. Initialize with interior points $X_i$ and boundary points $X_b$.
2. For $t = 1$ to $T$:
   - Compute $C_t$ and $C_t^{-1}$
   - Select $X_{new}$ using K-means-based active learning
   - Update:
     - $X_i \leftarrow X_i \cup X_{new}$
     - $f(x) = -1$ for $x \in X_{new}$ $\quad$ (-1 is for possion)
3. Return final GP model

---

## 5. Experimental Results

Clustering-based active learning shows improvements over uniform and non-clustered methods:

- **Accuracy**: Lower mean absolute error
- **Uncertainty Reduction**: Faster posterior variance decay
- **Diversity**: More uniform domain coverage
- **Convergence**: Faster convergence in complex regions

---

## 6. Complete Algorithm with Mathematical Details

**Algorithm: Adaptive GP-based PDE Solver with Clustering**

**Input:**
- Initial interior points $X_i$
- Boundary points $X_b$
- Candidate pool $X_{pool}$
- Active learning iterations $T$
- Points per iteration $n_a$
- Kernel hyperparameters $(\sigma, \ell)$
- Exclusion radius $r_{excl}$
- Active ratio $\alpha_r$

**Output:**
- Trained GP model

**Steps:**
1. Initialize:
   - Kernel: $k(x,x') = \sigma^2 \exp(-\|x - x'\|^2 / 2\ell^2)$
   - Set $g(X_b) = 0$, $f(X_i) = -1$

2. For $t = 1$ to $T$:
   - Compute:
     - $C_{ii} = \mathcal{L}_x \mathcal{L}_{x'} k(X_i, X_i)$
     - $C_{ib} = \mathcal{L}_x k(X_i, X_b)$
     - $C_{bb} = k(X_b, X_b)$
     - $C = \begin{bmatrix} C_{ii} & C_{ib} \\ C_{ib}^T & C_{bb} \end{bmatrix}$
   - Invert $C$ 
   - For $x \in X_{pool}$:
     - Compute $c(x)$ and $\sigma^2(x)$
   - Filter:
     $$
     X_{filtered} = \{x \in X_{pool} : \min_{x' \in X_i} ||x - x'|| \geq r_{excl} \}
     $$
   - Rank and select top candidates $X_{top}$
   - Cluster $X_{top}$ via K-means
   - Select representative points: $\{x_1^*, ..., x_{n_a}^*\}$
   - Update:
     - $X_i \leftarrow X_i \cup \{x_1^*, ..., x_{n_a}^*\}$
     - $f(x_j^*) = -1$

3. Return final trained GP model which is done using SDD optimisation.

---
