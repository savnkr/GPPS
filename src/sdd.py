import torch


@torch.no_grad()
def train_sdd(C, y, batch_size=20, beta='auto', rho=0.98, r=0.99,
              num_epochs=2000, jitter=1e-6, sigma_n=0.0,
              verbose=False, print_every=500):
    """
    Stochastic Dual Descent to approximate C^{-1}y without explicit matrix inversion.

    Solves the dual objective:  min_alpha  (1/2) alpha^T K alpha - y^T alpha
    using mini-batch SGD with Nesterov momentum and iterate averaging.

    Reference: "Stochastic Gradient Descent for Gaussian Processes Done Right"
               Jihao Andreas Lin, Javier Antoran, Jose Miguel Hernandez-Lobato (2023).

    Args:
        C: Covariance matrix [N, N]
        y: Target vector/matrix [N, du]
        batch_size: Mini-batch size B for stochastic gradient estimates
        beta: Step size (learning rate). Use 'auto' to set adaptively based on
              spectral radius of K and batch ratio N/B.
        rho: Momentum parameter (Nesterov-style look-ahead)
        r: Exponential moving average parameter for iterate averaging
        num_epochs: Number of optimization iterations
        jitter: Regularization added to diagonal of C for numerical stability
        sigma_n: Likelihood noise variance (0 for noiseless PDE constraints)
        verbose: Whether to print convergence progress
        print_every: Frequency of progress printing (in iterations)

    Returns:
        A_bar: Approximation of C^{-1}y, shape [N, du]
    """
    N = C.shape[0]
    du = y.shape[1]
    dtype = C.dtype
    device = C.device
    B = min(batch_size, N)

    K = C + jitter * torch.eye(N, dtype=dtype, device=device)

    # Auto-select step size for stability with Nesterov momentum.
    # Effective step per gradient component is beta * (N/B).
    # Stability requires: beta * (N/B) * lambda_max < 1 (conservative Nesterov bound).
    if beta == 'auto':
        # Power iteration for largest eigenvalue (O(N^2) per iter, 20 iters)
        v = torch.randn(N, 1, dtype=dtype, device=device)
        for _ in range(30):
            v = K @ v
            v = v / torch.norm(v)
        lambda_max = (v.T @ K @ v).item()
        beta = 0.5 * B / (N * lambda_max)
        if verbose:
            print(f"    SDD auto beta={beta:.4f} (lambda_max~{lambda_max:.4e}, N={N}, B={B})")

    A_t = torch.zeros(N, du, dtype=dtype, device=device)
    V_t = torch.zeros(N, du, dtype=dtype, device=device)
    A_bar_t = torch.zeros(N, du, dtype=dtype, device=device)

    for t in range(num_epochs):
        # Nesterov-style look-ahead
        S = A_t + rho * V_t

        # Mini-batch sampling (without replacement for unbiased gradient)
        It = torch.randperm(N, device=device)[:B]

        # Stochastic gradient: unbiased estimate of (K @ alpha - y)
        # Only non-zero at sampled indices, scaled by N/B for unbiasedness
        G_t = torch.zeros(N, du, dtype=dtype, device=device)
        G_t[It] = (N / B) * (K[It] @ S - y[It])

        # Momentum update
        V_t = rho * V_t - beta * G_t
        A_t = A_t + V_t

        # Iterate averaging (Polyak-Ruppert style with EMA)
        A_bar_t = r * A_t + (1 - r) * A_bar_t

        if verbose and (t % print_every == 0 or t == num_epochs - 1):
            pred = K @ A_t
            loss = 0.5 * torch.norm(y - pred) ** 2
            if sigma_n > 0:
                loss += (sigma_n / 2) * torch.sum(A_t * (K @ A_t))
            print(f"    SDD step {t}/{num_epochs}, loss: {loss.item():.6e}")

    return A_bar_t
