import math
import torch


@torch.no_grad()
def train_sdd(C, y, batch_size=20, beta='auto', rho=0.9,
              num_epochs=2000, jitter=1e-6,
              precondition=True, burn_in_frac=0.5,
              epoch_scaling='auto',
              verbose=False, print_every=500):
    """Approximate ``C^{-1} y`` with stochastic dual descent."""
    N = C.shape[0]
    du = y.shape[1]
    dtype = C.dtype
    device = C.device
    B = min(batch_size, N)

    # Scale epochs with system size to maintain convergence quality
    if epoch_scaling == 'auto' and N > B:
        total_epochs = int(num_epochs * (N / B) * math.log(max(N, 2)) / math.log(max(B, 2)))
    else:
        total_epochs = num_epochs

    K = C + jitter * torch.eye(N, dtype=dtype, device=device)

    # Jacobi (diagonal) preconditioner
    if precondition:
        P_inv = 1.0 / K.diag().clamp(min=1e-10)
    else:
        P_inv = torch.ones(N, dtype=dtype, device=device)

    # Auto-select step size based on spectral radius of preconditioned system
    if beta == 'auto':
        v = torch.randn(N, 1, dtype=dtype, device=device)
        for _ in range(30):
            v = P_inv.unsqueeze(1) * (K @ v)
            v = v / torch.norm(v)
        lambda_max_prec = (v.T @ (P_inv.unsqueeze(1) * (K @ v))).item()
        beta = 0.5 * B / (N * lambda_max_prec)
        if verbose:
            print(f"    SDD auto beta={beta:.6f} (prec_lmax~{lambda_max_prec:.2e}, "
                  f"N={N}, B={B}, epochs={total_epochs})")

    A_t = torch.zeros(N, du, dtype=dtype, device=device)
    V_t = torch.zeros(N, du, dtype=dtype, device=device)
    A_bar = torch.zeros(N, du, dtype=dtype, device=device)

    burn_in = int(total_epochs * burn_in_frac)
    n_avg = 0

    for t in range(total_epochs):
        # Mini-batch sampling (without replacement)
        It = torch.randperm(N, device=device)[:B]

        # Preconditioned stochastic gradient (heavy-ball: gradient at current position)
        raw_grad = K[It] @ A_t - y[It]
        G_t = torch.zeros(N, du, dtype=dtype, device=device)
        G_t[It] = (N / B) * P_inv[It].unsqueeze(1) * raw_grad

        # Heavy-ball momentum update
        V_t = rho * V_t - beta * G_t
        A_t = A_t + V_t

        # Tail averaging: only average iterates after burn-in
        if t >= burn_in:
            n_avg += 1
            A_bar = A_bar + (A_t - A_bar) / n_avg

        if verbose and (t % print_every == 0 or t == total_epochs - 1):
            out = A_bar if n_avg > 0 else A_t
            loss = 0.5 * torch.norm(y - K @ out) ** 2
            print(f"    SDD step {t}/{total_epochs}, loss: {loss.item():.6e}")

    return A_bar if n_avg > 0 else A_t
