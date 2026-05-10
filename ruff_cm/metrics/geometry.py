import torch


def safe_normalize(x: torch.Tensor, *, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x, dim=dim, keepdim=True), min=eps)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    XtX = X @ X.T
    YtY = Y @ Y.T
    hsic_xy = (XtX * YtY).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()
    denom = torch.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else 0.0


def subspace_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    QA, _ = torch.linalg.qr(A)
    QB, _ = torch.linalg.qr(B)
    singular_values = torch.linalg.svdvals(QA.T @ QB)
    return torch.arccos(torch.clamp(singular_values, 0.0, 1.0)).flip(0)


def orthogonal_procrustes(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    U, _, Vt = torch.linalg.svd(A.T @ B)
    return U @ Vt


def compute_rdm_layers(centroids: torch.Tensor) -> torch.Tensor:
    c_trans = torch.moveaxis(centroids, 1, 0)
    v_norm = safe_normalize(c_trans)
    return 1.0 - torch.einsum("lid,ljd->lij", v_norm, v_norm)


def compute_pairwise_cosine_similarity(vectors: torch.Tensor, *, layer_dim: int = 1) -> torch.Tensor:
    n_layers = vectors.shape[layer_dim]
    by_layer = torch.moveaxis(vectors, layer_dim, 0)
    sim_matrices = []
    for layer_idx in range(n_layers):
        v_norm = safe_normalize(by_layer[layer_idx])
        if v_norm.ndim == 2:
            sim = torch.einsum("id,jd->ij", v_norm, v_norm)
        elif v_norm.ndim == 3:
            sim = torch.mean(torch.einsum("ikd,jkd->kij", v_norm, v_norm), dim=0)
        else:
            raise ValueError(f"Unsupported vector shape: {tuple(v_norm.shape)}")
        sim_matrices.append(sim)
    return torch.stack(sim_matrices, dim=0)


def compute_rule_axis(rule_vectors, *, n_components: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(rule_vectors, dict):
        levels = sorted(rule_vectors)
        matrix = torch.stack([rule_vectors[level] for level in levels], dim=0)
    else:
        matrix = rule_vectors
        levels = list(range(matrix.shape[0]))

    n_levels, n_layers, hidden_dim = matrix.shape
    axes = torch.zeros(n_layers, n_components, hidden_dim, device=matrix.device, dtype=matrix.dtype)
    explained = torch.zeros(n_layers, device=matrix.device, dtype=matrix.dtype)
    ordinal = torch.tensor(levels, device=matrix.device, dtype=matrix.dtype)

    # Orient the first component so projections increase with the ordered rule levels.
    for layer_idx in range(n_layers):
        task_matrix = matrix[:, layer_idx, :] - matrix[:, layer_idx, :].mean(0, keepdim=True)
        _, S, Vt = torch.linalg.svd(task_matrix, full_matrices=False)
        comp = Vt[:n_components].clone()
        if n_components >= 1 and n_levels > 1:
            projections = task_matrix @ comp[0]
            if ((projections - projections.mean()) * (ordinal - ordinal.mean())).sum() < 0:
                comp[0] = -comp[0]
        axes[layer_idx] = safe_normalize(comp)
        total_var = (S**2).sum()
        explained[layer_idx] = (S[:n_components] ** 2).sum() / total_var if total_var > 0 else 0.0

    return axes.squeeze(1) if n_components == 1 else axes, explained
