import numpy as np
from tqdm import tqdm

def fastICA_prewhitened(
    X_white: np.ndarray,
    n_components: int, 
    max_iter: int = 200,
    tol: float = 1e-5,
    random_state: int = None
):
    """
    FastICA assuming the data X_white is already whitened.

    We will extract as many independent components as the dimension of X_white,
    i.e. if X_white has shape (d, n_samples), then we find d components.

    Parameters
    ----------
    X_white : ndarray of shape (d, n_samples)
        Pre-whitened data. Each column is a sample, each row is one feature dimension.
        The covariance of X_white is approximately the identity.
    n_components : int
        Number of independent components to estimate.
    max_iter : int, optional
        Maximum number of fixed-point iterations per component.
    tol : float, optional
        Convergence tolerance. The algorithm stops early if subsequent
        updates of the weight vector do not change significantly (up to sign).
    random_state : int, optional
        Seed for reproducible initializations.

    Returns
    -------
    S : ndarray of shape (d, n_samples)
        Estimated source signals. Each row is one independent component.
    W : ndarray of shape (d, d)
        The estimated unmixing matrix. Then S = W @ X_white.
    """
    rng = np.random.default_rng(random_state)

    d, n_samples = X_white.shape

    # Matrix for storing the unmixing vectors
    # Each row of W will be one separation vector.
    W = np.zeros((n_components, d), dtype=X_white.dtype)

    # For each component i in [0, d):
    for i in range(n_components):
        # 1) Random init for w_i
        w = rng.normal(size=(d,))
        w /= np.linalg.norm(w)

        # 2) Fixed-point iteration
        for _ in tqdm(range(max_iter)):
            # w^T x => shape (n_samples,)
            wx = w @ X_white

            # Nonlinearity g(u) = u^3
            g = wx**3
            # Derivative g'(u) = 3u^2
            g_prime = 3.0 * (wx**2)

            # Update rule:
            # w_new = E[x*g(w^T x)] - E[g'(w^T x)] * w
            # In whitened space, E[g'(wx)] ~ 3 if var(wx)=1, but we compute it exactly:
            w_new = (X_white * g).mean(axis=1) - g_prime.mean() * w

            # 3) Decorrelate from previously found components
            for j in range(i):
                # Remove projection on each previously found w_j
                w_new -= (w_new @ W[j]) * W[j]

            # 4) Normalize
            norm_w_new = np.linalg.norm(w_new)
            if norm_w_new < 1e-12:
                # Degenerate, re-initialize
                w_new = rng.normal(size=(d,))
                w_new /= np.linalg.norm(w_new)
            else:
                w_new /= norm_w_new

            # 5) Convergence check (up to sign)
            diff = np.abs(np.abs(w_new @ w) - 1.0)
            w = w_new
            if diff < tol:
                break

        # Store the found component
        W[i, :] = w

    # 6) Compute the estimated sources
    S = W @ X_white

    return S, W

