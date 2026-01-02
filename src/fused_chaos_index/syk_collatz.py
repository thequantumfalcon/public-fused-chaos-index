from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad


@dataclass(frozen=True)
class SYKCollatzResult:
    C: float
    kurt_collatz: float
    kurt_syk: float
    N: int


def compute_syk_collatz_fci_constant(
    *,
    N: int = 32,
    kurt_collatz: float = 3.72,
    kurt_syk: float = -1.23,
    method: str = "analytic",
    seed: int = 42,
    n_mc_samples: int = 200_000,
) -> SYKCollatzResult:
    """Compute the SYK–Collatz constant-approximation FCI value.

    The research script effectively uses a constant approximation:
    $C = (kurt_{Collatz} * kurt_{SYK}) * \\int_0^\\infty (1/N) e^{-F/N} dF$.

    With that integrand, the integral is 1, so the analytic result is simply:
    $C = kurt_{Collatz} * kurt_{SYK}$.

    Parameters
    ----------
    method:
        "analytic" (default), "quad", or "mc".
    """

    N = int(N)
    S = float(kurt_collatz) * float(kurt_syk)

    if method == "analytic":
        C = S

    elif method == "quad":

        def integrand(F: float, N: int, S: float) -> float:
            p_syk = (1.0 / float(N)) * np.exp(-float(F) / float(N))
            return p_syk * float(S)

        C, _err = quad(integrand, 0.0, np.inf, args=(N, S), limit=100)
        C = float(C)

    elif method == "mc":
        rng = np.random.default_rng(int(seed))
        samples = rng.exponential(scale=float(N), size=int(n_mc_samples))
        # For the constant approximation, integrand(F)/p(F) = S.
        # We still compute it explicitly for clarity.
        p = (1.0 / float(N)) * np.exp(-samples / float(N))
        integrand_vals = p * S
        w = integrand_vals / p
        C = float(np.mean(w))

    else:
        raise ValueError("method must be 'analytic', 'quad', or 'mc'")

    return SYKCollatzResult(C=C, kurt_collatz=float(kurt_collatz), kurt_syk=float(kurt_syk), N=N)
