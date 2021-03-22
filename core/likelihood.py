import numpy as np
import scipy as sp
from scipy import special
from scipy import stats


def gammaPriorPoissonLikelihood(k, alpha, beta):
    """Poisson distribution marginalized over the rate parameter, priored with
       a gamma distribution that has shape parameter alpha and inverse rate
       parameter beta.

    Parameters
    ----------
    k : int
        The number of observed events
    alpha : float
        Gamma distribution shape parameter
    beta : float
        Gamma distribution inverse rate parameter

    Returns
    -------
    float
        The log likelihood
    """
    values = [
        alpha * np.log(beta),
        sp.special.loggamma(k + alpha).real,
        -sp.special.loggamma(k + 1.0).real,
        -(k + alpha) * np.log1p(beta),
        -sp.special.loggamma(alpha).real,
    ]
    res = np.sum(values, axis=0)
    return res

# Add two value gradient tuples
def plus_grad(xg0, xg1, resdim0, resdim1, nres):
    x0, grad0 = xg0
    x1, grad1 = xg1
    x0 = np.atleast_1d(x0)
    resgrad = np.zeros(resgrad_shape(x0, nres))
    resgrad[..., resdim0] = grad0
    resgrad[..., resdim1] += grad1
    return x0 + x1, resgrad

def gammaPriorPoissonLikelihood_grad(k_, alpha_, beta_, resdim_k, resdim_alpha, resdim_beta, nres):
    k, k_g = k_
    alpha, alpha_g = alpha_
    beta, beta_g = beta_

    values = [
        alpha * op.log(beta),
        sp.special.loggamma(k + alpha).real,
        -sp.special.loggamma(k + 1.0).real,
        -(k + alpha) * np.log1p(beta),
        -sp.special.loggamma(alpha).real,
    ]


def poissonLikelihood(k, weight_sum, weight_sq_sum):
    """Computes Log of the Poisson Likelihood.

    Parameters
    ----------
    k : int
        the number of observed events
    weight_sum : float
        the sum of the weighted MC event counts
    weight_sq_sum : float
        the sum of the square of the weighted MC event counts

    Returns
    -------
    float
        The log likelihood
    """

    res = sp.stats.poisson.logpmf(k, weight_sum)
    return res


def LEff(k, weight_sum, weight_sq_sum):
    """Computes Log of the L_Eff Likelihood.
       This is the poisson likelihood, using a poisson distribution with
       rescaled rate parameter to describe the Monte Carlo expectation, and
       assuming a uniform prior on the rate parameter of the Monte Carlo.
       This is the main result of the paper arXiv:1901.04645

    Parameters
    ----------
    k : int
        the number of observed events
    weight_sum : float
        the sum of the weighted MC event counts
    weight_sq_sum : float
        the sum of the square of the weighted MC event counts

    Returns
    -------
    float
        The log likelihood
    """
    k = np.asarray(k)
    weight_sum = np.asarray(weight_sum)
    weight_sq_sum = np.asarray(weight_sq_sum)

    # Return -inf for an ill formed likelihood or 0 without observation
    res = np.zeros(np.shape(weight_sum))
    bad_mask = np.logical_and(np.logical_or(weight_sum <= 0, weight_sq_sum < 0), k != 0)
    res[bad_mask] = -np.inf
    res[bad_mask] = gammaPriorPoissonLikelihood(k[bad_mask], 1.0 + 1.0, 1e20)

    poisson_mask = weight_sq_sum == 0
    if np.any(poisson_mask):
        res[poisson_mask] = poissonLikelihood(
            k[poisson_mask], weight_sum[poisson_mask], weight_sq_sum[poisson_mask]
        )

    good_mask = np.logical_and(~bad_mask, ~poisson_mask)
    if np.any(good_mask):

        kk = k[good_mask]
        ws = weight_sum[good_mask]
        wss = weight_sq_sum[good_mask]

        alpha = np.power(ws, 2.0) / wss + 1.0
        beta = ws / wss
        L = gammaPriorPoissonLikelihood(kk, alpha, beta)
        res[good_mask] = L
    assert(not np.any(np.isnan(res)))
    return res
