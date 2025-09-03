"""
Bayesian Optimization Core Library


This module provides core components for Bayesian optimization.

- Parameter space definitions (Continuous, Integer, ParamSpace)
- Gaussian Process utilities (kernels, prediction, hyperparameter fitting)
- Acquisition functions (EI, PI, UCB, LCB)

Design philosophy:
- Flexible abstraction of parameter space for use from Optimizer
- Fast numerical computation and automatic differentiation via JAX
- Reusable for both research and production

Main use cases:
- Black-box function optimization
- Hyperparameter search
- Experiment planning and automation

All components are designed to be reusable and extensible for other optimization algorithms.
"""

from collections import namedtuple
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax.scipy.linalg import cholesky, solve_triangular
from jax.scipy.stats import norm

jax.config.update("jax_enable_x64", True)
from abc import ABC, abstractmethod

class ParameterDomain(ABC):
    """パラメータ空間の抽象基底クラス"""

    def __init__(self, dtype):
        self.dtype = dtype

    def __hash__(self):
        return hash((type(self), self.dtype))

    def __eq__(self, other):
        return type(self) == type(other) and self.dtype == other.dtype

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def sample(self, key, shape):
        pass



class Continuous(ParameterDomain):
    """連続値パラメータ空間"""

    def __init__(self, lower: float, upper: float):
        if not isinstance(lower, (float, int)):
            raise TypeError(f"lower must be float or int, got {type(lower)}")
        if not isinstance(upper, (float, int)):
            raise TypeError(f"upper must be float or int, got {type(upper)}")
        if lower > upper:
            raise ValueError("lower must be <= upper")
        self.lower = float(lower)
        self.upper = float(upper)
        super().__init__(dtype=jnp.float32)

    def __hash__(self):
        return hash((type(self), self.lower, self.upper))

    def __eq__(self, other):
        return (
            isinstance(other, Continuous)
            and self.lower == other.lower
            and self.upper == other.upper
        )

    def transform(self, x):
        return jnp.clip(x, self.lower, self.upper)

    def sample(self, key, shape):
        samples = jax.random.uniform(key, shape, minval=self.lower, maxval=self.upper)
        return self.transform(samples)


class Integer(ParameterDomain):
    """
    Discrete integer-valued domain with rounding and clipping.

    Represents a parameter that can take integer values within [lower, upper].
    """

    def __init__(self, lower, upper):
        """
        Initialize an integer domain with bounds.

        Args:
            lower: Lower integer bound (inclusive).
            upper: Upper integer bound (inclusive).
        """
        if not isinstance(lower, int):
            raise TypeError("Lower bound must be an integer")
        if not isinstance(upper, int):
            raise TypeError("Upper bound must be an integer")
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound")

        self.lower = int(lower)
        self.upper = int(upper)
        super().__init__(dtype=jnp.int32)

    def __hash__(self):
        """Return hash based on bounds."""
        return hash((self.lower, self.upper))

    def __eq__(self, other):
        """Check equality based on bounds."""
        return self.lower == other.lower and self.upper == other.upper

    def transform(self, x: jax.Array):
        """
        Round and clip values to the integer domain.

        Args:
            x: Input values.

        Returns
        -------
            Rounded and clipped values as float32.
        """
        return jnp.clip(jnp.round(x), self.lower, self.upper).astype(jnp.float32)

    def sample(self, key: jax.Array, shape: tuple):
        """
        Sample integers uniformly from the domain.

        Args:
            key: JAX PRNGKey.
            shape: Desired output shape.

        Returns
        -------
            Sampled values clipped to valid integer range.
        """
        samples = jax.random.randint(
            key, shape, minval=self.lower, maxval=self.upper + 1
        )
        return self.transform(samples)



class ParamSpace:
    """
    Parameter space manager for collections of named parameter domains.

    This utility encapsulates logic for sampling, transforming, and handling
    structured parameter inputs defined by a mapping of variable names to ParameterDomain
    instances (e.g., Continuous, Integer).
    """

    def __init__(self, space: dict):
        self.space = space

    def sample_params(self, key: jax.Array, shape: tuple) -> dict:
        """Sample parameters from all domains."""
        keys = jax.random.split(key, len(self.space))
        return {
            name: self.space[name].sample(k, shape)
            for name, k in zip(self.space, keys)
        }

    def to_array(self, tree: dict) -> jax.Array:
        """
        Transform a batch of parameter values into a 2D array suitable for GP input.

        Args:
            tree: A dictionary of parameter name → array of raw values.

        Returns
        -------
            A JAX array of shape (batch_size, num_params) with transformed values.
        """
        return jnp.stack([self.space[k].transform(tree[k]) for k in self.space], axis=1)

    def to_dict(self, xs: jax.Array) -> dict:
        """
        Convert a stacked parameter matrix back into named parameter trees.

        Args:
            xs: A 2D JAX array of shape (batch_size, num_params), with each column
                corresponding to a parameter.

        Returns
        -------
            A dictionary mapping parameter names to individual 1D arrays.
        """
        return {k: self.space[k].transform(xs[:, i]) for i, k in enumerate(self.space)}


# =============================================================================
# Gaussian Process Components
# =============================================================================

MASK_VARIANCE = 1e12  # High variance for masked points to not affect the process.

GPParams = namedtuple("GPParams", ["noise", "amplitude", "lengthscale"])
GPState = namedtuple("GPState", ["params", "momentums", "scales"])


def softplus(x):
    """Softplus activation function."""
    return jnp.logaddexp(x, 0.0)


def exp_quadratic(x1, x2, mask):
    """Exponential quadratic kernel function."""
    distance = jnp.sum((x1 - x2) ** 2)
    return jnp.exp(-distance) * mask


def cov(x1, x2, mask1, mask2):
    """Covariance function with masking."""
    M = jnp.outer(mask1, mask2)
    k = exp_quadratic
    return jax.vmap(jax.vmap(k, in_axes=(None, 0, 0)), in_axes=(0, None, 0))(x1, x2, M)


def gaussian_process(
    params,
    x: jnp.ndarray,
    y: jnp.ndarray,
    mask,
    xt: Any = None,
    compute_ml: bool = False,
) -> Any:
    """Core Gaussian Process implementation with masking support."""
    # Number of points in the prior distribution
    n = x.shape[0]

    noise, amp, ls = jax.tree_util.tree_map(softplus, params)

    ymean = jnp.mean(y, where=mask.astype(bool))
    y = (y - ymean) * mask
    x = x / ls
    K = amp * cov(x, x, mask, mask) + (jnp.eye(n) * (noise + 1e-6))
    K += jnp.eye(n) * (1.0 - mask.astype(float)) * MASK_VARIANCE
    L = cholesky(K, lower=True)
    K_inv_y = solve_triangular(L.T, solve_triangular(L, y, lower=True), lower=False)

    if compute_ml:
        logp = 0.5 * jnp.dot(y.T, K_inv_y)
        logp += jnp.sum(jnp.log(jnp.diag(L)))
        logp -= jnp.sum(1.0 - mask) * 0.5 * jnp.log(MASK_VARIANCE)
        logp += (jnp.sum(mask) / 2) * jnp.log(2 * jnp.pi)
        logp += jnp.sum(-0.5 * jnp.log(2 * jnp.pi) - jnp.log(amp) - jnp.log(amp) ** 2)
        return jnp.sum(logp)

    if xt is None:
        raise ValueError("xt cannot be None during prediction")
    xt = xt / ls

    # Compute the covariance with the new point xt
    mask_t = jnp.ones(len(xt)) == 1
    K_cross = amp * cov(x, xt, mask, mask_t)

    K_inv_y = K_inv_y * mask  # masking
    pred_mean = jnp.dot(K_cross.T, K_inv_y) + ymean
    v = solve_triangular(L, K_cross, lower=True)
    pred_var = amp * cov(xt, xt, mask_t, mask_t) - v.T @ v
    pred_std = jnp.sqrt(jnp.maximum(jnp.diag(pred_var), 1e-10))
    return pred_mean, pred_std


marginal_likelihood = partial(gaussian_process, compute_ml=True)
grad_fun = jax.jit(jax.grad(marginal_likelihood))
predict = jax.jit(partial(gaussian_process, compute_ml=False))


def neg_log_likelihood(params, x, y, mask):
    """Negative log likelihood for GP hyperparameter optimization."""
    ll = marginal_likelihood(params, x, y, mask)

    # Weak priors to keep things sane
    priors = GPParams(-8.0, 1.0, 1.0)
    log_prior = jax.tree.map(lambda p, m: jnp.sum((p - m) ** 2), params, priors)
    log_prior = sum(jax.tree.leaves(log_prior))
    log_posterior = ll - 0.5 * log_prior
    return -log_posterior


def posterior_fit(
    y: jax.Array,
    x: jax.Array,
    mask: jax.Array,
    params: GPParams,
    lr: float = 1e-3,
    trainsteps: int = 100,
) -> GPParams:
    """Fit GP hyperparameters using gradient descent."""
    optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adamw(lr))
    opt_state = optimizer.init(params)

    def train_step(carry, _):
        params, opt_state = carry
        grads = jax.grad(neg_log_likelihood)(params, x, y, mask)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None

    (params, _), __ = jax.lax.scan(
        train_step, (params, opt_state), None, length=trainsteps
    )
    return params


# =============================================================================
# Acquisition Functions
# =============================================================================


def expected_improvement(
    x_pred: jnp.ndarray,
    xs: jax.Array,
    ys: jax.Array,
    mask: jax.Array,
    gp_params: GPParams,
    xi: float = 0.01,
):
    r"""
    Compute Expected Improvement (EI) acquisition function.

    Favors points with high improvement over the current best observed value,
    balancing exploitation and exploration.

    The formula is:

    .. math::

        EI(x) = (\mu(x) - y^* - \xi) \Phi(z) + \sigma(x) \phi(z)

    where:

    .. math::

        z = \frac{\mu(x) - y^* - \xi}{\sigma(x)}

    Args:
        x_pred: Candidate input locations to evaluate.
        xs: Observed inputs.
        ys: Observed function values.
        mask: Boolean mask indicating valid entries in `ys`.
        gp_params: Gaussian Process hyperparameters.
        xi: Exploration-exploitation tradeoff parameter.

    Returns
    -------
        EI scores at `x_pred`.
    """
    ymax = jnp.max(ys, where=mask.astype(bool), initial=-jnp.inf)
    mu, std = predict(gp_params, xs, ys, mask, xt=x_pred)
    a = mu - ymax - xi
    z = a / (std + 1e-3)
    ei = a * norm.cdf(z) + std * norm.pdf(z)
    return ei


def probability_improvement(
    x_pred: jnp.ndarray,
    xs: jax.Array,
    ys: jax.Array,
    mask: jax.Array,
    gp_params: GPParams,
    xi: float = 0.01,
):
    """
    Probability of Improvement (PI) acquisition function.

    Estimates the probability that a candidate point will improve
    over the current best observed value.

    Args:
        x_pred: Candidate input locations to evaluate.
        xs: Observed inputs.
        ys: Observed function values.
        mask: Boolean mask indicating valid entries in `ys`.
        gp_params: Gaussian Process hyperparameters.
        xi: Improvement margin for sensitivity.

    Returns
    -------
        PI scores at `x_pred`.
    """
    y_max = ys.max()
    mu, std = predict(gp_params, xs, ys, mask, xt=x_pred)
    z = (mu - y_max - xi) / std
    return norm.cdf(z)


def upper_confidence_bounds(
    x_pred: jnp.ndarray,
    xs: jax.Array,
    ys: jax.Array,
    mask: jax.Array,
    gp_params: GPParams,
    kappa: float = 0.01,
):
    """
    Upper Confidence Bound (UCB) acquisition function.

    Promotes exploration by favoring points with high predictive uncertainty.

    Args:
        x_pred: Candidate input locations to evaluate.
        xs: Observed inputs.
        ys: Observed function values.
        mask: Boolean mask indicating valid entries in `ys`.
        gp_params: Gaussian Process hyperparameters.
        kappa: Weighting factor for uncertainty.

    Returns
    -------
        UCB scores at `x_pred`.
    """
    mu, std = predict(gp_params, xs, ys, mask, xt=x_pred)
    return mu + kappa * std


def lower_confidence_bounds(
    x_pred: jnp.ndarray,
    xs: jax.Array,
    ys: jax.Array,
    mask: jax.Array,
    gp_params: GPParams,
    kappa: float = 2.576,
):
    """
    Lower Confidence Bound (LCB) acquisition function.

    Useful for minimization tasks. Encourages sampling in uncertain regions
    with low predicted values.

    Args:
        x_pred: Candidate input locations to evaluate.
        xs: Observed inputs.
        ys: Observed function values.
        mask: Boolean mask indicating valid entries in `ys`.
        gp_params: Gaussian Process hyperparameters.
        kappa: Weighting factor for uncertainty.

    Returns
    -------
        LCB scores at `x_pred`.
    """
    mu, std = predict(gp_params, xs, ys, mask, xt=x_pred)
    return mu - kappa * std
