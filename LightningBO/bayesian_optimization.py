
"""
Bayesian Optimization High-Level Interface

This module provides a high-level API for Bayesian optimization using Gaussian Processes.
It manages the optimization loop, parameter space, and acquisition functions.

Main features:
- Optimizer class for black-box function optimization
- Flexible parameter space definition (Continuous, Integer, etc.)
- Acquisition functions: EI, PI, UCB, LCB
- Automatic state management and buffer expansion
- JAX-based fast computation and differentiation

Typical use cases:
- Hyperparameter tuning
- Experimental design
- Automated search for expensive functions

Designed for extensibility and integration with custom objective functions.
"""


from collections.abc import Callable
from functools import partial
from typing import NamedTuple


import jax
from typing import Any
import jax.numpy as jnp
import numpy as np
import optax

from .bayesian_core import (
    GPParams,
    ParamSpace,
    expected_improvement,
    lower_confidence_bounds,
    posterior_fit,
    probability_improvement,
    upper_confidence_bounds,
)


UtilityFunctionType = Callable[[jnp.ndarray], float]



class OptimizerState(NamedTuple):
    """
    State container for Bayesian optimizer.

    Attributes:
        params: dict[str, jnp.ndarray] - Padded parameter arrays.
        ys: jnp.ndarray | np.ndarray - Objective values (with padding).
        best_score: float - Best observed objective value.
        best_params: dict[str, float] - Parameter configuration for best_score.
        mask: jnp.ndarray - Boolean mask for valid entries.
        gp_params: GPParams - Fitted Gaussian Process parameters.
    """
    params: dict[str, jnp.ndarray]
    ys: jnp.ndarray | np.ndarray
    best_score: float
    best_params: dict[str, float]
    mask: jnp.ndarray
    gp_params: GPParams



def _optimize_suggestion(params: jnp.ndarray, fun: Callable[[jnp.ndarray], float], max_iter: int = 10) -> jnp.ndarray:
    """
    Local optimization using L-BFGS (Optax).

    Args:
        params: Initial point in input space.
        fun: Objective function to maximize (returns scalar, supports autodiff).
        max_iter: Number of L-BFGS steps.

    Returns:
        Optimized point after `max_iter` iterations.
    """
    opt = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(lambda x: -fun(x))

    def step(carry, _):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params=params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, -1e6, 1e6)
        return (params, state), None

    init_carry = (params, opt.init(params))
    (final_params, _), __ = jax.lax.scan(step, init_carry, None, length=max_iter)
    return final_params



class Optimizer:
    """
    Bayesian optimizer using Gaussian Processes and acquisition functions.

    Manages the optimization loop for expensive black-box functions by modeling
    them with a Gaussian Process and selecting samples via acquisition functions
    (EI, PI, UCB, LCB).
    """

    def __init__(self, domain: dict[str, Any], acq: str = "EI", maximize: bool = False):
        """
        Initialize the optimizer.

        Args:
            domain: A dict mapping parameter names to domain objects (e.g., Real, Integer).
            acq: Acquisition function ('EI', 'PI', 'UCB', 'LCB').
            maximize: Whether to maximize or minimize the objective.
        """
        self.domain = domain
        self.sign = 1 if maximize else -1
        self.param_space = ParamSpace(domain)

        acq_map = {
            "EI": expected_improvement,
            "PI": probability_improvement,
            "UCB": upper_confidence_bounds,
            "LCB": lower_confidence_bounds,
        }
        if acq not in acq_map:
            raise ValueError(f"Acquisition function {acq} is not implemented")
        self.acq = jax.jit(acq_map[acq])

    def init(self, ys: jax.Array, params: dict, noise_scale: float = -8.0):
        """
        Initialize the optimizer state from initial data.

        Args:
            ys: Objective values for the initial parameters.
            params: Dict of parameter arrays (same keys as domain).
            noise_scale: Initial noise scale for GP.

        Returns
        -------
            Initialized OptimizerState.
        """
        # Create a padded jax array for each parameter and each score.
        # In order to keep jax compilations at a bay.
        num_entries = len(ys)
        pad_value = int(np.ceil(len(ys) / 10) * 10)

        # Convert to jax arrays if they are not already
        ys = jnp.asarray(ys)
        ys = self.sign * ys
        params = jax.tree.map(lambda x: jnp.asarray(x), params)

        # Define padded arrays for the inputs and the outputs
        mask = jnp.zeros(shape=(pad_value,), dtype=jnp.bool_).at[:num_entries].set(True)
        ys = jnp.zeros(shape=(pad_value,), dtype=ys.dtype).at[:num_entries].set(ys)

        _params = {}
        for key, entries in params.items():
            # Assert that the parameter is in the domain dictionary
            if key not in self.domain:
                raise ValueError(f"Parameter {key} is not in the domain")

            # Get dtype from the domain and create a padded array
            dtype = self.domain[key].dtype
            values = (
                jnp.zeros(shape=(pad_value,), dtype=dtype).at[:num_entries].set(entries)
            )
            _params[key] = values

        # From the given observation, find the better one (either maxima or minima) and return the
        # initial optimizer state.
        best_score = float(jnp.max(ys[mask]))
        best_params_idx = jnp.argmax(ys[mask])
        best_params = jax.tree.map(lambda x: x[mask][best_params_idx], _params)

        # Initialize the gaussian processes state
        gp_params = GPParams(
            noise=jnp.full((1, 1), 1.0 * noise_scale),
            amplitude=jnp.zeros((1, 1)),
            lengthscale=jnp.zeros((1, len(_params))),
        )

        # Fit to the current observations
        xs = jnp.stack(
            [self.domain[key].transform(_params[key]) for key in _params], axis=1
        )
        gp_params = posterior_fit(ys, xs, mask=mask, params=gp_params)

        ys = self.sign * ys
        best_score = self.sign * best_score
        opt_state = OptimizerState(
            params=_params,
            ys=ys,
            best_score=best_score,
            best_params=best_params,
            mask=mask,
            gp_params=gp_params,
        )

        return opt_state

    @partial(jax.jit, static_argnames=("self", "size"))
    def sample(self, key, state, size=10_000):
        """
        Sample new parameters using the acquisition function.

        Args:
            key: JAX PseudoRandom key for random sampling.
            state: Current optimizer state.
            size: Number of samples to draw.

        Returns
        -------
            Sampled parameters (dict).
        """
        # Sample 'size' elements of each distribution.
        samples = self.param_space.sample_params(key, (size,))
        xs_samples = self.param_space.to_array(samples)

        # Prepare the data for the Gaussian process prediction.
        xs = self.param_space.to_array(state.params)
        ys = self.sign * state.ys
        mask = state.mask
        gp_params = state.gp_params

        # Compute the acquisition function values for the sampled points.
        acq_vals = self.acq(xs_samples, xs, ys, mask, gp_params)

        # Of those, find the best 50 points and optimize them using BFGS.
        top_idxs = jnp.argsort(acq_vals)[-50:]
        init_points = xs_samples[top_idxs]

        def acquisition_objective(x):
            return jnp.squeeze(self.acq(x[None, :], xs, ys, mask, gp_params))

        optimized = jax.vmap(
            lambda x: _optimize_suggestion(x, acquisition_objective, max_iter=10)
        )(init_points)
        opt_vals = self.acq(optimized, xs, ys, mask, gp_params)

        # Return the best point from the optimized points and the sampled points.
        all_points = jnp.concatenate((optimized, xs_samples), axis=0)
        all_vals = jnp.concatenate((opt_vals, acq_vals), axis=0)
        chosen_suggestion = jnp.argmax(all_vals)

        best_params = self.param_space.to_dict(all_points[chosen_suggestion][None])
        return best_params

    def expand(self, opt_state: OptimizerState):
        """
        Expand internal buffers if no space is available.

        Args:
            opt_state: Current optimizer state.

        Returns
        -------
            OptimizerState with expanded storage.
        """
        current = jnp.sum(opt_state.mask)

        if current == len(opt_state.mask):
            pad_value = int(np.ceil(len(opt_state.mask) * 2 / 10) * 10)
            diff = pad_value - len(opt_state.mask)
            mask = jnp.pad(opt_state.mask, (0, diff))
            ys = jnp.pad(opt_state.ys, (0, diff))
            params = {}
            for key in opt_state.params:
                params[key] = jnp.pad(opt_state.params[key], (0, diff))
        else:
            mask = opt_state.mask
            ys = opt_state.ys
            params = opt_state.params

        opt_state = OptimizerState(
            params=params,
            ys=ys,
            best_score=opt_state.best_score,
            best_params=opt_state.best_params,
            mask=mask,
            gp_params=opt_state.gp_params,
        )
        return opt_state

    def fit(self, opt_state, y, new_params):
        """
        Update optimizer state with a new observation.

        Args:
            opt_state: Current optimizer state.
            y: New objective value.
            new_params: Parameters that produced y.

        Returns
        -------
            Updated OptimizerState.
        """
        opt_state = self.expand(opt_state)  # Prompts recompilation
        opt_state = self._fit(opt_state, y, new_params)
        return opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _fit(self, opt_state, y, new_params):
        """Update optimizer state with new observation (JIT-compiled)."""
        last_idx = jnp.arange(len(opt_state.mask)) == jnp.argmin(opt_state.mask)
        mask = jnp.asarray(jnp.where(last_idx, True, opt_state.mask))
        ys = jnp.where(last_idx, y, opt_state.ys)
        params = jax.tree_util.tree_map(
            lambda x, y: jnp.where(last_idx, y, x), opt_state.params, new_params
        )

        xs = jnp.stack(
            [self.domain[key].transform(params[key]) for key in params], axis=1
        )
        ys = self.sign * ys
        gp_params = posterior_fit(ys, xs, mask=mask, params=opt_state.gp_params)

        best_score = jnp.max(ys, where=mask, initial=-jnp.inf)
        best_params_idx = jnp.argmax(jnp.where(mask, ys, -jnp.inf))
        best_params = jax.tree_util.tree_map(lambda x: x[best_params_idx], params)

        ys = self.sign * ys
        best_score = self.sign * best_score
        opt_state = OptimizerState(
            params=params,
            ys=ys,
            best_score=best_score,
            best_params=best_params,
            mask=mask,
            gp_params=gp_params,
        )
        return opt_state
