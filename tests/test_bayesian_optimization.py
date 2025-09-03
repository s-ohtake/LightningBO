"""Tests for Bayesian optimization functionality."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import LightningBO.bayesian_core as bc
import LightningBO.bayesian_optimization as bo

KEY = jax.random.PRNGKey(42)
SEED = 42

class BayesianOptimizationTest(absltest.TestCase):
	"""Test suite for Bayesian optimization functionality."""

	def test_1D_optim(self):
		"""Test 1D optimization with sin function."""
		def f(x):
			return jnp.sin(x) - 0.1 * x**2

		TARGET = np.max(f(np.linspace(-2, 2, 1000)))
		domain = {"x": bc.Continuous(-2, 2)}
		opt = bo.Optimizer(domain=domain, maximize=True)
		params = {"x": [0.0, 1.0, 2.0]}
		ys = [f(x) for x in params["x"]]
		opt_state = opt.init(ys, params)
		self.assertEqual(opt_state.best_score, np.max(ys))
		self.assertEqual(opt_state.best_params["x"], params["x"][np.argmax(ys)])
		self.assertEqual(type(opt_state), bo.OptimizerState)
		self.assertEqual(type(opt_state.gp_params), bc.GPParams)
		self.assertEqual(opt_state.params["x"].shape, (10,))
		self.assertEqual(opt_state.ys.shape, (10,))
		self.assertEqual(opt_state.mask.shape, (10,))
		self.assertTrue(np.allclose(opt_state.params["x"][:3], params["x"]))
		self.assertTrue(np.allclose(opt_state.ys[:3], ys))
		self.assertTrue(np.allclose(opt_state.mask[:3], [True, True, True]))
		self.assertTrue(np.allclose(opt_state.mask[3:], [False] * 7))
		key = jax.random.PRNGKey(SEED)
		sample_fn = jax.jit(opt.sample)
		for step in range(100):
			key = jax.random.fold_in(key, step)
			new_params = sample_fn(key, opt_state)
			y = f(**new_params)
			opt_state = opt.fit(opt_state, y, new_params)
			if jnp.allclose(opt_state.best_score, TARGET, atol=1e-03):
				break
# ...他のテスト...
