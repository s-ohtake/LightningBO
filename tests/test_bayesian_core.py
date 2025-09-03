"""Tests for the Bayesian core functionality including GP operations and acquisition functions."""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from LightningBO.bayesian_core import (
	GPParams,
	Integer,
	Continuous,
	cov,
	exp_quadratic,
	expected_improvement,
	gaussian_process,
	lower_confidence_bounds,
	marginal_likelihood,
	predict,
	probability_improvement,
	upper_confidence_bounds,
)

class BayesianCoreDomainTest(parameterized.TestCase):
	"""Test suite for Bayesian domain classes (Continuous, Integer)."""

	@parameterized.parameters(
		(0.0, 1.0),
		(-5.5, 5.5),
	)
	def test_continuous_transform_clips_within_bounds(self, lower, upper):
		"""Test that Continuous domain clips values within bounds."""
		domain = Continuous(lower, upper)
		x = jnp.array([lower - 1, (lower + upper) / 2, upper + 1])
		out = domain.transform(x)
		self.assertTrue(jnp.all(out >= lower) and jnp.all(out <= upper))

	def test_continuous_sample_within_bounds(self):
		"""Test that Continuous domain samples are within bounds."""
		domain = Continuous(0.0, 1.0)
		key = jax.random.PRNGKey(0)
		shape = (100,)
		samples = domain.sample(key, shape)
		self.assertEqual(samples.shape, shape)
		self.assertTrue(jnp.all(samples >= 0.0) and jnp.all(samples <= 1.0))

	@parameterized.parameters(
		(0, 5),
	)
	def test_integer_transform_clips_within_bounds(self, lower, upper):
		domain = Integer(lower, upper)
		x = jnp.array([lower - 1, lower, upper, upper + 1])
		out = domain.transform(x)
		self.assertTrue(jnp.all(out >= lower) and jnp.all(out <= upper))

	def test_integer_sample_within_bounds(self):
		domain = Integer(0, 5)
		key = jax.random.PRNGKey(0)
		shape = (100,)
		samples = domain.sample(key, shape)
		self.assertEqual(samples.shape, shape)
		self.assertTrue(jnp.all(samples >= 0) and jnp.all(samples <= 5))

# ...他のテスト...
