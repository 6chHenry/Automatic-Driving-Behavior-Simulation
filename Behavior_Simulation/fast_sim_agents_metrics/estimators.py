# Copyright (c) 2024 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================
"""Probability estimators for simulation distribution matching."""

import numpy as np
import sklearn.neighbors as sklearn_neighbors
import torch
from torch.distributions import Categorical

from waymo_open_dataset.protos import sim_agents_metrics_pb2


def log_likelihood_estimate_timeseries(
    feature_config: sim_agents_metrics_pb2.SimAgentMetricsConfig.FeatureConfig,
    log_values: torch.Tensor,
    sim_values: torch.Tensor,
) -> torch.Tensor:
  """Computes the log-likelihood estimates for a time-series simulated feature.

  Args:
    feature_config: A time-series compatible `FeatureConfig`.
    log_values: A float Tensor with shape (n_objects, n_steps).
    sim_values: A float Tensor with shape (n_rollouts, n_objects, n_steps).

  Returns:
    A tensor of shape (n_objects, n_steps) containing the log probability
    estimates of the log features under the simulated distribution of the same
    feature.
  """
  if log_values.dim() != 2:
    raise ValueError(f'Log values must be 2D tensor (Actual: {log_values.dim()}D)')
  if sim_values.dim() != 3:
    raise ValueError(f'Sim values must be 3D tensor (Actual: {sim_values.dim()}D)')
  
  n_rollouts, n_objects, n_steps = sim_values.shape
  if log_values.shape != (n_objects, n_steps):
    raise ValueError(f'Log values must be of shape: {(n_objects, n_steps)} '
                     f'(Actual: {log_values.shape})')
  if feature_config.independent_timesteps:
    # If time steps needs to be considered independent, reshape:
    # - `sim_values` as (n_objects, n_rollouts * n_steps)
    # - `log_values` as (n_objects, n_steps)
    sim_values = torch.transpose(sim_values, 0, 1).reshape(n_objects, n_rollouts * n_steps)
  else:
    # If values in time are instead to be compared per-step, reshape:
    # - `sim_values` as (n_objects * n_steps, n_rollouts)
    # - `log_values` as (n_objects * n_steps, 1)
    sim_values = torch.transpose(sim_values, 0, 2).reshape(n_objects * n_steps, n_rollouts)
    log_values = log_values.reshape(n_objects * n_steps, 1)
  
  if feature_config.WhichOneof('estimator') == 'histogram':
    log_likelihood = histogram_estimate(
        feature_config.histogram, log_values, sim_values)
  elif feature_config.WhichOneof('estimator') == 'kernel_density':
    log_likelihood = kernel_density_estimate(
        feature_config.kernel_density, log_values, sim_values)
  elif feature_config.WhichOneof('estimator') == 'bernoulli':
    log_likelihood = bernoulli_estimate(
        feature_config.bernoulli, log_values, sim_values)
  else:
    raise ValueError('`FeatureConfig` contains an invalid estimator. '
                     f'Found: {feature_config.WhichOneof("estimator")}')

  # Depending on `independent_timesteps`, the likelihoods might be flattened, so
  # reshape back to the initial `log_values` shape.
  log_likelihood = log_likelihood.reshape(n_objects, n_steps)
  return log_likelihood


def log_likelihood_estimate_scenario_level(
    feature_config: sim_agents_metrics_pb2.SimAgentMetricsConfig.FeatureConfig,
    log_values: torch.Tensor,
    sim_values: torch.Tensor,
) -> torch.Tensor:
  """Computes the log-likelihood estimates for time-agnostic simulated features.

  Args:
    feature_config: A single-valued compatible `FeatureConfig`.
    log_values: A float Tensor with shape (n_objects,).
    sim_values: A float Tensor with shape (n_rollouts, n_objects).

  Returns:
    A tensor of shape (n_objects,) containing the log-likelihoods estimates
    of the log features under the simulated distribution of the same feature.
  """
  if log_values.dim() != 1:
    raise ValueError(f'Log values must be 1D tensor (Actual: {log_values.dim()}D)')
  if sim_values.dim() != 2:
    raise ValueError(f'Sim values must be 2D tensor (Actual: {sim_values.dim()}D)')
  
  # Reuse `likelihood_estimate_timeseries` by just adding a "dummy" time axis,
  # and removing it once done. The `independent_timesteps` flag doesn't matter
  # here because there is going to be only 1 step anyway.
  timeseries_log_likelihood = log_likelihood_estimate_timeseries(
      feature_config=feature_config,
      # Shape: (n_objects, 1).
      log_values=log_values.unsqueeze(-1),
      # Shape: (n_rollouts, n_objects, 1).
      sim_values=sim_values.unsqueeze(-1))
  # Shape of `timeseries_log_likelihood`: (n_objects, 1).
  return timeseries_log_likelihood[..., 0]


def histogram_estimate(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig.HistogramEstimate,
    log_samples: torch.Tensor,
    sim_samples: torch.Tensor,
) -> torch.Tensor:
  """Computes log-likelihoods of samples based on histograms.

  Args:
    config: A `HistogramEstimate` config.
    log_samples: A float tensor of shape (batch_size, log_sample_size),
      containing `log_sample_size` samples from `batch_size` independent
      populations.
    sim_samples: A float tensor of shape (batch_size, sim_sample_size),
      containing `sim_sample_size` samples from `batch_size` independent
      populations.

  Note: While `batch_size` needs to be consistent, the two samples sizes can be
  different.

  Returns:
    A tensor of shape (batch_size, log_sample_size), where each element (i, k)
    is the log likelihood of the log sample (i, k) under the sim distribution
    (i).
  """
  batch_size = _assert_and_return_batch_size(log_samples, sim_samples)
  # We generate `num_bins`+1 edges for the histogram buckets.
  edges = torch.linspace(config.min_val, config.max_val, config.num_bins+1, dtype=torch.float32)
  # Clip the samples to avoid errors with histograms. Nonetheless, the min/max
  # values should be configured to never hit this condition in practice.
  log_samples = torch.clamp(log_samples, config.min_val, config.max_val).nan_to_num(config.max_val)
  sim_samples = torch.clamp(sim_samples, config.min_val, config.max_val).nan_to_num(config.max_val)

  # Create the categorical distribution for simulation.
  sim_counts = torch.zeros((batch_size, config.num_bins), dtype=torch.float32,device=sim_samples.device)
  for i in range(batch_size):
    sim_counts[i] = torch.histc(sim_samples[i], bins=config.num_bins, min=config.min_val, max=config.max_val)
  sim_counts += config.additive_smoothing_pseudocount
  distribution = Categorical(probs=sim_counts)
  
  # Generate the counts for the log distribution. We reshape the log samples to
  # (batch_size * log_sample_size, 1), so every log sample is independently
  # scored.
  log_values_flat = log_samples.reshape(-1, 1)
  # Shape of log_counts: (batch_size * log_sample_size, num_bins).
  log_counts = torch.zeros((log_values_flat.size(0), config.num_bins), dtype=torch.float32,device=log_samples.device)
  for i in range(log_values_flat.size(0)):
    log_counts[i] = torch.histc(log_values_flat[i], bins=config.num_bins, min=config.min_val, max=config.max_val)
  
  # Identify which bin each sample belongs to and get the log probability of
  # that bin under the sim distribution.
  max_log_bin = torch.argmax(log_counts, dim=1)
  batched_max_log_bin = max_log_bin.reshape(batch_size, -1)
  # Since we have defined the categorical distribution to have `batch_size`
  # independent populations, we need to transpose the log bins to
  # (log_sample_size, batch_size).
  log_likelihood = distribution.log_prob(batched_max_log_bin.t())
  # Transpose back to (batch_size, log_sample_size).
  return log_likelihood.t()


def kernel_density_estimate(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig.KernelDensityEstimate,
    log_samples: torch.Tensor,
    sim_samples: torch.Tensor,
) -> torch.Tensor:
  """Computes log likelihoods of samples based on kernel density estimation.

  Args:
    config: A `KernelDensityEstimate` config.
    log_samples: A float tensor of shape (batch_size, log_sample_size),
      containing `log_sample_size` samples from `batch_size` independent
      populations.
    sim_samples: A float tensor of shape (batch_size, sim_sample_size),
      containing `sim_sample_size` samples from `batch_size` independent
      populations.

  Note: While `batch_size` needs to be consistent, the two samples sizes can be
  different.

  Returns:
    A tensor of shape (batch_size, log_sample_size), where each element (i, k)
    is the log likelihood of the log sample (i, k) under the sim distribution
    (i).
  """
  if config.bandwidth <= 0:
    raise ValueError(
        'Bandwidth needs to be positive for KernelDensity estimation.')
  batch_size = _assert_and_return_batch_size(log_samples, sim_samples)
  scores = []
  for batch_index in range(batch_size):
    kde = sklearn_neighbors.KernelDensity(
        kernel='gaussian', bandwidth=config.bandwidth
    ).fit(sim_samples[batch_index, :, None])
    scores.append(kde.score_samples(log_samples[batch_index][:, None]))
  scores = torch.tensor(scores)
  # When using KDE, we are returned a continuous density estimate, which also
  # means its result is not bounded to the range [0, 1] when exponentiated.
  # We compute the maximum value the KDE estimate can return (i.e. when a
  # Dirac delta is used).
  max_score = 1 / (np.sqrt(2 * np.pi) * config.bandwidth)
  # Next we scale the scores by this `max_score` (subtraction in log space).
  return scores - np.log(max_score)


def bernoulli_estimate(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig.BernoulliEstimate,
    log_samples: torch.Tensor,
    sim_samples: torch.Tensor,
) -> torch.Tensor:
  """Computes log probabilities of samples based on Bernoulli distributions.

  Args:
    config: A `BernoulliEstimate` config.
    log_samples: A boolean tensor of shape (batch_size, log_sample_size),
      containing `log_sample_size` samples from `batch_size` independent
      populations.
    sim_samples: A boolean tensor of shape (batch_size, sim_sample_size),
      containing `sim_sample_size` samples from `batch_size` independent
      populations.

  Note: While `batch_size` needs to be consistent, the two samples sizes can be
  different.

  Returns:
    A tensor of shape (batch_size, log_sample_size), where each element (i, k)
    is the log probability of the log sample (i, k) under the sim distribution
    (i).
  """
  if log_samples.dtype != torch.bool:
    raise ValueError(
        'Tensor `log_samples` must be a boolean tensor for BernoulliEstimate.')
  if sim_samples.dtype != torch.bool:
    raise ValueError(
        'Tensor `sim_samples` must be a boolean tensor for BernoulliEstimate.')
  # The Bernoulli estimate can be computed directly using the histogram estimate
  # above and setting the min and max values accordingly, to create two bins
  # for either 0s or 1s. We also require each of the bins to have size 1,
  # so the normalized density has the correct scaling. This combination gives
  # us 2 bins: [-0.5, 0.5] and [0.5, 1.5] which will correctly contain boolean
  # values.
  histogram_config = (
      sim_agents_metrics_pb2.SimAgentMetricsConfig.HistogramEstimate(
          min_val=-0.5, max_val=1.5, num_bins=2,
          additive_smoothing_pseudocount=config.additive_smoothing_pseudocount
      ))
  # By casting bool Tensors to float32, we are effectively replacing bool
  # values with 0s and 1s, which then will be correctly bucketed by the
  # histogram estimator.
  return histogram_estimate(histogram_config,
                            log_samples.to(torch.float32),
                            sim_samples.to(torch.float32))


def _assert_and_return_batch_size(
    log_samples: torch.Tensor,
    sim_samples: torch.Tensor) -> int:
  """Asserts consistency in the tensor shapes and return batch size.

  Args:
    log_samples: A tensor of shape (batch_size, log_sample_size).
    sim_samples: A tensor of shape (batch_size, sim_sample_size).

  Returns:
    The `batch_size`.
  """
  if log_samples.size(0) != sim_samples.size(0):
    raise ValueError('Log and Sim batch sizes must be equal.')
  return log_samples.size(0)
