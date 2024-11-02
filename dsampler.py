import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
from typing import NamedTuple, Tuple
from utils import temp_tune, fit_dirichlet
from rich import print
from config import ADSConfig, EPS, MIN_TEMP, MAX_TEMP

class ADSState(NamedTuple):
    """State maintained by the Adaptive Dirichlet Sampler"""
    emwa_dir: jnp.ndarray
    emwa_logp: jnp.ndarray
    emwa_temp: jnp.ndarray
    emwa_top_logp: jnp.ndarray

    emwa_ent_scaffold: jnp.ndarray
    emwa_ent_naked: jnp.ndarray
    emwa_std_scaffold: jnp.ndarray
    emwa_std_naked: jnp.ndarray

    token_cross_ent_scaffold: jnp.ndarray
    token_cross_ent_naked: jnp.ndarray
    token_cross_std_scaffold: jnp.ndarray
    token_cross_std_naked: jnp.ndarray
    
    emwa_dir_ent: jnp.ndarray

    emwa_topk_ent_naked: jnp.ndarray

    
  
@jax.jit
def kl_divergence(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """Compute KL divergence between two probability distributions."""
    return jnp.sum(jnp.where(p > 0, p * (jnp.log(p + EPS) - jnp.log(q + EPS)), 0.0), axis=-1)

@jax.jit
def ent_std(logp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute entropy and standard deviation from log probabilities.
    
    Args:
        logp: Log probabilities of shape (..., K)
        
    Returns:
        ent: -∑ᵢ pᵢ log(pᵢ) of shape (...)
        std: √∑ᵢ pᵢ (ent + log(pᵢ))² of shape (...)
    """
    p = jnp.exp(logp)
    ent = -jnp.sum(p * logp, axis=-1)   
    varent = jnp.sum(p * (ent + logp)**2, axis=-1)
    return ent, jnp.sqrt(varent)


@jax.jit
def sample_dirichlet(key: jax.random.PRNGKey, alpha: jnp.ndarray) -> jnp.ndarray:
    """Sample from a Dirichlet distribution."""
    return jax.random.dirichlet(key, alpha)

@jax.jit
def dirichlet_expectation(alpha: jnp.ndarray) -> jnp.ndarray:
    """Compute the expectation E[X|X~Dir(alpha)]"""
    alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
    return alpha / alpha_sum

@jax.jit
def dirichlet_expected_entropy(alpha: jnp.ndarray) -> jnp.ndarray:
    """Compute the expected entropy of a Dirichlet distribution."""
    alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)  # alpha_0
    K = alpha.shape[-1]

    # ln B(alpha) term
    log_beta = jnp.sum(jsp.special.gammaln(alpha), axis=-1) - jsp.special.gammaln(alpha_sum.squeeze())

    # (alpha_0 - K)ψ(alpha_0) term
    digamma_sum = jsp.special.digamma(alpha_sum)
    second_term = (alpha_sum.squeeze() - K) * digamma_sum.squeeze()

    # -sum((alpha_j - 1)ψ(alpha_j)) term
    digamma_alpha = jsp.special.digamma(alpha)
    third_term = -jnp.sum((alpha - 1) * digamma_alpha, axis=-1)

    return log_beta + second_term + third_term

@jax.jit
def dirichlet_log_likelihood_from_logprob(logprobs: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    """Compute log probability of probs under Dirichlet(alpha)."""
    return jnp.sum((alpha - 1.0) * logprobs, axis=-1) - jsp.special.gammaln(jnp.sum(alpha, axis=-1)) + jnp.sum(jsp.special.gammaln(alpha), axis=-1)

@jax.jit
def dirichlet_expected_varentropy(alpha: jnp.ndarray) -> jnp.ndarray:
    """Compute the expected varentropy E[∑ᵢ ln(Xᵢ)² * Xᵢ] of a Dirichlet distribution.
    
    Args:
        alpha: Dirichlet parameters of shape (..., K)
        
    Returns:
        Expected varentropy of shape (...)
    """
    alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)  # α₀
    
    # E[Xᵢ] = αᵢ/α₀ 
    expected_x = alpha / alpha_sum
    
    # ψ(αᵢ)² + ψ₁(αᵢ) term
    digamma_alpha = jsp.special.digamma(alpha)
    trigamma_alpha = jsp.special.polygamma(1, alpha)
    squared_plus_deriv = digamma_alpha**2 + trigamma_alpha
    
    # Sum over dimensions: ∑ᵢ (αᵢ/α₀) * (ψ₁(αᵢ) + ψ(αᵢ)²)
    return jnp.sum(expected_x * squared_plus_deriv, axis=-1)


@jax.jit
def update_emwa_logp(
    raw_logp: jnp.ndarray,
    emwa_logp: jnp.ndarray,
    config: ADSConfig
) -> jnp.ndarray:
    """Update exponential moving average of log probabilities."""
    # Add numerical stability to softmax inputs
    raw_logp = raw_logp - jnp.max(raw_logp, axis=-1, keepdims=True)
    emwa_logp = emwa_logp - jnp.max(emwa_logp, axis=-1, keepdims=True)

    kl = kl_divergence(jax.nn.softmax(raw_logp), jax.nn.softmax(emwa_logp))
    # Add safety clipping to prevent extreme coefficients
    coeff = jnp.clip(
        config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS)),
        1e-6,
        0.999
    )
    coeff = coeff[..., None]  # Shape (batch_size, 1)
    coeff = coeff.astype(raw_logp.dtype)  # Ensure dtype consistency
    return coeff * raw_logp + (1 - coeff) * emwa_logp

def initialize_state(bsz: int, vsz: int, config: ADSConfig, dtype=jnp.bfloat16) -> ADSState:
    """Initialize the ADSState with specified dtype."""
    state = ADSState(
        emwa_dir=jnp.ones((bsz, vsz), dtype=dtype),
        emwa_logp=jnp.zeros((bsz, vsz), dtype=dtype),
        emwa_temp=jnp.ones((bsz,), dtype=dtype),
        token_cross_ent_scaffold=jnp.zeros((bsz,), dtype=dtype),
        token_cross_ent_naked=jnp.zeros((bsz,), dtype=dtype),
        emwa_dir_ent=jnp.zeros((bsz,), dtype=dtype),
        emwa_ent_scaffold=jnp.zeros((bsz,), dtype=dtype),
        entropy_rate_naked=jnp.zeros((bsz,), dtype=dtype)
    )
    return state

@jax.jit
def adaptive_dirichlet_step(
    key: jax.random.PRNGKey,
    state: ADSState,
    logits: jnp.ndarray,
    config: ADSConfig
) -> Tuple[ADSState, jnp.ndarray]:
    """Single step of the Adaptive Dirichlet Sampler."""
    dtype = logits.dtype
    bsz, _ = logits.shape
    output_tokens = jnp.zeros(bsz, dtype=jnp.int32)
    # Constants cast to dtype
    MIN_TEMP = jnp.array(1e-4, dtype=dtype)
    MAX_TEMP = jnp.array(1e4, dtype=dtype)
    EPS = jnp.array(1e-8, dtype=dtype)
    
    # normalize logits
    naked_log_probs = logits - jnp.max(logits, axis=-1, keepdims=True)
    naked_log_probs = naked_log_probs - jax.nn.logsumexp(naked_log_probs, axis=-1, keepdims=True)
    
    # update naked entropy rate
    naked_ent, naked_std = ent_std(naked_log_probs)
    
    emwa_ent_naked = (
        config.emwa_ent_naked_coeff * naked_ent  +
        (1 - config.emwa_ent_naked_coeff) * state.emwa_ent_naked
    )
    emwa_std_naked = (
        config.emwa_std_naked_coeff * (naked_std) +
        (1 - config.emwa_std_naked_coeff) * state.emwa_std_naked
    )
    # entropy and varentropy vectors - shape (bsz, 4)
    state_ent = jnp.array([
        state.token_cross_ent_scaffold,
        state.token_cross_ent_naked,
        state.emwa_ent_scaffold,
        state.emwa_ent_naked
    ]) # TODO: add dirichlet expected entropy...
    state_std = jnp.sqrt(jnp.array([
        state.token_cross_std_scaffold,
        state.token_cross_std_naked,
        state.emwa_std_scaffold,
        state.emwa_std_naked
    ])) # TODO: add dirichlet expected std...

    # First compute outlier threshold from state variables - shape (bsz,)
    outlier_threshold = (
        jnp.einsum('bi,ij,bj->b', state_ent, config.outlier_threshold.bilinear, state_std) +
        jnp.einsum('bi,i->b', state_ent, config.outlier_threshold.linear_state_ent) +
        jnp.einsum('bi,i->b', state_std, config.outlier_threshold.linear_state_std) +
        naked_ent * config.outlier_threshold.linear_naked_ent +
        naked_std * config.outlier_threshold.linear_naked_std +
        config.outlier_threshold.bias
    )
    outlier_mask = outlier_threshold > 0
    # extract topk
    topk_logprobs, topk_indices = jax.lax.top_k(naked_log_probs, config.token_outlier_k)
    # update emwa topk entropy
    norm_topk_logprobs = topk_logprobs - jnp.max(topk_logprobs, axis=-1, keepdims=True)
    norm_topk_logprobs = norm_topk_logprobs - jax.nn.logsumexp(norm_topk_logprobs, axis=-1, keepdims=True)
    naked_topk_ent, _ = ent_std(norm_topk_logprobs)
    new_emwa_topk_ent_naked = config.emwa_topk_ent_naked_coeff * naked_topk_ent + (1 - config.emwa_topk_ent_naked_coeff) * state.emwa_topk_ent_naked
    """
    argmax policy for concentrated inliers
    """
    argmax_threshold = config.argmax_threshold.weight * state.emwa_topk_ent_naked + config.argmax_threshold.bias
    argmax_mask = ~outlier_mask & (naked_topk_ent < argmax_threshold)
    # Set output tokens to argmax for concentrated inliers
    argmax_indices = jnp.argmax(topk_logprobs[argmax_mask], axis=-1)
    argmax_tokens = topk_indices[argmax_mask, argmax_indices]
    output_tokens = output_tokens.at[argmax_mask].set(argmax_tokens)
    """
    topk temperature tuning policy for dispersed inliers
    """
    inlier_sampling_indices = ~outlier_mask & ~argmax_mask
    # Handle less confident inliers by sampling with entropy-tuned temperature
    inlier_sampling_temp = temp_tune(topk_logprobs[inlier_sampling_indices], state.emwa_topk_ent_naked)
    sampling_inlier_choices = jax.random.categorical(key, topk_logprobs[inlier_sampling_indices] / inlier_sampling_temp)
    sampling_inlier_tokens = topk_indices[inlier_sampling_indices, sampling_inlier_choices]
    output_tokens = output_tokens.at[inlier_sampling_indices].set(sampling_inlier_tokens)
    """
    target entropy = affine function of state_ent and inverse emwa temperature
    """
    # outlier target entropy is affine function of state_ent and inverse emwa temperature
    target_entropy = (
        jnp.dot(config.target_entropy.linear, state_ent) +
        jnp.sum(config.target_entropy.linear_inv_temp / state.emwa_temp, axis=-1) +
        config.target_entropy.bias
    )
    temp, _, _ = temp_tune(naked_log_probs.astype(jnp.float32), target_entropy)
    # update emwa temperature
    emwa_temp = config.emwa_temp_coeff * temp + (1 - config.emwa_temp_coeff) * state.emwa_temp
    """
    scale log probabilities and update emwa logp on dirichlet support
    """
    # scale log probabilities
    tuned_logits = naked_log_probs / jnp.clip(emwa_temp[:, None], MIN_TEMP, MAX_TEMP)
    tuned_logp = tuned_logits - jax.nn.logsumexp(tuned_logits, axis=-1, keepdims=True)
    # update emwa logp on dirichlet support
    tuned_logits_on_supp = tuned_logp[:, config.dirichlet_support]
    tuned_logp_on_supp = tuned_logits_on_supp - jax.nn.logsumexp(tuned_logits_on_supp, axis=-1, keepdims=True)
    kl = kl_divergence(jax.nn.softmax(tuned_logp_on_supp), jax.nn.softmax(state.emwa_logp))
    emwa_logp_coeff = config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS))
    emwa_logp = emwa_logp_coeff * tuned_logp_on_supp + (1 - emwa_logp_coeff) * state.emwa_logp
    """
    update emwa logp and dirichlet parameters
    """    
    # update Dirichlet parameters
    new_dir_params, _, _ = fit_dirichlet(emwa_logp)
    new_emwa_dir = config.emwa_dir_coeff * new_dir_params + (1 - config.emwa_dir_coeff) * state.emwa_dir
    """
    update Dirichlet entropy
    """
    dir_log_likelihood = dirichlet_log_likelihood_from_logprob(tuned_logp_on_supp, state.emwa_dir)
    new_emwa_dir_ent = (
        config.emwa_dir_ent_coeff * (-dir_log_likelihood) +
        (1 - config.emwa_dir_ent_coeff) * state.emwa_dir_ent
    )
    dirichlet_threshold = config.dirichlet_threshold.weight * state.emwa_dir_ent + config.dirichlet_threshold.bias
    use_dirichlet = outlier_mask & (-dir_log_likelihood < dirichlet_threshold)
    """
    below dirichlet threshold, interpolate and sample (can improve this in the future)
    """
    # compute perturbation coefficient
    dir_expectation = dirichlet_expectation(state.emwa_dir)
    kl_div = dirichlet_expected_entropy(state.emwa_dir) - jnp.sum(dir_expectation * tuned_logp_on_supp, axis=-1) 
    perturb_coeff = 1 - jnp.pow(config.perturb_base_coeff, - config.perturb_exp_coeff * (1 / (kl_div + EPS)))
    # interpolate
    interpolated_probs = jnp.exp(tuned_logp, axis=-1)
    interpolated_probs = interpolated_probs.at[use_dirichlet, config.dirichlet_support].set(
        perturb_coeff[:, None] * dir_expectation + (1 - perturb_coeff[:, None]) * interpolated_probs[use_dirichlet, config.dirichlet_support]
    )
    # sample token ids
    output_tokens = output_tokens.at[use_dirichlet].set(jax.random.categorical(key, jnp.log(interpolated_probs))) # shape (bsz,)
    """
    above dirichlet threshold youre ngmi
    """
    # TODO: BEAM SEARCH (in the mean time we just sample from tuned_probs)
    output_tokens = output_tokens.at[outlier_mask & ~use_dirichlet].set(jax.random.categorical(key, tuned_logits)) # shape (bsz,)
    # update entropy rates
    scaffold_ent, _ =  ent_std(jnp.log(interpolated_probs + EPS))
    emwa_ent_scaffold = (
        config.emwa_ent_scaffold_coeff * scaffold_ent +
        (1 - config.emwa_ent_scaffold_coeff) * state.emwa_ent_scaffold
    )

    # Update token cross entropies
    batch_indices = jnp.arange(bsz)
    scaffold_token_logprob = jnp.log(interpolated_probs[batch_indices, output_tokens] + EPS)
    naked_token_logprob = jnp.log(naked_log_probs[batch_indices, output_tokens] + EPS)
    
    token_cross_ent_scaffold = (
        config.token_cross_ent_scaffold_coeff * (-scaffold_token_logprob) +
        (1 - config.token_cross_ent_scaffold_coeff) * state.token_cross_ent_scaffold
    )
    token_cross_ent_naked = (
        config.token_cross_ent_naked_coeff * (-naked_token_logprob) +
        (1 - config.token_cross_ent_naked_coeff) * state.token_cross_ent_naked
    )
    token_cross_std_naked = (
        config.token_cross_std_naked_coeff * (token_cross_ent_naked - naked_token_logprob) ** 2 +
        (1 - config.token_cross_std_naked_coeff) * state.token_cross_std_naked
    )
    token_cross_std_scaffold = (
        config.token_cross_std_scaffold_coeff * (token_cross_ent_scaffold - scaffold_token_logprob) ** 2 +
        (1 - config.token_cross_std_scaffold_coeff) * state.token_cross_std_scaffold
    )
    # assemble new state
    new_state = ADSState(
        emwa_dir=new_emwa_dir,
        emwa_logp=emwa_logp,
        emwa_temp=emwa_temp,
        token_cross_ent_scaffold=token_cross_ent_scaffold,
        token_cross_ent_naked=token_cross_ent_naked,
        token_cross_std_naked=token_cross_std_naked,
        token_cross_std_scaffold=token_cross_std_scaffold,
        emwa_dir_ent=new_emwa_dir_ent,
        emwa_ent_scaffold=emwa_ent_scaffold,
        emwa_ent_naked=emwa_ent_naked,
        emwa_std_naked=emwa_std_naked,
        emwa_topk_ent_naked=new_emwa_topk_ent_naked,
    )

    return new_state, output_tokens

