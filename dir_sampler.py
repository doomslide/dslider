import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
from typing import NamedTuple, Tuple
from utils import temp_tune, fit_dirichlet
from rich import print
from dataclasses import dataclass
# Constants
MIN_TEMP = 1e-4
MAX_TEMP = 1e4
EPS = 1e-8

class ADSState(NamedTuple):
    """State maintained by the Adaptive Dirichlet Sampler"""
    emwa_dir: jnp.ndarray
    emwa_logp: jnp.ndarray
    emwa_temp: jnp.ndarray
    token_cross_ent_scaffold: jnp.ndarray
    token_cross_ent_naked: jnp.ndarray
    emwa_dir_ent: jnp.ndarray
    entropy_rate_scaffold: jnp.ndarray
    entropy_rate_naked: jnp.ndarray
    

@dataclass(frozen=True)
class ADSConfig:
    emwa_logp_base: float
    emwa_logp_exp_factor: float
    emwa_dir_coeff: float
    emwa_temp_coeff: float
    emwa_dir_ent_coeff: float
    entropy_rate_scaffold_coeff: float
    entropy_rate_naked_coeff: float
    token_cross_ent_scaffold_coeff: float
    perturb_base_coeff: float
    perturb_exp_coeff: float
    probs_ent_offset: float
    dir_ent_offset: float
    entropy_a: float
    entropy_b: float
    entropy_c: float
    entropy_d: float
    dirichlet_d: float
    dirichlet_e: float
    token_cross_ent_naked_coeff: float

@jax.jit
def kl_divergence(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """Compute KL divergence between two probability distributions."""
    return jnp.sum(jnp.where(p > 0, p * (jnp.log(p + EPS) - jnp.log(q + EPS)), 0.0), axis=-1)

@jax.jit
def entropy(p: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy of a probability distribution."""
    return -jnp.sum(jnp.where(p > 0, p * jnp.log(p + EPS), 0.0), axis=-1)

@jax.jit
def sample_dirichlet(key: jax.random.PRNGKey, alpha: jnp.ndarray) -> jnp.ndarray:
    """Sample from a Dirichlet distribution."""
    return jax.random.dirichlet(key, alpha)

@jax.jit
def dirichlet_expectation(alpha: jnp.ndarray) -> jnp.ndarray:
    """Compute the expectation E[X|X~Dir(alpha)]"""
    alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
    return alpha / alpha_sum

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

@partial(jax.jit, static_argnames=('config',))
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

def initialize_state(bsz: int, vsz: int, dtype=jnp.bfloat16) -> ADSState:
    """Initialize the ADSState with specified dtype."""
    state = ADSState(
        emwa_dir=jnp.ones((bsz, vsz), dtype=dtype),
        emwa_logp=jnp.zeros((bsz, vsz), dtype=dtype),
        emwa_temp=jnp.ones((bsz,), dtype=dtype),
        token_cross_ent_scaffold=jnp.zeros((bsz,), dtype=dtype),
        token_cross_ent_naked=jnp.zeros((bsz,), dtype=dtype),
        emwa_dir_ent=jnp.zeros((bsz,), dtype=dtype),
        entropy_rate_scaffold=jnp.zeros((bsz,), dtype=dtype),
        entropy_rate_naked=jnp.zeros((bsz,), dtype=dtype)
    )
    return state

@partial(jax.jit, static_argnums=(3,)) 
def adaptive_dirichlet_step(
    key: jax.random.PRNGKey,
    state: ADSState,
    logits: jnp.ndarray,
    config: ADSConfig
) -> Tuple[ADSState, jnp.ndarray]:
    """Single step of the Adaptive Dirichlet Sampler."""
    dtype = logits.dtype  # Ensure consistency in data types
    bsz, vsz = logits.shape

    # Constants cast to dtype
    MIN_TEMP = jnp.array(1e-4, dtype=dtype)
    MAX_TEMP = jnp.array(1e4, dtype=dtype)
    EPS = jnp.array(1e-8, dtype=dtype)

    # 1. Normalize logits to log probabilities
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    log_probs = logits - logits_max
    log_probs = log_probs - jax.nn.logsumexp(log_probs, axis=-1, keepdims=True)

    # 2. Update exponential moving average of log probabilities
    kl = kl_divergence(jax.nn.softmax(log_probs), jax.nn.softmax(state.emwa_logp))
    coeff = config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS))
    coeff = coeff.astype(dtype)  # Cast to dtype
    emwa_logp = coeff * log_probs + (1 - coeff) * state.emwa_logp

    # 3. Fit Dirichlet parameters
    dir_params, _, _ = fit_dirichlet(emwa_logp.astype(jnp.float32))
    dir_params = dir_params.astype(dtype)
    emwa_dir = config.emwa_dir_coeff * dir_params + (1 - config.emwa_dir_coeff) * state.emwa_dir

    # 4. Calculate target entropy and temperature
    target_entropy = (
        config.entropy_a * state.token_cross_ent_scaffold +
        config.entropy_b * state.token_cross_ent_naked +
        config.entropy_c * state.entropy_rate_scaffold +
        config.entropy_d * state.entropy_rate_naked +
        config.probs_ent_offset
    ).astype(jnp.float32)  # Cast to float32 for temp_tune
    temp, _, _ = temp_tune(log_probs.astype(jnp.float32), target_entropy)
    temp = temp.astype(dtype)
    emwa_temp = config.emwa_temp_coeff * temp + (1 - config.emwa_temp_coeff) * state.emwa_temp

    # 5. Scale log probabilities
    scaled_log_probs = log_probs / jnp.clip(emwa_temp[:, None], MIN_TEMP, MAX_TEMP)
    scaled_log_probs = scaled_log_probs - jax.nn.logsumexp(scaled_log_probs, axis=-1, keepdims=True)
    naked_probs = jnp.exp(scaled_log_probs)

    # 6. Update entropy rates
    entropy_naked = entropy(naked_probs)
    entropy_rate_naked = (
        config.entropy_rate_naked_coeff * entropy_naked +
        (1 - config.entropy_rate_naked_coeff) * state.entropy_rate_naked
    )

    # 7. Update Dirichlet entropy
    dir_log_likelihood = dirichlet_log_likelihood_from_logprob(scaled_log_probs, state.emwa_dir)
    baseline_log_likelihood = dirichlet_log_likelihood_from_logprob(emwa_logp, state.emwa_dir)
    emwa_dir_ent = (
        config.emwa_dir_ent_coeff * (-dir_log_likelihood) +
        (1 - config.emwa_dir_ent_coeff) * state.emwa_dir_ent
    )

    # 8. Determine sampling strategy
    dir_threshold = (
        config.dirichlet_d * (-baseline_log_likelihood) +
        config.dirichlet_e * state.emwa_dir_ent +
        config.dir_ent_offset
    )
    use_dirichlet = (-dir_log_likelihood < dir_threshold)[..., None]

    # 9. Compute perturbation coefficient
    dir_expectation = dirichlet_expectation(state.emwa_dir)
    kl_div = kl_divergence(dir_expectation, naked_probs)
    exponent = -config.perturb_exp_coeff / (kl_div + EPS)
    perturb_coeff = 1 - config.perturb_base_coeff ** exponent
    perturb_coeff = perturb_coeff.astype(dtype)

    # 10. Combine probabilities
    final_probs = jnp.where(
        use_dirichlet,
        perturb_coeff * dir_expectation + (1 - perturb_coeff) * naked_probs,
        naked_probs
    )

    # 11. Sample token ids
    token_ids = jax.random.categorical(key, jnp.log(final_probs).astype(jnp.float32))

    # 12. Update entropy rates
    entropy_scaffold = entropy(final_probs)
    entropy_rate_scaffold = (
        config.entropy_rate_scaffold_coeff * entropy_scaffold +
        (1 - config.entropy_rate_scaffold_coeff) * state.entropy_rate_scaffold
    )

    # 13. Update token cross entropies
    batch_indices = jnp.arange(bsz)
    final_token_logprob = jnp.log(final_probs[batch_indices, token_ids] + EPS)
    naked_token_logprob = scaled_log_probs[batch_indices, token_ids]

    token_cross_ent_scaffold = (
        config.token_cross_ent_scaffold_coeff * (-final_token_logprob) +
        (1 - config.token_cross_ent_scaffold_coeff) * state.token_cross_ent_scaffold
    )
    token_cross_ent_naked = (
        config.token_cross_ent_naked_coeff * (-naked_token_logprob) +
        (1 - config.token_cross_ent_naked_coeff) * state.token_cross_ent_naked
    )

    # 14. Assemble new state
    new_state = ADSState(
        emwa_dir=emwa_dir,
        emwa_logp=emwa_logp,
        emwa_temp=emwa_temp,
        token_cross_ent_scaffold=token_cross_ent_scaffold,
        token_cross_ent_naked=token_cross_ent_naked,
        emwa_dir_ent=emwa_dir_ent,
        entropy_rate_scaffold=entropy_rate_scaffold,
        entropy_rate_naked=entropy_rate_naked
    )

    return new_state, token_ids

