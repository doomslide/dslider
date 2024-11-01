import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
from typing import NamedTuple, Tuple, Optional
from utils import temp_tune, fit_dirichlet


HF_TOKEN = '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'
CACHE_DIR = '/home/cloudforest/Weights'
jax.config.update('jax_platform_name', 'gpu')

MIN_TEMP = 1e-4
MAX_TEMP = 1e4
EPS = 1e-8

class ADSState(NamedTuple):
    """State maintained by the Adaptive Dirichlet Sampler"""
    emwa_dir: jnp.ndarray  # Dirichlet parameters
    emwa_logp: jnp.ndarray  # Log probabilities
    emwa_temp: jnp.ndarray  # Temperature
    token_cross_ent_scaffold: jnp.ndarray  # Scaffold cross entropy rate
    token_cross_ent_naked: jnp.ndarray # Naked cross entropy rate
    emwa_dir_ent: jnp.ndarray  # Dirichlet entropy rate
    entropy_rate_scaffold: jnp.ndarray  # Scaffold entropy rate
    entropy_rate_naked: jnp.ndarray # Naked entropy rate

class ADSConfig(NamedTuple):
    """Configuration for Adaptive Dirichlet Sampler"""
    emwa_logp_base: float = 0.99
    emwa_logp_exp_factor: float = 1.0
    emwa_dir_coeff: float = 0.99
    emwa_temp_coeff: float = 0.99
    emwa_dir_ent_coeff: float = 0.99
    entropy_rate_scaffold_coeff: float = 0.99
    entropy_rate_naked_coeff: float = 0.99
    token_cross_ent_scaffold_coeff: float = 0.99
    perturb_base_coeff: float = 0.99
    perturb_exp_coeff: float = 1.0
    probs_ent_offset: float = 0.1
    dir_ent_offset: float = 0.1
    entropy_a: float = 0.5
    entropy_b: float = 0.3
    entropy_c: float = 0.4
    entropy_d: float = 0.2
    dirichlet_d: float = 0.5
    dirichlet_e: float = 0.3
    token_cross_ent_naked_coeff: float = 0.99 

@jax.jit
def kl_divergence(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """Compute KL divergence between two probability distributions."""
    return jnp.sum(jnp.where(p > 0, p * (jnp.log(p) - jnp.log(q)), 0.0))

@jax.jit
def entropy(p: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy of a probability distribution."""
    return -jnp.sum(jnp.where(p > 0, p * jnp.log(p), 0.0))

@jax.jit
def sample_dirichlet(key: jax.random.PRNGKey, alpha: jnp.ndarray) -> jnp.ndarray:
    """Sample from a Dirichlet distribution."""
    return jax.random.dirichlet(key, alpha)

@jax.jit
def dirichlet_expectation(alpha: jnp.ndarray) -> jnp.ndarray:
    """Compute the expectation E[X|X~Dir(alpha)]"""
    # For Dir(alpha), E[X_i] = alpha_i / sum(alpha)
    alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
    return alpha / alpha_sum

def dirichlet_expected_entropy(alpha: jnp.ndarray) -> jnp.ndarray:
    """Compute the expected entropy of a Dirichlet distribution.
    
    For a Dirichlet distribution with parameter alpha, this computes:
    E[-sum(p_i * ln(p_i))] where p ~ Dir(alpha)
    
    The formula is:
    h(X) = ln B(alpha) + (alpha_0 - K)ψ(alpha_0) - sum((alpha_j - 1)ψ(alpha_j))
    where B is the beta function, ψ is the digamma function,
    alpha_0 is sum(alpha), and K is the dimension of alpha.
    
    Args:
        alpha: Parameter vector of the Dirichlet distribution
        
    Returns:
        Expected entropy of the distribution
    """
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
    return jnp.sum((alpha - 1.0) * logprobs) - jsp.special.gammaln(jnp.sum(alpha)) + jnp.sum(jsp.special.gammaln(alpha))

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
        1e-3,
        0.999
    )
    coeff = coeff[..., None]  # Shape (batch_size, 1)
    return coeff * raw_logp + (1 - coeff) * emwa_logp

@partial(jax.jit, static_argnums=(0,1))
def initialize_state(bsz: int, vsz: int, config: ADSConfig) -> ADSState:
    """Initialize the state for the Adaptive Dirichlet Sampler."""
    return ADSState(
        emwa_dir=jnp.ones((bsz, vsz)),  # [batch_size, vsz]
        emwa_logp=jnp.zeros((bsz, vsz)),  # [batch_size, vsz]
        emwa_temp=jnp.ones((bsz,)),  # [batch_size,]
        token_cross_ent_scaffold=jnp.zeros((bsz,)),  # [batch_size,]
        token_cross_ent_naked=jnp.zeros((bsz,)),  # [batch_size,] 
        emwa_dir_ent=jnp.zeros((bsz,)),  # [batch_size,]
        entropy_rate_scaffold=jnp.zeros((bsz,)),  # [batch_size,]
        entropy_rate_naked=jnp.zeros((bsz,))
    )

@partial(jax.jit, static_argnames=('config',))
def adaptive_dirichlet_step(
    key: jax.random.PRNGKey,
    state: ADSState,
    logits: jnp.ndarray,
    config: ADSConfig,
) -> Tuple[ADSState, jnp.ndarray]:
    """Single step of the Adaptive Dirichlet Sampler."""
    bsz, vsz = logits.shape 

    # Split PRNG key
    key1, key2 = jax.random.split(key)
    
    # 1. Convert logits to log probabilities
    raw_logp = jax.nn.log_softmax(logits)
    
    # 2. Find temperature (target entropy should be scalar)
    target_entropy = (
        config.entropy_a * state.token_cross_ent_scaffold + 
        config.entropy_b * state.token_cross_ent_naked + 
        config.entropy_c * state.entropy_rate_scaffold +
        config.entropy_d * state.entropy_rate_naked +
        config.probs_ent_offset
    )
    pre_temp, _, _ = temp_tune(raw_logp, target_entropy)
    
    # 3. Update EMWA temperature
    new_emwa_temp = config.emwa_temp_coeff * pre_temp + (1 - config.emwa_temp_coeff) * state.emwa_temp
    
    # 4. Compute naked_probs and update naked entropy rate and emwa of log probs
    naked_probs = jax.nn.softmax(raw_logp / jnp.clip(pre_temp[..., None], MIN_TEMP, MAX_TEMP))
    naked_logprobs = jax.nn.log_softmax(raw_logp / jnp.clip(pre_temp[..., None], MIN_TEMP, MAX_TEMP))
    new_emwa_logp = update_emwa_logp(naked_logprobs, state.emwa_logp, config)
    new_entropy_naked = jnp.sum(- naked_probs * naked_logprobs, axis=-1)
    new_entropy_rate_naked = (
        config.entropy_rate_naked_coeff * new_entropy_naked +
        (1 - config.entropy_rate_naked_coeff) * state.entropy_rate_naked
    )

    # 5. Update EMWA of Dirichlet parameters
    new_dir_params, _, _ = fit_dirichlet(new_emwa_logp)
    new_emwa_dir = config.emwa_dir_coeff * new_dir_params + (1 - config.emwa_dir_coeff) * state.emwa_dir
    
    # 6. Update EMWA Dirichlet entropy
    baseline_log_likelihood = dirichlet_log_likelihood_from_logprob(
        new_emwa_logp,
        new_emwa_dir
    )
    dir_log_likelihood = dirichlet_log_likelihood_from_logprob(naked_logprobs, state.emwa_dir)
    new_emwa_dir_ent = (
        config.emwa_dir_ent_coeff * (-dir_log_likelihood) + 
        (1 - config.emwa_dir_ent_coeff) * state.emwa_dir_ent
    )

    # logprob level

    # 7. Determine whether to use Dirichlet
    dir_threshold = (
        config.dirichlet_d * (-baseline_log_likelihood) +
        config.dirichlet_e * state.emwa_dir_ent +
        config.dir_ent_offset
    )
    use_dirichlet = (-dir_log_likelihood < dir_threshold)[..., None] 
    
    # 8. Interpolate if using Dirichlet
    dirichlet_expected_probs = dirichlet_expectation(state.emwa_dir)
    perturb_coeff = 1 - config.perturb_base_coeff ** (-config.perturb_exp_coeff / (kl_divergence(dirichlet_expected_probs, naked_probs) + EPS))
    final_probs = jnp.where(use_dirichlet, perturb_coeff * dirichlet_expected_probs + (1 - perturb_coeff) * naked_probs, naked_probs)
    
    # 9. Sample token and update entropy
    token_ids = jax.random.categorical(key2, jnp.log(final_probs))  # Shape: (batch_size,)
    entropy_rate = entropy(final_probs)  # Shape: (batch_size,)
    new_entropy_rate_scaffold = (
        config.entropy_rate_scaffold_coeff * entropy_rate + 
        (1 - config.entropy_rate_scaffold_coeff) * state.entropy_rate_scaffold
    )
    
    # 10. Compute token log likelihood
    batch_indices = jnp.arange(bsz)
    final_token_logprob = jnp.log(final_probs[batch_indices, token_ids])  # Shape: (batch_size,)
    naked_token_logprob = naked_logprobs[batch_indices, token_ids]  # Shape: (batch_size,)

    # token level:
        # TODO: compare (-naked_token_logprob) vs naked_token_cross_ent, (-scaffold_token_logprob) vs scaffold_token_cross_ent
        # treat sensibly each of the 4 cases.

    new_token_cross_ent_scaffold = (
        config.token_cross_ent_scaffold_coeff * (-final_token_logprob) + 
        (1 - config.token_cross_ent_scaffold_coeff) * state.token_cross_ent_scaffold
    )
    new_token_cross_ent_naked = (
        config.token_cross_ent_naked_coeff * (-naked_token_logprob) +
        (1 - config.token_cross_ent_naked_coeff) * state.token_cross_ent_naked
    )

    # Ensure all state variables maintain their original shapes
    new_state = ADSState(
        emwa_dir=new_emwa_dir,  # Shape: (batch_size, vsz)
        emwa_logp=new_emwa_logp,  # Shape: (batch_size, vsz)
        emwa_temp=new_emwa_temp,  # Shape: (batch_size,)
        token_cross_ent_scaffold=new_token_cross_ent_scaffold,  # Shape: (batch_size,)
        token_cross_ent_naked=new_token_cross_ent_naked,  # Shape: (batch_size,)  # Updated
        emwa_dir_ent=new_emwa_dir_ent,  # Shape: (batch_size,)
        entropy_rate_scaffold=new_entropy_rate_scaffold,  # Shape: (batch_size,)
        entropy_rate_naked=new_entropy_rate_naked # Shape: (batch size,)
    )
    
    return new_state, token_ids

# Example usage:
def create_sampler(bsz: int, vsz: int, config: Optional[ADSConfig] = None):
    """Create an Adaptive Dirichlet Sampler instance."""
    if config is None:
        config = ADSConfig()
    
    state = initialize_state(bsz, vsz, config)
    
    def sample(
        key: jax.random.PRNGKey,
        logits: jnp.ndarray,
        state: ADSState,
    ) -> Tuple[ADSState, jnp.ndarray]:
        """Sample a token using Adaptive Dirichlet Sampling."""
        return adaptive_dirichlet_step(
            key,
            state,
            logits,
            config
        )
    
    return sample, state