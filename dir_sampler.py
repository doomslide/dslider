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
    """Compute the expectation E[ln(X)|X~Dir(alpha)]"""
    # For Dir(alpha), E[ln(X_i)] = digamma(alpha_i) - digamma(sum(alpha))
    alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
    return -jnp.sum(jsp.special.digamma(alpha) - jsp.special.digamma(alpha_sum), axis=-1)

@jax.jit
def compute_dirichlet_logprob(probs: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    """Compute log probability of probs under Dirichlet(alpha)."""
    return jnp.sum((alpha - 1.0) * jnp.log(probs)) - jsp.special.gammaln(jnp.sum(alpha)) + \
           jnp.sum(jsp.special.gammaln(alpha))

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
    kl = jnp.clip(kl, 1e-8, 100.0)
    coeff = jnp.clip(
        config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS)),
        1e-3,
        0.999
    )
    coeff = coeff[..., None]  # Shape (batch_size, 1)
    return coeff * raw_logp + (1 - coeff) * emwa_logp

@partial(jax.jit, static_argnums=(0,1))
def initialize_state(bsz: int, vocab_size: int, config: ADSConfig) -> ADSState:
    """Initialize the state for the Adaptive Dirichlet Sampler."""
    return ADSState(
        emwa_dir=jnp.ones((bsz, vocab_size)),  # [batch_size, vocab_size]
        emwa_logp=jnp.zeros((bsz, vocab_size)),  # [batch_size, vocab_size]
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
    # Split PRNG key
    key1, key2 = jax.random.split(key)
    
    # 1. Update EMWA of log probabilities
    raw_logp = jax.nn.log_softmax(logits)
    new_emwa_logp = update_emwa_logp(raw_logp, state.emwa_logp, config)

    # 2. Find temperature (target entropy should be scalar)
    target_entropy = (
        config.entropy_a * state.token_cross_ent_scaffold + 
        config.entropy_b * state.entropy_rate_scaffold + 
        config.probs_ent_offset
    )
    pre_temp, _, _ = temp_tune(raw_logp, target_entropy)
    pre_temp = pre_temp
    
    # 3. Update EMWA temperature
    new_emwa_temp = config.emwa_temp_coeff * pre_temp + (1 - config.emwa_temp_coeff) * state.emwa_temp
    
    # 4. Compute naked_probs and update naked entropy rate
    naked_probs = jax.nn.softmax(raw_logp / jnp.clip(pre_temp[..., None], MIN_TEMP, MAX_TEMP))

    new_entropy_rate_naked = (
        config.entropy_rate_naked_coeff * entropy(naked_probs) +
        (1 - config.entropy_rate_naked_coeff) * state.entropy_rate_naked
    )
    
    # 5. Update EMWA of Dirichlet parameters
    new_dir_params, _, _ = fit_dirichlet(new_emwa_logp)
    new_emwa_dir = config.emwa_dir_coeff * new_dir_params + (1 - config.emwa_dir_coeff) * state.emwa_dir
    

    
    # 6. Update EMWA Dirichlet entropy
    baseline_likelihood = compute_dirichlet_logprob(
        jax.nn.softmax(new_emwa_logp),
        new_emwa_dir
    )
    dir_likelihood = compute_dirichlet_logprob(naked_probs, new_emwa_dir)
    new_emwa_dir_ent = (
        config.emwa_dir_ent_coeff * (-dir_likelihood) + 
        (1 - config.emwa_dir_ent_coeff) * state.emwa_dir_ent
    )
    
    # 7. Determine whether to use Dirichlet
    dir_threshold = (
        config.dirichlet_d * (-baseline_likelihood) +
        config.dirichlet_e * state.emwa_dir_ent +
        config.dir_ent_offset
    )
    
    use_dirichlet = (-dir_likelihood < dir_threshold)[..., None] 
    
    # 8. Interpolate if using Dirichlet
    dirichlet_expected_probs = dirichlet_expectation(new_emwa_dir)
    kl = kl_divergence(dirichlet_expected_probs, naked_probs)
    perturb_factor = 1 - config.perturb_base_coeff ** (-config.perturb_exp_coeff / (kl + EPS))
    final_probs = jnp.where(use_dirichlet, perturb_factor * dirichlet_expected_probs + (1 - perturb_factor) * naked_probs, naked_probs)
    
    # 9. Sample token and update entropy
    token = jax.random.categorical(key2, jnp.log(final_probs))
    entropy_rate = entropy(final_probs)  # Shape: (batch_size,)
    new_entropy_rate_scaffold = (
        config.entropy_rate_scaffold_coeff * entropy_rate + 
        (1 - config.entropy_rate_scaffold_coeff) * state.entropy_rate_scaffold
    )
    
    # 10. Update cross entropy
    # Get log prob for the sampled token - ensure scalar for each batch
    token_logprob = -jnp.log(final_probs.reshape(final_probs.shape[0], -1)[jnp.arange(final_probs.shape[0]), token])
    
    # Scale by temperature before the moving average
    scaled_token_logprob = token_logprob / jnp.maximum(new_emwa_temp, EPS)
    
    # Update the moving average with the scaled value - maintain shape [batch_size,]
    new_token_cross_ent_scaffold = (
        config.token_cross_ent_scaffold_coeff * scaled_token_logprob + 
        (1 - config.token_cross_ent_scaffold_coeff) * state.token_cross_ent_scaffold
    )

    # Update exponential moving weighted average of -raw_logp[token]
    new_token_cross_ent_naked = (
        config.token_cross_ent_naked_coeff * (-raw_logp[jnp.arange(logits.shape[0]), token]) +
        (1 - config.token_cross_ent_naked_coeff) * state.token_cross_ent_naked
    )

    # Ensure all state variables maintain their original shapes
    new_state = ADSState(
        emwa_dir=new_emwa_dir,  # Shape: (batch_size, vocab_size)
        emwa_logp=new_emwa_logp,  # Shape: (batch_size, vocab_size)
        emwa_temp=new_emwa_temp,  # Shape: (batch_size,)
        token_cross_ent_scaffold=new_token_cross_ent_scaffold,  # Shape: (batch_size,)
        token_cross_ent_naked=new_token_cross_ent_naked,  # Shape: (batch_size,)  # Updated
        emwa_dir_ent=new_emwa_dir_ent,  # Shape: (batch_size,)
        entropy_rate_scaffold=new_entropy_rate_scaffold,  # Shape: (batch_size,)
        entropy_rate_naked=new_entropy_rate_naked # Shape: (batch size,)
    )
    
    return new_state, token

# Example usage:
def create_sampler(bsz: int, vocab_size: int, config: Optional[ADSConfig] = None):
    """Create an Adaptive Dirichlet Sampler instance."""
    if config is None:
        config = ADSConfig()
    
    state = initialize_state(bsz, vocab_size, config)
    
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