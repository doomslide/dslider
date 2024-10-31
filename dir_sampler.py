import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
from typing import NamedTuple, Tuple, Optional
import chex
from utils import temp_tune, fit_dirichlet

class ADSState(NamedTuple):
    """State maintained by the Adaptive Dirichlet Sampler"""
    emwa_dir: jnp.ndarray  # Dirichlet parameters
    emwa_logp: jnp.ndarray  # Log probabilities
    emwa_temp: jnp.ndarray  # Temperature
    emwa_cross_ent: jnp.ndarray  # Cross entropy rate
    emwa_dir_ent: jnp.ndarray  # Dirichlet entropy
    emwa_entropy: jnp.ndarray  # Entropy rate

class ADSConfig(NamedTuple):
    """Configuration for Adaptive Dirichlet Sampler"""
    emwa_logp_base: float = 0.99
    emwa_logp_exp_factor: float = 1.0
    emwa_dir_coeff: float = 0.99
    emwa_temp_coeff: float = 0.99
    emwa_dir_ent_coeff: float = 0.99
    emwa_entropy_coeff: float = 0.99
    emwa_cross_ent_coeff: float = 0.99
    perturb_base_coeff: float = 0.99
    perturb_exp_coeff: float = 1.0
    probs_ent_offset: float = 0.1
    dir_ent_offset: float = 0.1
    entropy_a: float = 0.5
    entropy_b: float = 0.3
    dirichlet_d: float = 0.5
    dirichlet_e: float = 0.3

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
        config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + 1e-8)),
        1e-3,
        0.999
    )
    # Add keepdims to maintain proper broadcasting
    coeff = coeff[..., None]  # Shape becomes (batch_size, 1)
    return coeff * raw_logp + (1 - coeff) * emwa_logp

@partial(jax.jit, static_argnums=(0,1))
def initialize_state(bsz: int, vocab_size: int, config: ADSConfig) -> ADSState:
    """Initialize the state for the Adaptive Dirichlet Sampler."""
    return ADSState(
        emwa_dir=jnp.zeros((bsz, vocab_size)),
        emwa_logp=jnp.zeros((bsz, vocab_size)),
        emwa_temp=jnp.ones((bsz,)),
        emwa_cross_ent=jnp.zeros((bsz,)),
        emwa_dir_ent=jnp.zeros((bsz,)),
        emwa_entropy=jnp.zeros((bsz,))
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
    
    # 2. Update EMWA of Dirichlet parameters
    new_dir_params, _, _ = fit_dirichlet(new_emwa_logp[None])
    new_dir_params = new_dir_params[0]
    new_emwa_dir = config.emwa_dir_coeff * new_dir_params + (1 - config.emwa_dir_coeff) * state.emwa_dir
    
    # 3. Find temperature (target entropy should be scalar)
    target_entropy = (
        config.entropy_a * state.emwa_cross_ent + 
        config.entropy_b * state.emwa_entropy + 
        config.probs_ent_offset
    )
    print(f'shape of target_entropy: {target_entropy.shape}')
    pre_temp, _, _ = temp_tune(raw_logp, target_entropy)
    pre_temp = pre_temp
    
    # 4. Update EMWA temperature
    new_emwa_temp = config.emwa_temp_coeff * pre_temp + (1 - config.emwa_temp_coeff) * state.emwa_temp
    
    # 5. Compute pre_probs and check Dirichlet likelihood
    pre_probs = jax.nn.softmax(raw_logp[0] / jnp.clip(pre_temp, 1e-8, 100.0))
    pre_probs = jnp.clip(pre_probs, 1e-8, 1.0)
    pre_probs = pre_probs / jnp.sum(pre_probs)
    
    dir_likelihood = compute_dirichlet_logprob(pre_probs, new_emwa_dir)
    baseline_likelihood = compute_dirichlet_logprob(
        jax.nn.softmax(new_emwa_logp),
        new_emwa_dir
    )
    
    # 6. Update EMWA Dirichlet entropy
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
    
    use_dirichlet = -dir_likelihood < dir_threshold
    
    # 8. Sample and interpolate if using Dirichlet
    sampled_probs = sample_dirichlet(key1, new_emwa_dir)
    kl = kl_divergence(sampled_probs, pre_probs)
    perturb_factor = 1 - config.perturb_base_coeff ** (-config.perturb_exp_coeff / (kl + 1e-8))
    dirichlet_probs = perturb_factor * sampled_probs + (1 - perturb_factor) * pre_probs
    
    # Use where for batching
    final_probs = jnp.where(use_dirichlet, dirichlet_probs, pre_probs)
    
    # 9. Sample token and update entropy
    token = jax.random.categorical(key2, jnp.log(final_probs))
    entropy_rate = entropy(final_probs)  # Should be shape (batch_size,)
    new_emwa_entropy = (
        config.emwa_entropy_coeff * entropy_rate + 
        (1 - config.emwa_entropy_coeff) * state.emwa_entropy
    )
    
    # 10. Update cross entropy (per spec)
    token_logprob = -jnp.log(final_probs)[jnp.arange(final_probs.shape[0]), token]  # Get logprob for each batch
    new_emwa_cross_ent = (
        config.emwa_cross_ent_coeff * (token_logprob / new_emwa_temp) +
        (1 - config.emwa_cross_ent_coeff) * state.emwa_cross_ent
    )
    
    # Construct new state
    new_state = ADSState(
        emwa_dir=new_emwa_dir,
        emwa_logp=new_emwa_logp,
        emwa_temp=new_emwa_temp,
        emwa_cross_ent=new_emwa_cross_ent,
        emwa_dir_ent=new_emwa_dir_ent,
        emwa_entropy=new_emwa_entropy
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