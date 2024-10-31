# Adaptive Dirichlet Sampling

The adaptive Dirichlet sampler is a decooding method for language models which utilizes an exponentially moving average Dirichlet distribution modeling the quasi-stationary state in order to inject entropy into the model. 

## State variables:

The system maintains a set of state variables that are updated as the model processes tokens.

1. Exponential moving weighted average of the Dirichlet parameters: `emwa_dir`.
2. Exponential moving weighted average of the raw log probability vectors predicted by the language model: `emwa_logp`.
3. Exponential moving weighted average of the temperature: `emwa_temp`.
4. Exponential moving weighted average of -log_softmax(logits/emwa_temp)[sampled token] # proxy for cross entropy rate: `emwa_cross_ent`.
5. Exponential moving weighted average of negative log dirichlet likelihood, or "Dirichlet entropy": `emwa_dir_ent`
5. Exponential moving weighted average of entropy(coeff * softmax(logits/emwa_temp) + (1-coeff) * sample(dirichlet)) # proxy for entropy rate: `emwa_entropy`.

The sampler works as follows:

1. The model generates a logit vector for the next token: `logits`.
2. The emwa of log probabilities is updated: 
    - `raw_logp = log_softmax(logits)`
    - `emwa_logp_coeff = emwa_logp_base ** (- emwa_logp_exp_factor/KL(softmax(raw_logp)||softmax(emwa_logp)))`
    - `emwa_logp = emwa_logp_coeff * raw_logp + (1-emwa_logp_coeff) * emwa_logp`.
3. The emwa of dirichlet parameters is updated: 
    - `emwa_dir = emwa_dir_coeff *  fit_dirichlet(emwa_logp) + (1-emwa_dir_coeff) * emwa_dir` where `fit_dirichlet` is a function that fits dirichlet parameters to the emwa log probabilities. (See below.)
4. A temperature tuner finds `pre_temp` so that `pre_probs := softmax(raw_logp/pre_temp)` satisfies `entropy(pre_prob) == a * emwa_cross_ent + b * emwa_entropy  + probs_ent_offset` where `0<a, b <1` and `probs_ent_offset > 0` are pretuned hyperparameterss. (See below for details.)
5. Update emwa temperature: 
    - `emwa_temp = emwa_temp_coeff * pre_temp + (1-emwa_temp_coeff) * emwa_temp`
6. Compute dirichlet likelihood `dir_likelihood = Dir(pre_probs|emwa_dir)`
7. Update `emwa_dir_ent = emwa_dir_ent_coeff * (-log(dir_likelihood)) + (1-emwa_dir_ent_coeff) * emwa_dir_ent`
5. If `-log(dir_likelihood) < d * (-log(Dir(softmax(emwa_logp)|emwa_dir))) + e * emwa_dir_ent + dir_ent_offset`: 
    - `sampled_probs ~ Dir(emwa_dir)` # sample a probability vector from the dirichlet parameters
    - `perturb_factor = 1 - perturb_base_coeff ** (- perturb_exp_coeff/KL(sampled_probs||pre_probs))`
    - `probs = perturb_factor * sampled_probs + (1-perturb_factor) pre_probs` # interpolate
6. otherwise:
    - `probs = pre_probs`
    - `emwa_entropy = emwa_entropy_coeff * entropy(probs) + (1-emwa_entropy_coeff) * emwa_entropy`
7. Sample a token `token ~ probs`.
8. Update emwa of cross entropy:
    - `emwa_cross_ent = emwa_cross_ent_coeff * (-log(probs))[token]/emwa_temp) + (1-emwa_cross_ent_coeff) * emwa_cross_ent`
6. Append the token to the sequence and put back into the model. 

All hyperparameters are optimized (via grid search or possibly faster alternative) to minimize average `emwa_cross_ent`. 


Here is the function `temp_tune` which finds temperature to fit a given entropy target:

```python
@jax.jit
def ent_grad_hess(logits: jnp.ndarray, T: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    p = jax.nn.softmax(logits / T[:, None], axis=-1)
    log_p = jax.nn.log_softmax(logits / T[:, None], axis=-1)
    mu1 = jnp.sum(p * log_p, axis=-1)
    diff = log_p - mu1[:, None]
    mu2 = jnp.sum(p * diff**2, axis=-1)
    mu3 = jnp.sum(p * diff**3, axis=-1)
    return -mu1, mu2 / T, -(2 * mu3 + 3 * mu2) / (T * T)

@partial(jax.jit, static_argnums=(3, 4, 5))  # Mark static arguments
def temp_tune(
    logits: jnp.ndarray,
    target_ent: jnp.ndarray,
    T_init: float = 1.0,
    lr: float = 0.1,
    max_iters: int = 10,
    tol: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def scan_body(carry, _):
        T, iters, converged = carry
        ent, grad, hess = ent_grad_hess(logits, T)
        error = ent - target_ent
        new_converged = converged | (jnp.abs(error) < tol)
        denominator = 2 * grad * grad - error * hess
        halley_step = 2 * error * grad / jnp.where(
            jnp.abs(denominator) > 1e-8,
            denominator,
            jnp.inf
        )
        newton_step = error / jnp.where(jnp.abs(grad) > 1e-8, grad, jnp.inf)
        grad_step = jnp.where(error > 0, lr * T, -lr * T)
        delta_T = jnp.where(
            jnp.abs(grad) < 1e-8,
            grad_step,
            jnp.where(jnp.abs(denominator) < 1e-8, newton_step, halley_step)
        )
        delta_T = jnp.clip(delta_T, -0.5 * T, 0.5 * T)
        new_T = jnp.where(
            new_converged,
            T,
            jnp.maximum(T - delta_T, T / 2)
        )
        
        return (new_T, iters + 1, new_converged), None
    batch_size = logits.shape[0]
    init_state = (
        jnp.full((batch_size,), T_init, dtype=logits.dtype),
        jnp.zeros(batch_size, dtype=jnp.int32),
        jnp.zeros(batch_size, dtype=jnp.bool_),
    )
    (final_T, final_iters, final_converged), _ = jax.lax.scan(
        scan_body,
        init_state,
        None,
        length=max_iters
    )
    return final_T, final_iters, final_converged
```

Here is the function `fit_dirichlet`:

```python
@jax.jit
def halley_update(alpha, target_values):
    """
    Compute the Halley's method update direction.
    """
    p1 = jsp.polygamma(1, alpha)
    p2 = jsp.polygamma(2, alpha)
    S = jnp.sum(alpha)
    s1 = jsp.polygamma(1, S)
    s2 = jsp.polygamma(2, S)
    p1_inv = 1.0 / p1
    sum_p1_inv = jnp.sum(p1_inv)
    denom = 1.0 - s1 * sum_p1_inv
    denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)
    coeff = s1 / denom
    error = jsp.digamma(alpha) - jsp.digamma(S) - target_values
    temp = p1_inv * error
    J_inv_error = temp + coeff * jnp.sum(temp) * p1_inv
    sum_J_inv_error = jnp.sum(J_inv_error)
    H_J_inv_error = p2 * J_inv_error - s2 * sum_J_inv_error
    temp2 = p1_inv * H_J_inv_error
    sum_temp2 = jnp.sum(temp2)
    J_inv_H_J_inv_error = temp2 + coeff * sum_temp2 * p1_inv
    return -J_inv_error + 0.5 * J_inv_H_J_inv_error

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def fit_dirichlet(
    target_values,
    init_alpha=None,
    initial_lr=1.2,
    decay_alpha=0.1,
    decay_beta=2.0,
    decay_gamma=0.25,
    decay_nu=0.75,
    max_iters=140,
    tol=1e-4
):
    """
    Estimate Dirichlet parameters (alpha) from target logprobs.
    """
    n = target_values.size
    min_lr = 1e-8
    if init_alpha is None:
        init_alpha = jnp.ones(n)
    def scan_body(carry, _):
        alpha, converged, error_norm, step = carry
        S = jnp.sum(alpha)
        digamma_alpha = jsp.digamma(alpha)
        psi_S = jsp.digamma(S)
        error = digamma_alpha - psi_S - target_values
        error_norm = jnp.linalg.norm(error)
        new_converged = converged | (error_norm < tol)
        lr = initial_lr * jnp.exp(-decay_alpha * (step ** decay_nu)) * jnp.abs(
            jnp.cos(decay_beta / (step ** decay_gamma))
        )
        lr = jnp.maximum(lr, min_lr)
        delta_alpha = halley_update(alpha, target_values)
        scaled_delta_alpha = lr * delta_alpha
        max_delta = 0.5 * alpha
        scaled_delta_alpha = jnp.clip(scaled_delta_alpha, -max_delta, max_delta)
        new_alpha = jnp.where(
            new_converged,
            alpha,
            jnp.maximum(alpha + scaled_delta_alpha, alpha / 2)
        )
        return (new_alpha, new_converged, error_norm, step + 1), None

    init_state = (
        init_alpha,
        jnp.bool_(False),
        jnp.inf,
        jnp.int32(1)
    )
    (final_alpha, final_converged, _, final_step), _ = jax.lax.scan(
        scan_body,
        init_state,
        None,
        length=max_iters
    )
    return final_alpha, final_step - 1, final_converged
```


------

"""
# Adaptive Dirichlet Sampling

The adaptive Dirichlet sampler is a decoding method for language models that uses exponentially moving averages to track and adjust sampling behavior over time.

## State Variables
The sampler maintains several moving averages:
- `emwa_dir`: Dirichlet parameters fit to recent token distributions
- `emwa_logp`: Log probabilities from the language model
- `emwa_temp`: Temperature used for softmax
- `emwa_cross_ent`: Cross entropy of sampled tokens
- `emwa_dir_ent`: Entropy of the Dirichlet distribution
- `emwa_entropy`: Entropy of the final sampling distribution

## Process
For each token generation:
1. Get logits from the language model
2. Update moving average of log probabilities
3. Fit and update Dirichlet parameters
4. Adjust temperature to target entropy level
5. Either:
   - Sample from Dirichlet and blend with softmax probabilities
   - Or use softmax probabilities directly
6. Sample next token
7. Update state variables

The sampler aims to maintain diversity in the generated text while staying close to the model's original probability distribution. The moving averages help adapt the sampling behavior based on recent history.
"""
