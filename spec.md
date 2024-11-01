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
5. Update emwa temperature and entropy: 
    - `emwa_temp = emwa_temp_coeff * pre_temp + (1-emwa_temp_coeff) * emwa_temp`
    - `emwa_entropy = emwa_entropy_coeff * entropy(pre_probs) + (1-emwa_entropy_coeff) * emwa_entropy`
6. Compute dirichlet likelihood `dir_likelihood = Dir(pre_probs|emwa_dir)`
7. Update `emwa_dir_ent = emwa_dir_ent_coeff * (-log(dir_likelihood)) + (1-emwa_dir_ent_coeff) * emwa_dir_ent`
5. If `-log(dir_likelihood) < d * (-log(Dir(softmax(emwa_logp)|emwa_dir))) + e * emwa_dir_ent + dir_ent_offset`: 
    - `sampled_probs ~ Dir(emwa_dir)` # sample a probability vector from the dirichlet parameters (or take `E[p|emwa_dir]`)
    - `perturb_factor = 1 - perturb_base_coeff ** (- perturb_exp_coeff/KL(sampled_probs||pre_probs))`
    - `probs = perturb_factor * sampled_probs + (1-perturb_factor) pre_probs` # interpolate
    - 
6. otherwise:
    - `probs = pre_probs`
7. Sample a token `token ~ probs`.
8. Update emwa of cross entropy:
    - `emwa_cross_ent = emwa_cross_ent_coeff * (-log(probs))[token]/emwa_temp) + (1-emwa_cross_ent_coeff) * emwa_cross_ent`
6. Append the token to the sequence and put back into the model. 

All hyperparameters are optimized (via grid search or possibly faster alternative) to minimize `emwa_cross_ent/emwa_entropy`. 
