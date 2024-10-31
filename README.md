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
