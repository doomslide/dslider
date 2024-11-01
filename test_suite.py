import os
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple, Callable
import numpy as np
from dir_sampler import ADSState, ADSConfig, adaptive_dirichlet_step, initialize_state
from test import CACHE_DIR
from rich import print


# Type definitions
class GenerationState(NamedTuple):
    """State maintained during generation"""
    sequence: jnp.ndarray  # Full pre-allocated sequence
    ads_state: ADSState  # ADS state
    current_pos: jnp.ndarray  # Current position in sequence

@partial(jax.jit, static_argnames=('model_fn', 'max_new_tokens'))
def generate_sequence(
    key: jax.random.PRNGKey,
    input_ids: jnp.ndarray,
    model_fn: Callable,
    config: ADSConfig,
    max_new_tokens: int,
) -> jnp.ndarray:
    """Generate text sequence using ADS sampling"""
    # Pre-allocate full sequence
    batch_size, init_len = input_ids.shape
    full_seq = jnp.zeros((batch_size, init_len + max_new_tokens), dtype=jnp.int32)
    full_seq = full_seq.at[:, :init_len].set(input_ids)
    
    # Initialize ADS state
    ads_state = initialize_state(batch_size, model_fn.config.vocab_size, config)
    
    def scan_fn(carry, idx):
        seq, state, key = carry
        key, subkey = jax.random.split(key)
        
        # Get the current sequence by using JAX mask operations
        # Create a mask that's 1 up to current position, 0 afterwards
        seq_len_mask = jnp.arange(seq.shape[1]) < (init_len + idx)
        
        # Use the mask to zero out future positions
        masked_seq = jnp.where(seq_len_mask[None, :], seq, 0)
        
        # Get logits from model (model should handle masked inputs appropriately)
        logits = model_fn(masked_seq)[:, -1, :]
        
        # Sample token using ADS
        new_state, token = adaptive_dirichlet_step(subkey, state, logits, config)
        
        # Update sequence using at-based indexing
        new_seq = seq.at[:, init_len + idx].set(token)
        
        return (new_seq, new_state, key), None
    
    # Run generation
    final_seq, _, _ = jax.lax.scan(
        scan_fn,
        (full_seq, ads_state, key),
        jnp.arange(max_new_tokens)
    )[0]
    
    return final_seq

# Example usage with a Hugging Face model:
def create_model_fn(model, params):
    """Create a model function compatible with the sampler"""
    @partial(jax.jit, static_argnames=('train',))
    def model_fn(input_ids, train=False):
        return model(
            input_ids=input_ids,
            params=params,
            train=train
        ).logits
    model_fn.config = model.config
    return model_fn

def main():
    from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, LlamaConfig
    
    # Verify GPU is available
    print("JAX devices:", jax.devices())
    
    # Load model and tokenizer with memory optimizations
    model_name = "meta-llama/Llama-3.2-1B"
    
    # Create config with smaller context window
    config = LlamaConfig.from_pretrained(model_name)
    config.max_position_embeddings = 2048  # Reduce from default (usually 4096 or higher)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load model with reduced context length
    model = FlaxAutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        cache_dir=CACHE_DIR,
        _do_init=True,
        from_pt=False
    )
    
    # Clear JAX cache after loading
    jax.clear_caches()
    
    # Create model function
    model_fn = create_model_fn(model, model.params)
    
    # Prepare inputs
    prompts = ["The quick brown fox", "Once upon a time"]
    encoded = tokenizer(prompts, padding=True, return_tensors="np")
    
    # Run evaluation
    key = jax.random.PRNGKey(0)
    sequences = generate_sequence(
        key=key,
        model_fn=model_fn,
        input_ids=encoded['input_ids'],
        config=ADSConfig(),
        max_new_tokens=20
    )
    # Print results
    print("\nGenerated Sequences:")
    for i, seq in enumerate(sequences):
        print(f"Prompt {i+1}: {tokenizer.decode(seq)}")

if __name__ == "__main__":
    main()