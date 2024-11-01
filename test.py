import jax
import jax.numpy as jnp
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer, AutoConfig
import os

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import time
from functools import partial
import json
from dataclasses import asdict
from tqdm.auto import tqdm
from dir_sampler import create_sampler, ADSConfig, ADSState, adaptive_dirichlet_step
from rich import print
import psutil

HF_TOKEN = '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'
CACHE_DIR = '/home/cloudforest/Weights'
jax.config.update('jax_platform_name', 'gpu')

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*unhashable type.*')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORMS'] = '' 

class GenerationMetrics(NamedTuple):
    """Metrics collected during generation"""
    temperatures: jnp.ndarray  # Shape: (seq_len,)
    cross_ents: jnp.ndarray  # Shape: (seq_len,)
    entropies: jnp.ndarray  # Shape: (seq_len,)
    dir_ents: jnp.ndarray  # Shape: (seq_len,)
    generation_time: float
    output_ids: jnp.ndarray  # Shape: (1, seq_len)

@partial(jax.jit, static_argnames=('model', 'max_new_tokens'))
def generate_sequence_step(
    carry: Tuple[jnp.ndarray, jnp.ndarray, ADSState, jax.random.PRNGKey],
    _: Any,
    model: FlaxAutoModelForCausalLM,
    config: ADSConfig,
    max_new_tokens: int,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, ADSState, jax.random.PRNGKey], GenerationMetrics]:
    """Single step of the generation process, JIT-compiled for efficiency."""
    output_ids, attention_mask, state, key = carry
    
    # Get model outputs
    outputs = model(input_ids=output_ids, attention_mask=attention_mask)
    next_token_logits = outputs.logits[:, -1, :]
    
    # Sample next token
    key, subkey = jax.random.split(key)
    new_state, token = adaptive_dirichlet_step(
        subkey,
        state,
        next_token_logits,
        config
    )
    
    # Update the sequences in place
    curr_seq_len = output_ids.shape[1]
    new_output_ids = output_ids.at[:, -1].set(token)
    new_attention_mask = attention_mask.at[:, -1].set(1)
    
    # Collect metrics
    metrics = GenerationMetrics(
        temperatures=new_state.emwa_temp,
        cross_ents=new_state.emwa_cross_ent,
        entropies=new_state.emwa_entropy,
        dir_ents=new_state.emwa_dir_ent,
        generation_time=0.0,
        output_ids=new_output_ids
    )
    
    return (new_output_ids, new_attention_mask, new_state, key), metrics

@partial(jax.jit, static_argnames=('model', 'max_new_tokens'))
def generate_sequence(
    model: FlaxAutoModelForCausalLM,
    input_ids: jnp.ndarray,
    attention_mask: jnp.ndarray,
    state: ADSState,
    key: jax.random.PRNGKey,
    config: ADSConfig,
    max_new_tokens: int,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Generate sequence with JIT-compiled steps."""
    
    # Start with just the input sequence
    init_carry = (input_ids, attention_mask, state, key)
    
    def scan_fn(carry, _):
        output_ids, attn_mask, state, key = carry
        return generate_sequence_step(carry, None, model, config, max_new_tokens)
    
    # Use scan for efficient iteration
    (final_output_ids, _, final_state, _), metrics = jax.lax.scan(
        scan_fn,
        init_carry,
        None,
        length=max_new_tokens
    )
    
    return final_output_ids, {
        "temperatures": metrics.temperatures,
        "cross_ents": metrics.cross_ents,
        "entropies": metrics.entropies,
        "dir_ents": metrics.dir_ents,
        "output_ids": metrics.output_ids
    }
def create_test_suite(
    model: FlaxAutoModelForCausalLM,
    bsz: int,
    tokenizer: AutoTokenizer,
    num_samples: int = 5,
    sequence_length: int = 100,
    prompt: str = "In a magical forest,"
) -> Dict[str, Any]:
    print("\n" + "="*80)
    print(f"Test Configuration:")
    print(f"Batch size: {bsz}")
    print(f"Number of samples: {num_samples}")
    print(f"Sequence length: {sequence_length}")
    print(f"Prompt: '{prompt}'")
    print("="*80 + "\n")

    # Create samplers and configs (keeping your existing config)
    configs = {
        "standard": ADSConfig(
            emwa_logp_base=0.99,
            emwa_logp_exp_factor=1.0,
            emwa_dir_coeff=0.99,
            emwa_temp_coeff=0.99,
            emwa_dir_ent_coeff=0.99,
            emwa_entropy_coeff=0.99,
            emwa_cross_ent_coeff=0.99,
            perturb_base_coeff=0.99,
            perturb_exp_coeff=1.0,
            probs_ent_offset=0.1,
            dir_ent_offset=0.1,
            entropy_a=0.5,
            entropy_b=0.3,
            dirichlet_d=0.5,
            dirichlet_e=0.3
        )
    }
    
    print("Initializing samplers...")
    samplers = {
        name: create_sampler(bsz, model.config.vocab_size, config)
        for name, config in configs.items()
    }
    
    results = {}
    
    input_ids = tokenizer(prompt, return_tensors="jax").input_ids
    attention_mask = jnp.ones_like(input_ids)
    
    for name, (sampler, state) in samplers.items():
        print(f"\n{'='*40} Testing {name} configuration {'='*40}")
        samples = []
        metrics = []
        
        for i in range(num_samples):
            jax.clear_caches()
            key = jax.random.PRNGKey(i)
            
            try:
                output_ids, generation_metrics = generate_sequence(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    state=state,
                    key=key,
                    config=configs[name],
                    max_new_tokens=sequence_length
                )
                
                text = tokenizer.decode(output_ids[0])
                samples.append(text)
                metrics.append(generation_metrics)
                
                print(f"\n----- Sample {i+1} -----")
                print(f"Generated text: {text[:200]}...")
                print("\nMetrics:")
                print(f"  Temperature: {jnp.mean(generation_metrics['temperatures']):.3f}")
                print(f"  Entropy: {jnp.mean(generation_metrics['entropies']):.3f}")
                print(f"  Cross Entropy: {jnp.mean(generation_metrics['cross_ents']):.3f}")
                print(f"  Dirichlet Entropy: {jnp.mean(generation_metrics['dir_ents']):.3f}")
                
            except Exception as e:
                print(f"\nError generating sample {i+1}:")
                print(f"  {str(e)}")
                continue
        
        # Print summary statistics for this configuration
        print(f"\n{'-'*40} Summary for {name} {'-'*40}")
        avg_metrics = {
            key: jnp.mean(jnp.array([m[key] for m in metrics])) 
            for key in ['temperatures', 'entropies', 'cross_ents', 'dir_ents']
        }
        print("\nAverage Metrics Across All Samples:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {float(value):.3f}")
        
        results[name] = {
            "samples": samples,
            "metrics": metrics,
            "average_metrics": avg_metrics
        }
    
    return results

# def print_memory_usage():
#     process = psutil.Process()
#     print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# print("Loading model gpt2...")
# tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=CACHE_DIR)

# # Modify the model loading section
# model = FlaxAutoModelForCausalLM.from_pretrained(
#     "gpt2",
#     cache_dir=CACHE_DIR,
#     dtype=jnp.float32,  # Explicitly request float32
#     _do_init=True  # Make sure the model is initialized
# )

# # Clear memory after loading
# jax.clear_caches()

# print("\nMemory before first inference:")
# print_memory_usage()

# # Fix the model inference call
# test_input = jnp.ones((1, 2), dtype=jnp.int32)
# _ = model(input_ids=test_input, attention_mask=jnp.ones_like(test_input))  # Add attention mask

# print("\nMemory after first inference:")
# print_memory_usage()

# if __name__ == "__main__":
#     results = create_test_suite(
#         model=model,
#         bsz=1,
#         tokenizer=tokenizer,
#         num_samples=5,
#         sequence_length=100,
#         prompt="In a magical forest,"
#     )
