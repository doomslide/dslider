import jax
import jax.numpy as jnp
from flax.training.common_utils import shard
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import time
from functools import partial
import json
from dataclasses import asdict
from tqdm.auto import tqdm
from dir_sampler import create_sampler, ADSConfig, ADSState, adaptive_dirichlet_step

HF_TOKEN = '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'
CACHE_DIR = '/home/cloudforest/Weights'

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
    
    # Update sequence
    output_ids = jnp.concatenate([output_ids, token[None, None]], axis=1)
    attention_mask = jnp.concatenate([attention_mask, jnp.ones((1, 1), dtype=jnp.int32)], axis=1)
    
    # Collect metrics
    metrics = GenerationMetrics(
        temperatures=new_state.emwa_temp,
        cross_ents=new_state.emwa_cross_ent,
        entropies=new_state.emwa_entropy,
        dir_ents=new_state.emwa_dir_ent,
        generation_time=0.0,  # hf_KiGgljxzcqpbXkiJiyuHQySrOermsPtTeWWill be updated later
        output_ids=output_ids
    )
    
    return (output_ids, attention_mask, new_state, key), metrics

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
    
    init_carry = (input_ids, attention_mask, state, key)
    
    # Use scan for efficient iteration
    (final_output_ids, _, final_state, _), metrics = jax.lax.scan(
        lambda c, _: generate_sequence_step(c, None, model, config, max_new_tokens),
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
    model_name: str = "gpt2",
    cache_dir: str = CACHE_DIR,
    num_samples: int = 5,
    sequence_length: int = 100,
    prompt: str = "In a magical forest,",
    configs: Optional[Dict[str, ADSConfig]] = None
):
    """Create a test suite for the Adaptive Dirichlet Sampler."""
    
    # Load model and tokenizer
    print(f"Loading model {model_name}...")
    model = FlaxAutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        token=HF_TOKEN
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create default configs if none provided
    if configs is None:
        configs = {
            "standard": ADSConfig(),
            "high_entropy": ADSConfig(
                entropy_a=0.7,
                entropy_b=0.5,
                probs_ent_offset=0.2,
                dir_ent_offset=0.2,
                dirichlet_d=0.7,
                dirichlet_e=0.5
            ),
            "low_entropy": ADSConfig(
                entropy_a=0.3,
                entropy_b=0.1,
                probs_ent_offset=0.05,
                dir_ent_offset=0.05,
                dirichlet_d=0.3,
                dirichlet_e=0.1
            )
        }
    
    # Initialize samplers
    print("Initializing samplers...")
    samplers = {
        name: create_sampler(1, model.config.vocab_size, config)
        for name, config in configs.items()
    }
    
    # Pre-compile generation function
    generate_fn = partial(
        generate_sequence,
        model,
        max_new_tokens=sequence_length
    )
    
    # Warm up JIT compilation
    print("Warming up JIT compilation...")
    dummy_input = tokenizer("test", return_tensors="jax", padding=True)
    _, dummy_state = samplers["standard"]
    _ = generate_fn(
        dummy_input.input_ids,
        dummy_input.attention_mask,
        dummy_state,
        jax.random.PRNGKey(0),
        configs["standard"]
    )
    
    def run_test(name: str):
        print(f"\nRunning test for {name}...")
        config = configs[name]
        _, state = samplers[name]
        
        all_metrics = []
        generated_texts = []
        
        for i in tqdm(range(num_samples)):
            # Prepare input
            model_inputs = tokenizer(prompt, return_tensors="jax", padding=True)
            
            # Generate
            start_time = time.time()
            output_ids, metrics = generate_fn(
                model_inputs.input_ids,
                model_inputs.attention_mask,
                state,
                jax.random.PRNGKey(i),
                config
            )
            generation_time = time.time() - start_time
            
            # Process outputs
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_texts.append(output_text)
            
            # Collect metrics
            metrics["generation_time"] = generation_time
            all_metrics.append(metrics)
            
            # Print results
            print(f"\nSample {i+1}:")
            print(output_text)
            print(f"Generation time: {generation_time:.2f}s")
            print(f"Average temperature: {np.mean(metrics['temperatures']):.3f}")
            print(f"Average cross entropy: {np.mean(metrics['cross_ents']):.3f}")
            print(f"Average entropy: {np.mean(metrics['entropies']):.3f}")
            print(f"Average Dirichlet entropy: {np.mean(metrics['dir_ents']):.3f}")
        
        return {
            "config": asdict(config),
            "metrics": all_metrics,
            "generated_texts": generated_texts
        }
    
    # Run all configurations
    results = {
        name: run_test(name)
        for name in configs.keys()
    }
    
    # Print comparative analysis
    print("\nComparative Analysis:")
    print("=" * 50)
    
    for name, result in results.items():
        metrics = result["metrics"]
        avg_metrics = {
            k: float(np.mean([m[k] for m in metrics]))
            for k in ["generation_time", "temperatures", "cross_ents", "entropies", "dir_ents"]
        }
        
        print(f"\n{name.upper()} Configuration:")
        for metric_name, value in avg_metrics.items():
            print(f"Average {metric_name}: {value:.3f}")
    
    return results

if __name__ == "__main__":
    results = create_test_suite(
        model_name="meta-llama/Llama-3.2-1B",
        cache_dir=CACHE_DIR,
        num_samples=5,
        sequence_length=100,
        prompt="In a magical forest,"
    )
