import jax
import jax.numpy as jnp
import gc
import logging
import os
from rich import print
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
    AutoConfig,
)
from functools import partial

# Constants
HF_TOKEN = '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'
CACHE_DIR = '/home/cloudforest/Weights'

# JAX configuration
try:
    jax.config.update('jax_platform_name', 'gpu')
except RuntimeError:
    print("GPU not available, falling back to CPU")
    
# Configure logging
logging.getLogger("transformers").setLevel(logging.WARNING)

def clear_memory():
    """Clear JAX memory caches and run garbage collection"""
    try:
        gc.collect()
        jax.clear_caches()
    except Exception as e:
        print(f"Warning: Memory clearing failed: {e}")

def inference_step(params, input_ids, model):
    """Single inference step with gradient stopping"""
    # Stop gradients on both parameters and inputs
    params = jax.tree_map(jax.lax.stop_gradient, params)
    input_ids = jax.lax.stop_gradient(input_ids)
    
    # Run model in eval mode with stopped gradients
    outputs = model(
        input_ids=input_ids,
        params=params,
        train=False,
        dropout_rng=None
    )
    
    # Stop gradients on outputs
    return jax.tree_map(jax.lax.stop_gradient, outputs)

def create_generation_fn(model):
    """Create a JIT-compiled function for inference-only generation."""
    try:
        # Convert and freeze params
        params = jax.tree_map(
            lambda x: jax.lax.stop_gradient(jnp.asarray(x, dtype=jnp.float32)),
            model.params
        )
        
        # Test with minimal input
        test_input = jnp.zeros((1, 1), dtype=jnp.int32)
        
        # Create inference-only test
        outputs = inference_step(params, test_input, model)
        clear_memory()
        
        # Create JIT-compiled inference function
        @partial(jax.jit, static_argnums=(2,))
        def get_next_token_logits(params, input_ids, model):
            """Get logits for next token prediction with stopped gradients."""
            outputs = inference_step(params, input_ids, model)
            return jax.lax.stop_gradient(outputs.logits[:, -1, :])
                
        # Return a wrapped version that closes over the params
        return lambda input_ids: get_next_token_logits(params, input_ids, model)
        
    except Exception as e:
        print(f"Error creating generation function: {e}")
        raise

def main():
    if not HF_TOKEN:
        raise ValueError("HuggingFace token not provided")
        
    try:
        login(HF_TOKEN, add_to_git_credential=True)
    except Exception as e:
        print(f"HuggingFace login failed: {e}")
        raise

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    try:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        print("Loading model configuration...")
        
        config = AutoConfig.from_pretrained(model_name)
        config.max_position_embeddings = 128
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )
        
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Loading model in inference-only mode...")
        model = FlaxAutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            config=config,
            dtype=jnp.float32,
            _do_init=True,
            local_files_only=False,
        )
        params = model.params
        
        print("Creating inference-only generation function...")
        get_logits = create_generation_fn(model)
        
        # Test generation
        prompt = "The quick brown fox"
        print(f"\nTesting generation with prompt: {prompt}")
        
        try:
            # Convert input to JAX array and stop its gradients
            input_ids = jax.lax.stop_gradient(
                tokenizer(prompt, return_tensors="jax").input_ids
            )
            
            # Get logits with stopped gradients
            logits = get_logits(input_ids)
            
            # Sample from logits
            key = jax.random.PRNGKey(0)
            next_token = jax.random.categorical(key, logits, axis=-1)
            
            generated = tokenizer.decode(next_token[0])
            print(f"Generated token: {generated}")
            
        except Exception as e:
            print(f"Generation test failed: {e}")
            raise
            
        # Clean up
        del model
        clear_memory()
        
        print("Model loaded and tested successfully!")
        return get_logits, tokenizer
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program failed: {e}")