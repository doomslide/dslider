import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, AutoConfig
from functools import partial
import time
from dir_sampler import (
    adaptive_dirichlet_step, 
    initialize_state, 
    entropy
)
from rich import print
from config import ADSConfig, DEFAULT_ADS_CONFIG, CACHE_DIR, MODEL_NAME, EPS

# JAX configuration
try:
    jax.config.update('jax_platform_name', 'gpu')
except RuntimeError:
    print("GPU not available, falling back to CPU")


def create_model_fn(model, pad_token_id):
    """Create a wrapper function that handles the model call correctly."""
    @jax.jit
    def model_fn(params, input_ids):
        # Ensure input_ids is int32
        input_ids = input_ids.astype(jnp.int32)
        
        batch_size, seq_len = input_ids.shape
        pad_length = 128 - seq_len
        
        padded_input_ids = jnp.pad(
            input_ids,
            ((0, 0), (0, pad_length)),
            mode='constant',
            constant_values=pad_token_id
        ).astype(jnp.int32)
        
        attention_mask = jnp.pad(
            jnp.ones_like(input_ids, dtype=jnp.int32),
            ((0, 0), (0, pad_length)),
            mode='constant',
            constant_values=0
        )
        
        position_ids = jnp.pad(
            jnp.arange(seq_len, dtype=jnp.int32)[None, :],
            ((0, 0), (0, pad_length)),
            mode='constant',
            constant_values=0
        )
        
        # Debug print before model call
        print("Model input dtypes:", {
            'input_ids': padded_input_ids.dtype,
            'attention_mask': attention_mask.dtype,
            'position_ids': position_ids.dtype
        })
        
        # Call model with explicitly typed inputs
        outputs = model.module.apply(
            {'params': params},
            padded_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=True,
            method=model.module.__call__
        )
        return outputs
    return model_fn

def get_next_token(params, apply_fn, input_ids, key, temperature=1e-3):
    """Get next token using sampling."""
    @jax.jit
    def _get_next_token(params, input_ids, key):
        outputs = apply_fn(params, input_ids)
        logits = outputs.logits[:, -1, :] / temperature
        return jax.random.categorical(key, logits, axis=-1)
    return _get_next_token(params, input_ids, key)

@partial(jax.jit, static_argnames=('apply_fn', 'max_length', 'config', 'vocab_size', 'dtype'))
def generate_sequence_ads(params, apply_fn, input_ids, key, vocab_size, max_length=20, config=DEFAULT_ADS_CONFIG, dtype=jnp.bfloat16):
    """Generate sequence using adaptive Dirichlet sampling with efficient JAX implementation."""
    batch_size = input_ids.shape[0]
    
    # Initialize context buffer with input_ids
    context_buffer = jnp.zeros((batch_size, 128), dtype=jnp.int32)
    context_buffer = context_buffer.at[:, -input_ids.shape[1]:].set(input_ids)
    
    # Initialize ADS state - pass dtype here
    ads_state = initialize_state(batch_size, vocab_size, config, dtype=dtype)
    
    def scan_fn(carry, _):
        context, key, state = carry
        next_key, sample_key = jax.random.split(key)
        
        outputs = apply_fn(params, context)
        logits = outputs.logits[:, -1, :]
        
        # Generate next token using ADS
        new_state, next_token = adaptive_dirichlet_step(sample_key, state, logits, config)
        
        # Shift context buffer and add new token
        new_context = jnp.roll(context, shift=-1, axis=1)
        new_context = new_context.at[:, -1].set(next_token)
        
        return (new_context, next_key, new_state), next_token
    
    # Run the scan
    _, output_tokens = jax.lax.scan(
        scan_fn,
        (context_buffer, key, ads_state),
        jnp.arange(max_length)
    )
    
    return output_tokens.T

def analyze_sampling_stats(logits, final_probs, token_id):
    """Analyze sampling statistics for a single step."""
    # Convert inputs to bfloat16 for stable computation
    logits = logits.astype(jnp.bfloat16)
    final_probs = final_probs.astype(jnp.bfloat16)
    
    probs = jax.nn.softmax(logits)
    entropy_before = entropy(probs)
    entropy_after = entropy(final_probs)
    
    top_k_before = jnp.sort(probs, axis=-1)[...,-5:]
    top_k_after = jnp.sort(final_probs, axis=-1)[...,-5:]
    
    # Get probabilities for the selected token and ensure they're scalar
    token_prob_before = probs[0, token_id].squeeze()  # Take first batch element and squeeze
    token_prob_after = final_probs[0, token_id].squeeze()
    
    # Ensure all values are scalar
    return {
        'entropy_before': float(entropy_before[0]),  # Take first batch element
        'entropy_after': float(entropy_after[0]),
        'entropy_diff': float(entropy_after[0] - entropy_before[0]),
        'top_k_before': top_k_before[0],  # Keep array for top-k
        'top_k_after': top_k_after[0],
        'token_prob_before': float(token_prob_before),
        'token_prob_after': float(token_prob_after),
        'prob_ratio': float(token_prob_after / (token_prob_before + EPS))
    }

def test_ads_generation(params, apply_fn, input_ids, tokenizer, vocab_size):
    """Test function for ADS generation with detailed analysis."""
    try:
        prng_key = jax.random.PRNGKey(0)
        
        # Validate input shape
        if len(input_ids.shape) != 2:
            input_ids = input_ids[None, :]
            
        print("\nTesting ADS generation...")
        print(f"Input shape: {input_ids.shape}")
        
        # Test different sequence lengths
        for max_len in [20, 50]:
            print(f"\n{'='*40}")
            print(f"Testing sequence length: {max_len}")
            print(f"{'='*40}")
            
            # Regular sampling
            vanilla_start = time.time()
            vanilla_key, ads_key = jax.random.split(prng_key)
            vanilla_output = generate_sequence(
                params, apply_fn, input_ids, vanilla_key, 
                max_length=max_len, temperature=1e-3
            )
            vanilla_time = time.time() - vanilla_start
            
            # ADS sampling
            ads_start = time.time()
            ads_output = generate_sequence_ads(
                params, apply_fn, input_ids, ads_key, vocab_size,
                max_length=max_len, config=DEFAULT_ADS_CONFIG
            )
            ads_time = time.time() - ads_start
            
            # Print timing comparison
            print("\nTiming Comparison:")
            print(f"Vanilla sampling: {vanilla_time:.3f}s ({max_len/vanilla_time:.1f} tokens/s)")
            print(f"ADS sampling: {ads_time:.3f}s ({max_len/ads_time:.1f} tokens/s)")
            print(f"Overhead: {((ads_time/vanilla_time) - 1)*100:.1f}%")
            
            # Print outputs
            print("\nOutput Comparison:")
            vanilla_text = tokenizer.decode(vanilla_output[0])
            ads_text = tokenizer.decode(ads_output[0])
            
            print("\nVanilla sampling:")
            print(vanilla_text[:100] + "..." if len(vanilla_text) > 100 else vanilla_text)
            print("\nADS sampling:")
            print(ads_text[:100] + "..." if len(ads_text) > 100 else ads_text)
            
            # Compare token overlap
            overlap = jnp.mean(vanilla_output == ads_output)
            print(f"\nToken overlap: {overlap*100:.1f}%")
            
            # Get some example statistics from a single forward pass
            outputs = apply_fn(params, input_ids)
            logits = outputs.logits[:, -1, :]
            
            # Initialize ADS state for analysis
            ads_state = initialize_state(input_ids.shape[0], vocab_size, DEFAULT_ADS_CONFIG)
            new_state, token_id = adaptive_dirichlet_step(prng_key, ads_state, logits, DEFAULT_ADS_CONFIG)
            
            # Get probabilities after ADS transformation
            final_probs = jax.nn.softmax(logits / new_state.emwa_temp[:, None])
            
            # Analyze sampling statistics
            stats = analyze_sampling_stats(logits, final_probs, token_id)
            
            print("\nSampling Statistics:")
            for key, value in stats.items():
                if isinstance(value, (float, int)):
                    print(f"{key}: {value:.3f}")
                elif hasattr(value, 'shape') and len(value.shape) > 0:  # Array values
                    print(f"{key}:\n{value}")
                else:
                    print(f"{key}: {value}")
            
            # Get new PRNG key for next iteration
            prng_key = jax.random.fold_in(prng_key, 0)
            
    except Exception as e:
        print(f"Error during ADS generation: {str(e)}")
        print("Debug info:")
        if 'stats' in locals():
            print("Stats keys:", stats.keys())
            for k, v in stats.items():
                print(f"{k}: type={type(v)}")
        raise

@partial(jax.jit, static_argnames=('apply_fn', 'max_length'))
def generate_sequence(params, apply_fn, input_ids, key, max_length=20, temperature=1.0):
    """Generate sequence using regular sampling."""
    batch_size = input_ids.shape[0]
    
    # Initialize context buffer with input_ids
    context_buffer = jnp.zeros((batch_size, 128), dtype=jnp.int32)
    context_buffer = context_buffer.at[:, -input_ids.shape[1]:].set(input_ids)
    
    def scan_fn(carry, _):
        context, key = carry
        next_key, sample_key = jax.random.split(key)
        
        # Generate next token using vanilla sampling
        next_token = get_next_token(params, apply_fn, context, sample_key, temperature)
        
        # Shift context buffer and add new token
        new_context = jnp.roll(context, shift=-1, axis=1)
        new_context = new_context.at[:, -1].set(next_token)
        
        return (new_context, next_key), next_token
    
    # Run the scan
    _, output_tokens = jax.lax.scan(
        scan_fn,
        (context_buffer, key),
        jnp.arange(max_length)
    )
    
    return output_tokens.T

def main():
    print("Starting model load...")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Tokenizer loaded successfully")
    
    # Load config
    print("Loading config...")
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    print("Config loaded successfully")
    
    print("Loading pretrained model...")
    # Let AutoModelForCausalLM pick the right model class
    model = FlaxAutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        cache_dir=CACHE_DIR,
        dtype=jnp.bfloat16,
        _do_init=True,
    )
    
    # Convert all parameters except lm_head to bfloat16
    def convert_dtype(path, x):
        if len(path) >= 2 and path[0] == 'params' and path[1] != 'lm_head':
            return x.astype(jnp.bfloat16)
        return x
        
    model.params = jax.tree_util.tree_map_with_path(convert_dtype, model.params)
    
    # Print simplified model info
    print("\nModel Info:")
    print(f"- Model class: {model.__class__.__name__}")
    print(f"- Model dtype: {model.dtype}")
    print(f"- Vocab size: {config.vocab_size:,}")
    
    # Get model components
    params = model.params
    apply_fn = create_model_fn(model, pad_token_id=tokenizer.pad_token_id)
    
    # After model is loaded, get vocab size
    vocab_size = config.vocab_size
    
    # Prepare input
    prompt = "Once upon a time"
    tokenizer_output = tokenizer(prompt, return_tensors="jax")
    input_ids = tokenizer_output['input_ids']
    
    # Test ADS generation
    test_ads_generation(params, apply_fn, input_ids, tokenizer, vocab_size)

if __name__ == "__main__":
    main()