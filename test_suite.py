import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, AutoConfig
from functools import partial
import time
from dir_sampler import (
    adaptive_dirichlet_step, 
    initialize_state, 
    ADSState
)
from config import ADSConfig

# Constants
CACHE_DIR = '/home/cloudforest/Weights'
MODEL_NAME = "meta-llama/Llama-3.2-1B"

# Add default ADS config after the existing constants
DEFAULT_ADS_CONFIG = ADSConfig(
    emwa_logp_base=1.1,
    emwa_logp_exp_factor=2.0,
    emwa_dir_coeff=0.1,
    emwa_temp_coeff=0.1,
    emwa_dir_ent_coeff=0.1,
    entropy_rate_scaffold_coeff=0.1,
    entropy_rate_naked_coeff=0.1,
    token_cross_ent_scaffold_coeff=0.1,
    perturb_base_coeff=0.9,
    perturb_exp_coeff=2.0,
    probs_ent_offset=0.0,
    dir_ent_offset=0.0,
    entropy_a=1.0,
    entropy_b=0.0,
    entropy_c=0.0,
    entropy_d=0.0,
    dirichlet_d=1.0,
    dirichlet_e=0.0,
    token_cross_ent_naked_coeff=0.1
)

def create_model_fn(model):
    """Create a wrapper function that handles the model call correctly."""
    @jax.jit
    def model_fn(params, input_ids):
        # Get current sequence length and calculate padding
        batch_size, seq_len = input_ids.shape
        pad_length = 128 - seq_len  # 128 is our max sequence length
        
        # Pad input_ids
        padded_input_ids = jnp.pad(
            input_ids,
            ((0, 0), (0, pad_length)),
            mode='constant',
            constant_values=0
        )
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = jnp.pad(
            jnp.ones_like(input_ids),
            ((0, 0), (0, pad_length)),
            mode='constant',
            constant_values=0
        )
        
        # Create position IDs
        position_ids = jnp.pad(
            jnp.arange(seq_len)[None, :],
            ((0, 0), (0, pad_length)),
            mode='constant',
            constant_values=0
        )
        
        # Call model with padded inputs
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

def get_next_token(params, apply_fn, input_ids, key, temperature=1.0):
    """Get next token using sampling."""
    @jax.jit
    def _get_next_token(params, input_ids, key):
        outputs = apply_fn(params, input_ids)
        logits = outputs.logits[:, -1, :] / temperature
        return jax.random.categorical(key, logits, axis=-1)
    return _get_next_token(params, input_ids, key)

@partial(jax.jit, static_argnames=('apply_fn', 'max_length', 'config', 'vocab_size'))
def generate_sequence_ads(params, apply_fn, input_ids, key, vocab_size, max_length=20, config=DEFAULT_ADS_CONFIG):
    """Generate sequence using adaptive Dirichlet sampling with efficient JAX implementation."""
    batch_size = input_ids.shape[0]
    
    # Initialize context buffer with input_ids
    context_buffer = jnp.zeros((batch_size, 128), dtype=jnp.int32)
    context_buffer = context_buffer.at[:, -input_ids.shape[1]:].set(input_ids)
    
    # Initialize ADS state
    ads_state = initialize_state(batch_size, vocab_size, dtype=jnp.float32)
    
    def scan_fn(carry, _):
        context, key, state = carry
        next_key, sample_key = jax.random.split(key)
        
        # Get model outputs
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

def test_ads_generation(params, apply_fn, input_ids, tokenizer, vocab_size):
    """Test function for ADS generation with proper error handling."""
    try:
        key = jax.random.PRNGKey(0)
        
        # Validate input shape
        if len(input_ids.shape) != 2:
            input_ids = input_ids[None, :]  # Add batch dimension if needed
            
        print("\nTesting ADS generation...")
        print(f"Input shape: {input_ids.shape}")
        
        # Test different sequence lengths
        for max_len in [20, 50]:
            start_time = time.time()
            
            # Compile the function (first run will include compilation time)
            output_ids = generate_sequence_ads(
                params, apply_fn, input_ids, key, vocab_size,
                max_length=max_len, config=DEFAULT_ADS_CONFIG
            )
            compile_time = time.time() - start_time
            
            # Run again to measure actual execution time
            start_time = time.time()
            output_ids = generate_sequence_ads(
                params, apply_fn, input_ids, key, vocab_size,
                max_length=max_len, config=DEFAULT_ADS_CONFIG
            )
            execution_time = time.time() - start_time
            
            print(f"\nLength {max_len}:")
            print(f"Compilation time: {compile_time:.3f}s")
            print(f"Execution time: {execution_time:.3f}s")
            
            # Verify output
            output_text = tokenizer.decode(output_ids[0])
            print(f"Sample output ({len(output_text)} chars):")
            print(output_text[:100] + "..." if len(output_text) > 100 else output_text)
            
    except Exception as e:
        print(f"Error during ADS generation: {str(e)}")
        raise

def main():
    print("Starting model load...")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    print("Tokenizer loaded successfully")
    
    # Load config
    print("Loading config...")
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    config.max_position_embeddings = 128  # Set this to match our expected sequence length
    config.vocab_size = config.vocab_size  # Make sure this is set in the config
    print("Config loaded and modified successfully")
    
    print("Loading pretrained model...")
    model = FlaxAutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        cache_dir=CACHE_DIR,
        dtype=jnp.float32,  # Explicitly set dtype to float32
        _do_init=True
    )
    print("Model loaded successfully")
    
    # Get model components
    params = model.params
    apply_fn = create_model_fn(model)  # Use our wrapper
    
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