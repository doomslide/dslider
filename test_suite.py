import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, AutoConfig
from functools import partial
import time
# Constants
CACHE_DIR = '/home/cloudforest/Weights'
MODEL_NAME = "meta-llama/Llama-3.2-1B"


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

def generate_sequence(params, apply_fn, input_ids, key, max_length=20):
    """Generate sequence using scan."""
    batch_size = input_ids.shape[0]
    
    # Initialize context buffer with input_ids
    
    # Create a fixed-size context buffer (padded to 128)
    context_buffer = jnp.zeros((batch_size, 128), dtype=jnp.int32)
    context_buffer = context_buffer.at[:, -input_ids.shape[1]:].set(input_ids)
    
    # Pre-allocate output sequence
    output_sequence = jnp.zeros((batch_size, max_length), dtype=jnp.int32)
    
    def scan_fn(carry, idx):
        context, key = carry
        next_key, sample_key = jax.random.split(key)
        
        # Generate next token using current context
        next_token = get_next_token(params, apply_fn, context, sample_key)
        
        # Shift context left by 1 and add new token at end
        new_context = jnp.roll(context, shift=-1, axis=1)
        new_context = new_context.at[:, -1].set(next_token)
        
        return (new_context, next_key), next_token
    
def generate_sequence(params, apply_fn, input_ids, key, max_length=20):
    """Generate sequence using scan."""
    batch_size = input_ids.shape[0]
    
    # Initialize context buffer with input_ids
    
    # Create a fixed-size context buffer (padded to 128)
    context_buffer = jnp.zeros((batch_size, 128), dtype=jnp.int32)
    context_buffer = context_buffer.at[:, -input_ids.shape[1]:].set(input_ids)
    
    # Pre-allocate output sequence
    output_sequence = jnp.zeros((batch_size, max_length), dtype=jnp.int32)
    
    def scan_fn(carry, idx):
        context, key = carry
        next_key, sample_key = jax.random.split(key)
        
        # Generate next token using current context
        next_token = get_next_token(params, apply_fn, context, sample_key)
        
        # Shift context left by 1 and add new token at end
        new_context = jnp.roll(context, shift=-1, axis=1)
        new_context = new_context.at[:, -1].set(next_token)
        
        return (new_context, next_key), next_token
    
    # Initialize carry
    init_carry = (context_buffer, key)
    
    # Run the scan
    _, output_tokens = jax.lax.scan(
        scan_fn,
        init_carry,
        jnp.arange(max_length)
    )
    
    # Reshape output tokens into sequence
    output_sequence = output_tokens.T
    
    return output_sequence
    # Initialize carry
    init_carry = (context_buffer, key)
    
    # Run the scan
    _, output_tokens = jax.lax.scan(
        scan_fn,
        init_carry,
        jnp.arange(max_length)
    )
    
    # Reshape output tokens into sequence
    output_sequence = output_tokens.T
    
    return output_sequence

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
    
    # Prepare input
    prompt = "Once upon a time"
    
    # Tokenize input and extract input_ids tensor
    tokenizer_output = tokenizer(prompt, return_tensors="jax")
    input_ids = tokenizer_output['input_ids']
    print(f"\nGenerating text for prompt: {prompt}")
    print(type(input_ids))
    print(input_ids.shape)
    
    # Test the function before JIT
    print("\nTesting model call...")
    test_output = apply_fn(params, input_ids)
    print("Test successful!")
    
    # Test the function with explicit arguments
    print("\nTesting with explicit arguments...")
    # Pad input_ids for the test
    batch_size, seq_len = input_ids.shape
    pad_length = 128 - seq_len
    
    padded_input_ids = jnp.pad(
        input_ids,
        ((0, 0), (0, pad_length)),
        mode='constant',
        constant_values=0
    )
    
    attention_mask = jnp.pad(
        jnp.ones_like(input_ids),
        ((0, 0), (0, pad_length)),
        mode='constant',
        constant_values=0
    )
    
    position_ids = jnp.pad(
        jnp.arange(seq_len)[None, :],
        ((0, 0), (0, pad_length)),
        mode='constant',
        constant_values=0
    )
    
    test_output2 = model.module.apply(
        {'params': params},
        padded_input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        deterministic=True,
        method=model.module.__call__
    )
    print("Second test successful!")
    
    # Performance testing for text generation
    print("\nRunning performance tests...")
    
    # Test with different sequence lengths
    lengths = [20, 50, 100]
    for max_len in lengths:
        key = jax.random.PRNGKey(0)
        start_time = time.time()
        output_ids = generate_sequence(params, apply_fn, input_ids, key, max_length=max_len)
        end_time = time.time()
        print(f"\nGeneration time for length {max_len}: {end_time - start_time:.3f} seconds")
        
        # Memory usage
        usage = jax.device_get(output_ids).nbytes / 1024  # KB
        print(f"Memory usage: {usage:.2f} KB")
        
        # Tokens per second
        tokens_generated = output_ids.shape[1] - input_ids.shape[1]
        tokens_per_sec = tokens_generated / (end_time - start_time)
        print(f"Generation speed: {tokens_per_sec:.2f} tokens/second")
        
        # Sample output for verification
        output_text = tokenizer.decode(output_ids[0])
        print(f"Sample output ({len(output_text)} chars):")
        print(output_text[:100] + "..." if len(output_text) > 100 else output_text)

if __name__ == "__main__":
    main()