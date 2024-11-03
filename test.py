import warnings
warnings.filterwarnings("ignore", message=".*unhashable type.*")
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, AutoConfig
from functools import partial
from dslider import (
    adaptive_dirichlet_step, 
    initialize_state
)
from config import DEFAULT_DS_CONFIG, CACHE_DIR, MODEL_NAME
def create_model_fn(model, pad_token_id):
    @jax.jit
    def model_fn(params, input_ids):
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
    @partial(jax.jit, static_argnames=('apply_fn',))
    def _get_next_token(params, apply_fn, input_ids, key):
        outputs = apply_fn(params, input_ids)
        logits = outputs.logits[:, -1, :] / temperature
        return jax.random.categorical(key, logits, axis=-1)
    return _get_next_token(params, apply_fn, input_ids, key)

@partial(jax.jit, static_argnames=('apply_fn', 'max_length', 'config'))
def generate_sequence_dslider(
    params, 
    apply_fn, 
    input_ids, 
    key, 
    vocab_size, 
    max_length=20, 
    config=None, 
    dtype=jnp.bfloat16
):
    if config is None:
        config = DEFAULT_DS_CONFIG
        
    batch_size = input_ids.shape[0]
    context_buffer = jnp.zeros((batch_size, 128), dtype=jnp.int32)
    context_buffer = context_buffer.at[:, -input_ids.shape[1]:].set(input_ids)
    state = initialize_state(batch_size, vocab_size, config, dtype=dtype)

    def scan_fn(carry, _):
        context, key, state = carry
        next_key, sample_key = jax.random.split(key)
        outputs = apply_fn(params, context)
        logits = outputs.logits[:, -1, :]
        new_state, output_token, *_ = adaptive_dirichlet_step(sample_key, state, logits, config)
        new_context = jnp.roll(context, shift=-1, axis=1)
        new_context = new_context.at[:, -1].set(output_token)
        return (new_context, next_key, new_state), output_token
    
    _, output_tokens = jax.lax.scan(
        scan_fn,
        (context_buffer, key, state),
        jnp.arange(max_length)
    )
    return output_tokens.T

def generate_sequence(params, apply_fn, input_ids, key, max_length=20, temperature=1.0):
    @partial(jax.jit, static_argnames=('apply_fn', 'max_length'))
    def _generate_sequence(params, apply_fn, input_ids, key, max_length):
        batch_size = input_ids.shape[0]
        context_buffer = jnp.zeros((batch_size, 128), dtype=jnp.int32)
        context_buffer = context_buffer.at[:, -input_ids.shape[1]:].set(input_ids)
        
        def scan_fn(carry, _):
            context, key = carry
            next_key, sample_key = jax.random.split(key)
            outputs = apply_fn(params, context)
            logits = outputs.logits[:, -1, :] / temperature
            next_token = jax.random.categorical(sample_key, logits, axis=-1)
            new_context = jnp.roll(context, shift=-1, axis=1)
            new_context = new_context.at[:, -1].set(next_token)
            return (new_context, next_key), next_token
        
        _, output_tokens = jax.lax.scan(
            scan_fn,
            (context_buffer, key),
            jnp.arange(max_length)
        )
        return output_tokens.T
    
    return _generate_sequence(params, apply_fn, input_ids, key, max_length)

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    model = FlaxAutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        cache_dir=CACHE_DIR,
        dtype=jnp.bfloat16,
        _do_init=True,
    ) 
    
    params = model.params
    apply_fn = create_model_fn(model, pad_token_id=tokenizer.pad_token_id)
    vocab_size = config.vocab_size
    
    prompts = ["Once upon a time"]
    input_ids = tokenizer(prompts, return_tensors="jax", padding=True)['input_ids']
    
    prng_key = jax.random.PRNGKey(0)
    vanilla_key, dslider_key = jax.random.split(prng_key)
    
    vanilla_output = generate_sequence(params, apply_fn, input_ids, vanilla_key, max_length=50, temperature=1e-3)
    dslider_output = generate_sequence_dslider(params, apply_fn, input_ids, dslider_key, vocab_size, max_length=50)

    print("\nvanilla output:")
    print(tokenizer.decode(vanilla_output[0]))
    print("\ndslider output:")
    print(tokenizer.decode(dslider_output[0]))

if __name__ == "__main__":
    main()