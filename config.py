import jax.numpy as jnp
from typing import NamedTuple
from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class

# Constants
MIN_TEMP = 1e-4
MAX_TEMP = 1e4
EPS = 1e-8

@dataclass(frozen=True, eq=True)
class ADSConfig:
    emwa_logp_base: float
    emwa_logp_exp_factor: float
    emwa_dir_coeff: float
    emwa_temp_coeff: float
    emwa_dir_ent_coeff: float
    entropy_rate_scaffold_coeff: float
    entropy_rate_naked_coeff: float
    token_cross_ent_scaffold_coeff: float
    perturb_base_coeff: float
    perturb_exp_coeff: float
    probs_ent_offset: float
    dir_ent_offset: float
    entropy_a: float
    entropy_b: float
    entropy_c: float
    entropy_d: float
    dirichlet_a: float
    dirichlet_b: float
    token_outlier_threshold: jnp.ndarray 
    token_cross_ent_naked_coeff: float
    token_outlier_k: int # still very interesting for k=1
    token_outlier_emwa_weight: float
    token_outlier_emwa_bias: float
    token_outlier_a: float 
    token_outlier_b: float 
    token_outlier_c: float 
    token_outlier_d: float 
    token_outlier_threshold_bias: float

    def __hash__(self):
        return hash(tuple(getattr(self, field.name) for field in self.__dataclass_fields__.values()))
    
    def tree_flatten(self):
        """For JAX pytree handling"""
        return (tuple(getattr(self, field.name) for field in self.__dataclass_fields__.values()), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """For JAX pytree handling"""
        return cls(*children)

register_pytree_node_class(ADSConfig)

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
    dirichlet_a=1.0,
    dirichlet_b=0.0,
    token_cross_ent_naked_coeff=0.1,
    toekn_outlier_k=3,
    token_outlier_emwa_weight=0.6,
    token_outlier_emwa_bias=2.0,
    token_outlier_a=0.4,
    token_outlier_b=0.6,
    token_outlier_c=0.5,
    token_outlier_d=0.2,
    token_outlier_threshold_bias=0.3,
)

CACHE_DIR = '/home/cloudforest/Weights'
MODEL_NAME = "HuggingFaceTB/SmolLM-360M"
HF_TOKEN = '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'
