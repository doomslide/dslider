import jax.numpy as jnp
from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class
import jax 

# Constants
MIN_TEMP = 1e-4
MAX_TEMP = 1e4
EPS = 1e-8

@dataclass(frozen=True)
class OutlierThreshold:
    bilinear: jnp.ndarray  # Shape (4, 4)
    linear_state_ent: jnp.ndarray  # Shape (4,)
    linear_state_std: jnp.ndarray  # Shape (4,)
    linear_naked_ent: float
    linear_naked_std: float
    linear_naked_varent: float
    bias: float
    
    def tree_flatten(self):
        """For JAX pytree handling"""
        children = (self.bilinear, self.linear_state_ent, self.linear_state_std, 
                   self.linear_naked_ent, self.linear_naked_std, self.linear_naked_varent, self.bias)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """For JAX pytree handling"""
        return cls(*children)

@dataclass(frozen=True)
class ArgmaxThreshold:
    weight: float
    bias: float
    
    def tree_flatten(self):
        return (self.weight, self.bias), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@dataclass(frozen=True)
class DirichletThreshold:
    weight: float
    bias: float
    
    def tree_flatten(self):
        return (self.weight, self.bias), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@dataclass(frozen=True)
class TargetEntropy:
    linear: jnp.ndarray  # Shape (4,)
    linear_inv_temp: jnp.ndarray  # Shape (batch_size,)
    bias: float
    
    def tree_flatten(self):
        return (self.linear, self.linear_inv_temp, self.bias), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@dataclass(frozen=True, eq=True)
class ADSConfig:
    # EMWA coefficients
    emwa_logp_base: float
    emwa_logp_exp_factor: float
    emwa_dir_coeff: float
    emwa_temp_coeff: float
    emwa_dir_ent_coeff: float
    emwa_ent_scaffold_coeff: float
    emwa_varent_scaffold_coeff: float
    emwa_ent_naked_coeff: float
    emwa_varent_naked_coeff: float
    emwa_topk_ent_naked_coeff: float
    
    # Token cross entropy coefficients
    token_cross_ent_scaffold_coeff: float
    token_cross_ent_naked_coeff: float
    token_cross_var_scaffold_coeff: float
    token_cross_var_naked_coeff: float
    
    # Dirichlet parameters
    perturb_base_coeff: float
    perturb_exp_coeff: float
    dirichlet_support: jnp.ndarray
    
    # Threshold parameters
    outlier_threshold: OutlierThreshold
    argmax_threshold: ArgmaxThreshold
    dirichlet_threshold: DirichletThreshold
    target_entropy: TargetEntropy
    
    
    # Token outlier
    outlier_topk: int
    
    def __hash__(self):
        """Custom hash that handles JAX arrays."""
        hashable_items = []
        for field in self.__dataclass_fields__.values():
            value = getattr(self, field.name)
            if isinstance(value, (jnp.ndarray, jax.Array)):
                # Hash array data and shape for arrays
                hashable_items.append(hash((value.shape, value.dtype)))
            else:
                hashable_items.append(hash(value))
        return hash(tuple(hashable_items))
    
    def tree_flatten(self):
        """For JAX pytree handling"""
        return (tuple(getattr(self, field.name) for field in self.__dataclass_fields__.values()), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """For JAX pytree handling"""
        return cls(*children)


register_pytree_node_class(ADSConfig)
register_pytree_node_class(OutlierThreshold)
register_pytree_node_class(ArgmaxThreshold)
register_pytree_node_class(DirichletThreshold)
register_pytree_node_class(TargetEntropy)

# Default config values
DEFAULT_ADS_CONFIG = ADSConfig(
    # EMWA coefficients
    emwa_logp_base=1.5, 
    emwa_logp_exp_factor=2.5,  
    emwa_dir_coeff=0.2, 
    emwa_temp_coeff=0.15, 
    emwa_dir_ent_coeff=0.15,  
    emwa_ent_scaffold_coeff=0.15,
    emwa_varent_scaffold_coeff=0.15,
    emwa_ent_naked_coeff=0.15,
    emwa_varent_naked_coeff=0.15,
    emwa_topk_ent_naked_coeff=0.15,
    
    # Token cross entropy coefficients
    token_cross_ent_scaffold_coeff=0.15, 
    token_cross_ent_naked_coeff=0.15,
    token_cross_var_scaffold_coeff=0.15,
    token_cross_var_naked_coeff=0.15,
    
    # Dirichlet parameters
    perturb_base_coeff=0.95, 
    perturb_exp_coeff=2.5, 
    dirichlet_support=jnp.arange(32000), 
    
    # Threshold parameters
    outlier_threshold=OutlierThreshold(
        bilinear=jnp.eye(4) * 0.15,  # Increased sensitivity
        linear_state_ent=jnp.ones(4) * 0.15,
        linear_state_std=jnp.ones(4) * 0.15,
        linear_naked_ent=0.15,
        linear_naked_std=0.15,
        linear_naked_varent=0.15,
        bias=0.1  # Added small positive bias
    ),
    argmax_threshold=ArgmaxThreshold(
        weight=1.2,  # Increased from 1.0
        bias=0.1  # Added small positive bias
    ),
    dirichlet_threshold=DirichletThreshold(
        weight=1.2,  # Increased from 1.0
        bias=0.1  # Added small positive bias
    ),
    target_entropy=TargetEntropy(
        linear=jnp.ones(4) * 0.15,
        linear_inv_temp=jnp.ones(1) * 1.2,  # Increased from 1.0
        bias=0.1  # Added small positive bias
    ),
    
    # Token outlier parameters
    outlier_topk=3,
)

CACHE_DIR = '/home/cloudforest/Weights'
MODEL_NAME = "gpt2"
HF_TOKEN = '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'
