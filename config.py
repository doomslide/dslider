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
        arrays = [self.bilinear, self.linear_state_ent, self.linear_state_std]
        aux_data = {
            'linear_naked_ent': self.linear_naked_ent,
            'linear_naked_std': self.linear_naked_std,
            'linear_naked_varent': self.linear_naked_varent,
            'bias': self.bias
        }
        return arrays, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, arrays):
        """For JAX pytree handling"""
        return cls(
            bilinear=arrays[0],
            linear_state_ent=arrays[1],
            linear_state_std=arrays[2],
            **aux_data
        )

    def __hash__(self):
        """Static hash implementation"""
        return hash((
            'OutlierThreshold',
            self.bilinear.shape,
            str(self.bilinear.dtype),
            self.linear_state_ent.shape,
            str(self.linear_state_ent.dtype),
            self.linear_state_std.shape,
            str(self.linear_state_std.dtype),
            self.linear_naked_ent,
            self.linear_naked_std,
            self.linear_naked_varent,
            self.bias
        ))

@dataclass(frozen=True)
class ArgmaxThreshold:
    weight: float
    bias: float
    
    def tree_flatten(self):
        """For JAX pytree handling"""
        aux_data = {
            'weight': self.weight,
            'bias': self.bias
        }
        return [], aux_data  # No arrays, just auxiliary data

    @classmethod
    def tree_unflatten(cls, aux_data, arrays):
        """For JAX pytree handling"""
        return cls(**aux_data)

    def __hash__(self):
        return hash((self.weight, self.bias))

@dataclass(frozen=True)
class DirichletThreshold:
    weight: float
    bias: float
    
    def tree_flatten(self):
        """For JAX pytree handling"""
        aux_data = {
            'weight': self.weight,
            'bias': self.bias
        }
        return [], aux_data  # No arrays, just auxiliary data

    @classmethod
    def tree_unflatten(cls, aux_data, arrays):
        """For JAX pytree handling"""
        return cls(**aux_data)

    def __hash__(self):
        return hash((self.weight, self.bias))

@dataclass(frozen=True)
class TargetEntropy:
    linear: jnp.ndarray  # Shape (4,)
    linear_inv_temp: jnp.ndarray  # Shape (batch_size,)
    bias: float
    
    def tree_flatten(self):
        arrays = [self.linear, self.linear_inv_temp]
        aux_data = {'bias': self.bias}
        return arrays, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, arrays):
        return cls(
            linear=arrays[0],
            linear_inv_temp=arrays[1],
            bias=aux_data['bias']
        )

    def __hash__(self):
        """Static hash implementation"""
        return hash((
            'TargetEntropy',
            self.linear.shape,
            str(self.linear.dtype),
            self.linear_inv_temp.shape,
            str(self.linear_inv_temp.dtype),
            self.bias
        ))

@dataclass(frozen=True, eq=True)
class DSConfig:
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
        """Static hash implementation that avoids hashing array values"""
        hashable_items = []
        for field in self.__dataclass_fields__.values():
            value = getattr(self, field.name)
            if isinstance(value, (jnp.ndarray, jax.Array)):
                # Only hash shape and dtype for arrays
                hashable_items.append(hash((str(field.name), value.shape, str(value.dtype))))
            elif isinstance(value, (OutlierThreshold, ArgmaxThreshold, DirichletThreshold, TargetEntropy)):
                # Use the class's hash method
                hashable_items.append(hash(value))
            else:
                # For primitive types
                hashable_items.append(hash((str(field.name), value)))
        return hash(tuple(hashable_items))

    def tree_flatten(self):
        """Improved flattening for JAX pytree"""
        arrays = []
        aux_data = {}
        
        for field in self.__dataclass_fields__.values():
            value = getattr(self, field.name)
            if isinstance(value, (jnp.ndarray, jax.Array)):
                arrays.append(value)
            elif isinstance(value, (OutlierThreshold, ArgmaxThreshold, DirichletThreshold, TargetEntropy)):
                nested_arrays, nested_aux = value.tree_flatten()
                arrays.extend(nested_arrays)
                aux_data[field.name] = (type(value), nested_aux)
            else:
                aux_data[field.name] = value
                
        return arrays, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, arrays):
        """Improved unflattening for JAX pytree"""
        array_idx = 0
        field_values = {}
        
        for field_name, field in cls.__dataclass_fields__.items():
            if field_name in aux_data:
                value = aux_data[field_name]
                if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], type):
                    # Reconstruct nested dataclass
                    klass, nested_aux = value
                    if klass in (OutlierThreshold, TargetEntropy):
                        n_arrays = 3 if klass == OutlierThreshold else 2
                        nested_arrays = arrays[array_idx:array_idx + n_arrays]
                        array_idx += n_arrays
                        field_values[field_name] = klass.tree_unflatten(nested_aux, nested_arrays)
                    else:
                        # For ArgmaxThreshold and DirichletThreshold which have no arrays
                        field_values[field_name] = klass(**nested_aux)
                else:
                    field_values[field_name] = value
            else:
                field_values[field_name] = arrays[array_idx]
                array_idx += 1
                
        return cls(**field_values)

register_pytree_node_class(DSConfig)
register_pytree_node_class(OutlierThreshold)
register_pytree_node_class(ArgmaxThreshold)
register_pytree_node_class(DirichletThreshold)
register_pytree_node_class(TargetEntropy)

# Default config values
DEFAULT_DS_CONFIG = DSConfig(
    # EMWA coefficients
    emwa_logp_base=1.5, 
    emwa_logp_exp_factor=2.5,  
    emwa_dir_coeff=0.2, 
    emwa_temp_coeff=1, 
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
    dirichlet_support=jnp.arange(50257), # this is gpt2 vocab size
    
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

CACHE_DIR = '.cache'
MODEL_NAME = "gpt2"
