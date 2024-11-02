import jax.numpy as jnp
from typing import NamedTuple
from dataclasses import dataclass
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
    dirichlet_d: float
    dirichlet_e: float
    token_cross_ent_naked_coeff: float

    def __hash__(self):
        return hash(tuple(getattr(self, field.name) for field in self.__dataclass_fields__.values()))

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

CACHE_DIR = '/home/cloudforest/Weights'
MODEL_NAME = "meta-llama/Llama-3.2-1B"
HF_TOKEN = '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'
