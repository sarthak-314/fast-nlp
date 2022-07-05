from flax.serialization import from_bytes, to_bytes
from flax import jax_utils, struct, traverse_util
from flax.training.common_utils import shard
from flax.training import train_state
import flax.linen as nn
import flax 

import jax.numpy as jnp
import jax

import optax