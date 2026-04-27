from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax._src.mesh import thread_resources
from jax.ad_checkpoint import checkpoint_name
import nki.language as nl

# Monkey-patch to fix unhashable list bug in NKI
import neuronxcc.nki._jax as nki_jax

def _patched_hash(self):
    k = self.kernel
    grid = tuple(k.grid) if isinstance(k.grid, list) else k.grid
    return hash((k.func, grid, k.opts))

nki_jax.JaxTraceResult.__hash__ = _patched_hash

from nkilib.core.moe.moe_cte.bwmm_shard_on_I import (
    blockwise_mm_baseline_shard_intermediate as blockwise_mm_nki,
    SkipMode as FwdSkipMode,
)
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from nkilib.experimental.moe.bwd.moe_bwd_parameters import SkipMode as BwdSkipMode
from nkilib.experimental.moe.bwd.blockwise_mm_backward import blockwise_mm_bwd as blockwise_mm_bwd_nki

jax.tree_util.register_dataclass(
    BwdSkipMode,
    data_fields=[],
    meta_fields=["skip_token", "skip_weight"]
)

jax.tree_util.register_dataclass(
    FwdSkipMode,
    data_fields=[],
    meta_fields=["skip_token", "skip_weight"]
)

Tensor = jax.Array

def _backend():
    global_mesh = thread_resources.env.physical_mesh
    if len(global_mesh.devices):
        backend = global_mesh.devices.flat[0].platform
    else:
        backend = jax.default_backend()
    return backend

def can_use_blockwise_matmul_nki(
    hidden_size,
    intermediate_size_tp,
    block_size,
    glu_mlp,
):
    if _backend() != "neuron":
        return False
    if not glu_mlp:
        print("Blockwise NKI kernel incompatible with glu_mlp=False")
        return False
    if blockwise_mm_nki is None:
        print("Failed to load Blockwise NKI kernel.")
        return False
    return True


@partial(custom_vjp, nondiff_argnums=(6,))
def blockwise_mm(
    hidden_states: Tensor,
    expert_affinities_masked: Tensor,
    gate_up_weight: Tensor,
    down_proj_weight: Tensor,
    token_position_to_id: Tensor,
    block_to_expert: Tensor,
    block_size: int,
):
    out, _ = _blockwise_mm_fwd(hidden_states, expert_affinities_masked, gate_up_weight,
                            down_proj_weight, token_position_to_id, block_to_expert, block_size)
    return out

def _blockwise_mm_fwd(
    hidden_states: Tensor,
    expert_affinities_masked: Tensor,
    gate_up_weight: Tensor,
    down_proj_weight: Tensor,
    token_position_to_id: Tensor,
    block_to_expert: Tensor,
    block_size: int,
):
    orig_expert_affin_shape = expert_affinities_masked.shape
    
    with jax.named_scope("take_out_OG"):
        hidden_states = jnp.squeeze(hidden_states, axis=(0,1,))
        expert_affinities_masked = jnp.squeeze(expert_affinities_masked, axis=(0,1,))
        token_position_to_id = jnp.squeeze(token_position_to_id, axis=(0,1,))
        block_to_expert = jnp.squeeze(block_to_expert, axis=(0,1,))
        # Remap global expert IDs to local device indices
        # To prevent block_to_expert indices from OOB error
        min_expert_id = jnp.min(block_to_expert)
        block_to_expert = block_to_expert - min_expert_id
         
    with jax.named_scope("add padding"):
        padding_h = jnp.zeros((1, hidden_states.shape[1]), dtype=hidden_states.dtype)
        padding_e = jnp.zeros((1, expert_affinities_masked.shape[1]), dtype=expert_affinities_masked.dtype)
        hidden_states = jnp.concat([hidden_states, padding_h], axis=0)
        expert_affinities_masked = jnp.concat([expert_affinities_masked, padding_e], axis=0)
        expert_affinities_masked = jnp.reshape(expert_affinities_masked, (-1, 1))

    print("blockwise_mm_nki kernel is used")
    out, gate_up_activations_T, down_activations = blockwise_mm_nki[2](
        hidden_states,
        expert_affinities_masked,
        gate_up_weight,
        down_proj_weight,
        block_size,
        token_position_to_id,
        block_to_expert,
        activation_function=ActFnType.SiLU,
        skip_dma=FwdSkipMode(False, False),
        compute_dtype=nl.bfloat16,
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        checkpoint_activation=True,
    )

    down_activations = checkpoint_name(down_activations, "blockwise.down_activations")
    gate_up_activations_T = checkpoint_name(gate_up_activations_T, "blockwise.gate_up_activations_T")

    return out[None, None, None, :-1, :], (hidden_states, expert_affinities_masked, orig_expert_affin_shape,
                gate_up_weight, down_proj_weight, down_activations, gate_up_activations_T,
                token_position_to_id, block_to_expert)

def _blockwise_mm_bwd(
    block_size,
    res,
    grad_output
):
    (hidden_states, expert_affinities_masked, orig_expert_affin_shape, gate_up_proj_weight,
     down_proj_weight, down_activations, gate_up_activations_T,
     token_position_to_id, block_to_expert) = res
    T, H = hidden_states.shape

    with jax.named_scope("blockwise_backward"):
        grad_output = jnp.squeeze(grad_output, axis=(0,1,2))
        padding_h = jnp.zeros((1, hidden_states.shape[1]), dtype=hidden_states.dtype)
        grad_output = jnp.concat([grad_output, padding_h], axis=0)
        
        print("blockwise_mm_bwd_nki kernel is used")
        hidden_states_grad, affinities_grad, gate_up_proj_weight_grad, down_weight_grad = blockwise_mm_bwd_nki[2](
            hidden_states,
            expert_affinities_masked,
            gate_up_proj_weight,
            down_proj_weight,
            gate_up_activations_T,
            down_activations,
            token_position_to_id.astype(jnp.int32),
            block_to_expert.astype(jnp.int32),
            grad_output,
            block_size=block_size,
            skip_dma=BwdSkipMode(False, False),
        )
        
        sliced_tensor = hidden_states_grad[:-1, :]
        hidden_states_grad = sliced_tensor.reshape(1, 1, -1, H)
        
        affinities_grad = jnp.reshape(affinities_grad, (-1, orig_expert_affin_shape[-1]))
        affinities_grad = affinities_grad[:-1, :].reshape(1, 1, -1, orig_expert_affin_shape[-1])
    
    return (
        hidden_states_grad,
        affinities_grad,
        gate_up_proj_weight_grad,
        down_weight_grad,
        token_position_to_id,
        block_to_expert
    )

blockwise_mm.defvjp(_blockwise_mm_fwd, _blockwise_mm_bwd)
