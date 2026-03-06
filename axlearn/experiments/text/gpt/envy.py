# Copyright © 2024 Apple Inc.

"""Utilities to set up the 'Envy' MoE style model trainer configs.

Add MoE style model configs for the GPT model class.
- SwitchTransformer <https://arxiv.org/pdf/2101.03961>.
- Apple MoE <https://arxiv.org/pdf/2405.15052>
- Mistral 8x7B <https://arxiv.org/pdf/2401.04088>

We follow most of the practice in switch-transformer for MoE, however there are some key
differences:
- We do not use a T5-style model, but a GPT-style decoder-only model.
- We do not follow the optimizer settings in the paper, instead follow practice from Fuji and Gala.
- We use the same tokenizer as Fuji model classes.
- We increase the hidden dimension per head to 128 for better use of tensorcore in both TPU and GPU.
- We increase the sequence length to 8k instead of 512 in most of the T5 models,
    and increase global tokens/batch to 8M instead of 1M.
- We use rotary positional embeddings instead of the relative positional embeddings.
- We set other hyperparameters arbitrarily when unspecified by the paper.

We follow most of the practice in Mistral 8x7B, however there are some subtle differences:
- We don't use dropless routing in Megablocks, instead we use fixed capacity with drop and padding.
- We use the same tokenizer as Fuji model classes.

Architecture names follow apple varieties: Fuji, Gala, etc.
"""

import functools
from typing import Any, Literal, Sequence, Union, NamedTuple, List
import os
import jax
from axlearn.common.utils import (
    extended_checkpoint_policies,
    save_and_offload_only_these_names_regex,
)
from jax.ad_checkpoint import checkpoint_policies as jax_remat_policies
from axlearn.common.config import config_for_function
from axlearn.common import causal_lm, config
from axlearn.common.attention import (
    FusedGroupedQKVLinear,
    GroupedQueryAttention,
    RoFormerQKVLinear,
    GroupedQKVLinear,
    ScaleKey,
    ScaleQuery,
    TransformerLayer,
    StackedTransformerLayer,
    RepeatedTransformerLayer,
    RematRegexSavePatterns
)
from axlearn.common.base_layer import RematSpec
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import RMSNorm
from axlearn.common.mixture_of_experts import TransformerFeedForwardMoE, get_outer_batch_from_mesh, TopKGatingGather, TopKGatingGatherBlockwise, TopKGatingGatherBlockwiseV2
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.trainer_config_modifier import (
    ChainConfigModifier,
    GradientAccumulationModifier,
    MeshShapeModifier,
    RematSpecModifier,
    ModuleConfigModifier,
    PartitionSpecModifier
)
from axlearn.common.utils import HybridMeshShape, MeshShape, PartitionSpec, DataPartitionType
from axlearn.experiments.text.gpt.common import (
    MESH_AXIS_NAMES,
    STEP_DTYPE,
    SourceBuilder,
    adamw_decoupled_learner_config,
    evaler_config_dict,
    flash_attention_config,
    get_trainer_config_fn,
    make_config_name,
    mesh_shape_from_axes,
)
from axlearn.experiments.text.gpt.common import model_config as common_model_config
from axlearn.experiments.text.gpt.common import (
    mup_simple_adam_update_transformation,
    scaled_hidden_dim,
)
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

MODEL_SIZES = ("test", "Switch-Base", "Switch-Large", "Switch-XXL", "Mistral-8x7B", "Mistral-8x20B", "Mistral-toy", "Mistral-16x10B")

NUM_EXPERTS = {
    "test": 8,
    "Switch-Base": 128,
    "Switch-Large": 128,
    "Switch-XXL": 64,
    "Mistral-8x7B": 8,
    "Mistral-8x20B": 8,
    "Mistral-toy": 8,
    "Mistral-16x10B": 16,
}

# T5 uses 32128 vocab size, we make it 32768 for simplicity.
VOCAB_SIZE = 32 * 1024

MAX_SEQUENCE_LENGTH = {
    "test": 8192,
    "Switch-Base": 8192,
    "Switch-Large": 8192,
    "Switch-XXL": 2048,
    "Mistral-toy": 256,
    "Mistral-8x7B": 8192,
    "Mistral-8x20B": 8192,
    "Mistral-16x10B": 8192,
}

_BASE_MODEL_HIDDEN_DIM = 768

TP_DEGREE=int(os.getenv("AXLEARN_TP_DEGREE", 4))
SEQ_DEGREE=int(os.getenv("AXLEARN_SEQ_DEGREE", 1))
EP_DEGREE=int(os.getenv("AXLEARN_EP_DEGREE", 1))

MOE_OUTER_BATCH_AXIS_NAMES = ("data", "fsdp")
def get_effective_ep_degree():
    if EP_DEGREE > 1:
        if EP_DEGREE * TP_DEGREE * SEQ_DEGREE == 64:
            return 64
        elif EP_DEGREE * SEQ_DEGREE == 16:
            return 16
    return 1

def get_moe_dim_to_mesh_axis_map(ep_degree_local, tp_degree, cp_degree):
    if ep_degree_local > 1:
        # fsdp = 1
        FSDP_AXIS_NAMES = None
        if ep_degree_local == 64:
            EP_AXIS_NAMES = "expert"
            TP_AXIS_NAMES = None
            FSDP_AXIS_NAMES = "fsdp"
        elif ep_degree_local * tp_degree * cp_degree == 64:
            # doesn't allow tp>1
            EP_AXIS_NAMES = ("expert", "model", "seq")
            TP_AXIS_NAMES = None
        elif ep_degree_local * cp_degree == 16:
            # not used for EP=16 TP=4
            # TODO
            EP_AXIS_NAMES = ("expert", "seq")
            TP_AXIS_NAMES = "model"
        else:
            raise NotImplementedError
    else:
        TP_AXIS_NAMES = "model"
        EP_AXIS_NAMES = "expert"
        FSDP_AXIS_NAMES = "fsdp"
    
    
    # EP v2 within node map
    MOE_DIM_TO_MESH_AXIS_MAP = {
        "me": PartitionSpec(None, None),
        "emnh": PartitionSpec(EP_AXIS_NAMES, FSDP_AXIS_NAMES, None, TP_AXIS_NAMES),
        "eMnh": PartitionSpec(EP_AXIS_NAMES, None, None, TP_AXIS_NAMES),
        "emh": PartitionSpec(EP_AXIS_NAMES, FSDP_AXIS_NAMES, TP_AXIS_NAMES),
        "ehm": PartitionSpec(EP_AXIS_NAMES, TP_AXIS_NAMES, FSDP_AXIS_NAMES),
        "ehM": PartitionSpec(EP_AXIS_NAMES, TP_AXIS_NAMES, None),

        # "onse": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, None, EP_AXIS_NAMES),
        "ogsm": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, None, TP_AXIS_NAMES),
        "ogsM": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, None, None),
        "ogse": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, None, None),
        "ogec": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, EP_AXIS_NAMES, None),
        "oghsM": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, TP_AXIS_NAMES, None, None),
        "oehxx": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, EP_AXIS_NAMES, TP_AXIS_NAMES, None, None),
        # Dispatch and combine tensors.
        "ogsec": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, None, EP_AXIS_NAMES, None),
        "oegcm": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, EP_AXIS_NAMES, None, None, TP_AXIS_NAMES),
        "oegcM": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, EP_AXIS_NAMES, None, None, None),
        "ogecm": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, EP_AXIS_NAMES, None, TP_AXIS_NAMES),
        "ogecM": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, EP_AXIS_NAMES, None, None),
        "oegch": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, EP_AXIS_NAMES, None, None, TP_AXIS_NAMES),
        # "hoesm": PartitionSpec(TP_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, EP_AXIS_NAMES, None, None),
        # "oehx": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, EP_AXIS_NAMES, TP_AXIS_NAMES, None),
        "oexx": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, EP_AXIS_NAMES, None, None),
        "oxxx": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, None, None),
        "oxxe": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, None, EP_AXIS_NAMES),
        "oxe": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, EP_AXIS_NAMES),
        "oex": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, EP_AXIS_NAMES, None),
        "oxx": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, None),
        # "hoex": PartitionSpec(TP_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, EP_AXIS_NAMES, None),
        "hoxx": PartitionSpec(TP_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, None, None),
        "hoxe": PartitionSpec(TP_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, None, EP_AXIS_NAMES),
    }
    return MOE_DIM_TO_MESH_AXIS_MAP


def get_ffn_layer_types():
    ffn_type = os.getenv("AXLEARN_MOE_LAYER_FREQ", "1")
    if ffn_type == "0":
        ffn_layer_types = ["dense"]
    elif ffn_type == "1":
        ffn_layer_types = ["sparse"]
    elif ffn_type == "2":
        ffn_layer_types = ["dense", "sparse"]
    return ffn_layer_types

def get_remat_policy():
    remat_config = os.getenv('AXLEARN_REMAT_LAYER', 'true')
    if remat_config == 'true':
        remat_policy = RematSpecModifier.default_config().set(
            remat_policies={
                "model.decoder.transformer.layer": RematSpec(
                    prevent_cse=True,
                    policy=jax_remat_policies.nothing_saveable,
                ),
            }
        )
    elif remat_config == 'nonmatmul':
        # to be used for non kernel case
        remat_policy = RematSpecModifier.default_config().set(
            remat_policies={
                "model.decoder.transformer.layer": RematSpec(
                    prevent_cse=True,
                    policy=config_for_function(
                        save_and_offload_only_these_names_regex
                    ).set(
                        names_which_can_be_saved="|".join(
                            [
                                RematRegexSavePatterns.QKV_PROJ.value,
                                RematRegexSavePatterns.LINEAR1_X.value,
                            ]
                        ),
                        names_which_can_be_offloaded=None,
                        offload_src=None,
                        offload_dst=None,
                    ),
                ),
            }
        )
    elif remat_config == 'selective':
        remat_policy = RematSpecModifier.default_config().set(
            remat_policies={
                "model.decoder.transformer.layer": RematSpec(
                    prevent_cse=True,
                    policy=config_for_function(
                        save_and_offload_only_these_names_regex
                    ).set(
                        names_which_can_be_saved="|".join(
                            [
                                RematRegexSavePatterns.FLASH_ATTENTION.value,
                                # RematRegexSavePatterns.BLOCKWISE.value,
                                r".*blockwisegating\.token_position_to_id",
                            ]
                        ),
                        names_which_can_be_offloaded=None,
                        offload_src=None,
                        offload_dst=None,
                    ),
                ),
            }
        )
    elif remat_config == 'selkernel':
        remat_policy = RematSpecModifier.default_config().set(
            remat_policies={
                "model.decoder.transformer.layer": RematSpec(
                    prevent_cse=True,
                    policy=config_for_function(
                        save_and_offload_only_these_names_regex
                    ).set(
                        names_which_can_be_saved="|".join(
                            [
                                # RematRegexSavePatterns.QKV_PROJ.value,
                                RematRegexSavePatterns.BLOCKWISE.value,
                                r".*blockwisegating\.token_position_to_id",
                            ]
                        ),
                        names_which_can_be_offloaded=None,
                        offload_src=None,
                        offload_dst=None,
                    ),
                ),
            }
        )
    else:
        remat_policy = {}
    return remat_policy

def callback_modify_trainer_kwargs_env_vars(trainer_kwargs):
    """
    Update trainer_kwargs based on environment variables.

    Args:
        trainer_kwargs (dict): Trainer configuration to modify.

    Returns:
        dict: Updated trainer_kwargs.
    """
    if EVAL_GBS := os.getenv("EVAL_GBS"):
        trainer_kwargs["eval_batch_size"] = int(EVAL_GBS)

    if all(os.getenv(var) for var in ["OPTIMIZER_LR_BASE", "OPTIMIZER_LR_EXP"]):
        lr_exp = int(os.getenv("OPTIMIZER_LR_EXP"))
        opt_lr = float(os.getenv("OPTIMIZER_LR_BASE")) * (10 ** lr_exp)
        trainer_kwargs["learner_kwargs"]["peak_lr"] = opt_lr
    if OPTIMIZER_WD := os.getenv("OPTIMIZER_WD"):
        trainer_kwargs["learner_kwargs"]["weight_decay"] = float(OPTIMIZER_WD)

    if SAVE_EVERY_N_STEPS := os.getenv("SAVE_EVERY_N_STEPS"):
        trainer_kwargs["save_every_n_steps"] = int(SAVE_EVERY_N_STEPS)

    if EVAL_EVERY_N_STEPS := os.getenv("EVAL_EVERY_N_STEPS"):
        trainer_kwargs["eval_every_n_steps"] = int(EVAL_EVERY_N_STEPS)

    return trainer_kwargs
 
def check_env_vars():
    required_vars = ["OPTIMIZER_LR_EXP", "OPTIMIZER_LR_BASE", "OPTIMIZER_WD", "SAVE_EVERY_N_STEPS", "EVAL_EVERY_N_STEPS"]
    missing_vars = [var for var in required_vars if os.getenv(var) is None]
    if missing_vars:
        raise EnvironmentError(f"The following environment variables are expected but not found: {', '.join(missing_vars)}")


def common_trainer_kwargs() -> dict[str, Any]:
    """Returns kwargs that are common to all configs."""
    return {
        "model_kwargs": {
            "z_loss_scale": 1e-6,
        },
        "learner_kwargs": {
            "peak_lr": 1e-2,
            "alpha": 1 / 200.0,
            "weight_decay": 3.16e-4,
        },
        "save_every_n_steps": 250000,
        "keep_every_n_steps": 250000,
        "eval_every_n_steps": 250_000,
        "mesh_shape": mesh_shape_from_axes(data=-1),
    }

class _Trn2CustomConfig(NamedTuple):
    """Config modifications required to run Fuji models on TRN2."""

    # Module config modifications.
    module_modifications: List[ModuleConfigModifier.Config]
    # Partition spec modifications.
    partition_spec_modifications: List[PartitionSpecModifier.Config]


def _generate_trn2_custom_configs(
    model_size: str,
) -> _Trn2CustomConfig:
    """Generate custom module config and PartitionSpec modification for TRN2.

    Args:
        model_size: Size of the Envy model.

    Returns:
        A _Trn2CustomConfig object that contains the generated modifications.
    """
    # TRN2 specific model config modifications.
    if int(os.getenv("AXLEARN_REPEATED", 0)) == 1:
        trn2_module_modifications = [
            ModuleConfigModifier.default_config().set(
                target_config="model.decoder.transformer",
                modification=RepeatedTransformerLayer.default_config(),
            )
        ]
    else:
        trn2_module_modifications = [
            ModuleConfigModifier.default_config().set(
                target_config="model.decoder.transformer",
                modification=StackedTransformerLayer.default_config(),
            )
        ]

    trn2_partition_spec_modifications = [
        PartitionSpecModifier.default_config().set(
            partition_specs={
                # Vocab parallel embeddings sharding from Megatron LM.
                "model.decoder.emb.token_emb": {
                    "param_partition_spec": (
                        "model",
                        ("expert", "fsdp", "seq"),
                    ),
                    "input_partition_spec": (("data", "fsdp"), None),
                    "output_partition_spec": (("data", "fsdp"), None, None),
                    "embedding_partition_spec": ("model", None),
                },
                "model.decoder.output_norm": {
                    "input_partition_spec": (("data", "fsdp"), "model", None),
                    "output_partition_spec": (("data", "fsdp"), None, None),
                },

            },
        ),
    ]

    ffn_layer_types = get_ffn_layer_types()
    if len(ffn_layer_types) == 1:
        target_config="model.decoder.transformer.layer.self_attention.attention.input_linear.input_linear"
        if int(os.getenv("AXLEARN_USE_FUSED_QKV", "0")) == 0:
            mcm = ModuleConfigModifier.default_config().set(
                target_config=target_config,
                modification=GroupedQKVLinear.default_config(),
            )
            trn2_module_modifications.append(mcm)

        # verify after consolidating with tp+fsdp jobs
        if SEQ_DEGREE > 1:
            input_norm_partition = ('expert', 'seq', 'model')
            output_norm_partition = ('expert', 'seq', 'model')
        else:
            input_norm_partition = "model"
            output_norm_partition = None

        trn2_partition_spec_modifications.append(
            PartitionSpecModifier.default_config().set(
                partition_specs={
                    # Sequence parallel shardings for normalization layers
                    "model.decoder.transformer.layer.self_attention.norm": {
                        "input_partition_spec": (("data", "fsdp"), input_norm_partition, None),
                        "output_partition_spec": (("data", "fsdp"), output_norm_partition, None),
                    },
                    "model.decoder.transformer.layer.feed_forward.norm": {
                        "input_partition_spec": (("data", "fsdp"), input_norm_partition, None),
                        "output_partition_spec": (("data", "fsdp"), output_norm_partition, None),
                    },
                    "model.decoder.output_norm": {
                        "input_partition_spec": (("data", "fsdp"), input_norm_partition, None),
                        "output_partition_spec": (("data", "fsdp"), None, None),
                    },
                    "input.input_partitioner": {
                        "path_rank_to_partition": {
                            # Originally be partitioned using sequence degree that would introduce
                            # all-to-alls
                            (None, 1): PartitionSpec(("data", "expert", "fsdp")),
                            (None, 2): PartitionSpec(("data", "expert", "fsdp"), None),
                        }
                    }
                },
            )
        )

        if ffn_layer_types[0] == "dense":
            trn2_partition_spec_modifications[-1].partition_specs[f"model.decoder.transformer.layer.feed_forward.linear2"] = {
                "output_partition_spec": (("data", "fsdp"), None, None),
            }
    elif len(ffn_layer_types) == 2:
        for i in range(2):
            target_config=f"model.decoder.transformer.layer.layer.{i}.self_attention.attention.input_linear.input_linear"
            mcm = ModuleConfigModifier.default_config().set(
                target_config=target_config,
                modification=GroupedQKVLinear.default_config(),
            )
            trn2_module_modifications.append(mcm)
            trn2_partition_spec_modifications.append(
                PartitionSpecModifier.default_config().set( 
                    partition_specs={
                        # Sequence parallel shardings for norms.
                        f"model.decoder.transformer.layer.layer.{i}.self_attention.norm": {
                            "input_partition_spec": (("data", "fsdp"), "model", None),
                            "output_partition_spec": (("data", "fsdp"), None, None),
                        },
                        f"model.decoder.transformer.layer.layer.{i}.feed_forward.norm": {
                            "input_partition_spec": (("data", "fsdp"), "model", None),
                            "output_partition_spec": (("data", "fsdp"), None, None),
                        },
                    },
                )
            )

            if ffn_layer_types[i] == "dense":
                trn2_partition_spec_modifications[-1].partition_specs[f"model.decoder.transformer.layer.layer.{i}.feed_forward.linear2"] = {
                    "output_partition_spec": (("data", "fsdp"), None, None),
                }
    # trn2_lm_head_partition_spec = [
    #     PartitionSpecModifier.default_config().set(
    #         partition_specs={
    #             # Vocab parallel embeddings sharding from Megatron LM.
    #             "model.decoder.lm_head": {
    #                 "param_partition_spec": (
    #                     "model",
    #                     ("expert", "fsdp", "seq"),
    #                 ),
    #             },
    #         },
    #     ),
    # ]
    # trn2_partition_spec_modifications += trn2_lm_head_partition_spec

    return _Trn2CustomConfig(
        module_modifications=trn2_module_modifications,
        partition_spec_modifications=trn2_partition_spec_modifications,
    )



def get_trainer_kwargs(
    model_size: str,
    *,
    vocab_size: int,
    max_sequence_length: int,
    flash_attention: bool,
) -> dict[str, Any]:
    """Construct default trainer kwargs given a model size."""
    tokens_per_batch = 8 * (1024**2)  # 8M tokens.
    trn2_config = _generate_trn2_custom_configs(model_size)
    remat_policy = get_remat_policy()
    ffn_layer_types = get_ffn_layer_types()
    fsdp_degree=int(os.getenv("AXLEARN_FSDP_DEGREE", -1))
    #neuron_mesh = mesh_shape_from_axes(fsdp=fsdp_degree, model=TP_DEGREE, expert=EP_DEGREE, seq=SEQ_DEGREE)
    neuron_mesh = mesh_shape_from_axes(data=-1,fsdp=fsdp_degree, model=TP_DEGREE, expert=EP_DEGREE, seq=SEQ_DEGREE)
    # potentially change for different models
    if EP_DEGREE > 1:
        # to use default of ("expert", "fsdp", "seq")
        attn_dense_fsdp_axis_names = None
        dense_batch_axis_names = ("data", "fsdp")
        attn_dense_seq_axis_names = ("seq", "expert")
    else:
        attn_dense_fsdp_axis_names = attn_dense_seq_axis_names = dense_batch_axis_names = None
    MOE_OUTER_BATCH_AXIS_NAMES = ("data", "fsdp")
    num_groups = 1 #get_effective_ep_degree()
    train_batch_size = int(os.getenv("AXLEARN_TRAIN_BATCH_SIZE", 16))

    # check_env_vars()
    # pylint: disable=use-dict-literal
    if model_size == "test":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=4,
                hidden_dim=8,
                ffn_dim=scaled_hidden_dim(scale=8/3, round_up_to_multiples_of=16),
                num_heads=4,
                num_kv_heads=2,
                vocab_size=32,
                num_experts=8,
                train_capacity_factor=2.0,
                num_groups=2,
                ffn_layer_types=[
                    "dense",
                    "sparse",
                ],
            ),
            learner_kwargs=dict(),
            max_sequence_length=64,
            train_batch_size=16,
            max_step=3000,
            mesh_shape=mesh_shape_from_axes(data=-1),
        )
    elif model_size == "Switch-Base":
        # Num of parameters: 30B.
        # ffn_layer_types = get_ffn_layer_types()
        num_layers=int(os.getenv("AXLEARN_NUM_LAYERS", 12))
        ffn_layer_types = get_ffn_layer_types()
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=num_layers,
                hidden_dim=12 * 128,
                ffn_dim=scaled_hidden_dim(scale=4, round_up_to_multiples_of=128),
                num_heads=12,
                num_kv_heads=12,
                num_experts=NUM_EXPERTS[model_size],
                train_capacity_factor=2.0,
                num_groups=1,
                ffn_structure="hybridnorm",
                # MoE layer every 2 layers.
                ffn_layer_types=ffn_layer_types,
                outer_batch_size=get_outer_batch_from_mesh(MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, neuron_mesh),
            ),
            learner_kwargs=dict(peak_lr=0.01, weight_decay=1e-4, lr_warmup_steps=5_000),
            max_sequence_length=max_sequence_length,
            train_batch_size=train_batch_size,
            max_step=250_000,
            mesh_shape=mesh_shape_from_axes(fsdp=-1, expert=16),
            mesh_rules=(
                (
                    "tpu-v5p-(1024|2048|4096)",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, expert=16, fsdp=16)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=True,
                                        policy=jax_remat_policies.dots_saveable,
                                    ),
                                }
                            ),
                        ],
                    ),
                ),
                (
                    "tpu-v6e-256",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, expert=16, fsdp=16)
                            ),
                        ],
                    ),
                ),
                (
                    "neuron-(trn2|trn2n).48xlarge-64",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                # TP within the chip, FSDP across chips.
                                # Each TRN2 chip has 4 XLA cores.
                                mesh_shape=neuron_mesh
                            ),
                            *trn2_config.module_modifications,
                            *trn2_config.partition_spec_modifications,
                            remat_policy,
                        ],
                    ),
                ),
            ),
        )
    elif model_size == "Switch-Large":
        # Num of parameters: 104B.
        num_layers=int(os.getenv("AXLEARN_NUM_LAYERS", 24))
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=num_layers,
                hidden_dim=16 * 128,
                ffn_dim=scaled_hidden_dim(scale=4, round_up_to_multiples_of=128),
                num_heads=16,
                num_kv_heads=16,
                num_experts=NUM_EXPERTS[model_size],
                train_capacity_factor=2.0,
                num_groups=num_groups,
                ffn_structure="hybridnorm",
                # MoE layer every 2 layers.
                ffn_layer_types=get_ffn_layer_types(),
                outer_batch_size=get_outer_batch_from_mesh(MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, neuron_mesh),
            ),
            learner_kwargs=dict(peak_lr=0.01, weight_decay=1e-4, lr_warmup_steps=5_000),
            max_sequence_length=max_sequence_length,
            train_batch_size=train_batch_size,
            max_step=250_000,  # Most of the evals were done at 100k steps in the paper.
            mesh_shape=mesh_shape_from_axes(fsdp=-1, expert=16),
            mesh_rules=(
                (
                    "tpu-v5p-(2048|4096)",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, expert=16, fsdp=16)
                            ),
                        ],
                    ),
                ),
                (
                    "tpu-v6e-256-4",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, expert=16, fsdp=16)
                            ),
                        ],
                    ),
                ),
                (
                    "tpu-v6e-256",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, expert=16, fsdp=16)
                            ),
                            GradientAccumulationModifier.default_config().set(grad_acc_steps=4),
                        ],
                    ),
                ),
                (
                    "neuron-(trn2|trn2n).48xlarge-64",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                # TP within the chip, FSDP across chips.
                                # Each TRN2 chip has 4 XLA cores.
                                mesh_shape=neuron_mesh
                            ),
                            *trn2_config.module_modifications,
                            *trn2_config.partition_spec_modifications,
                            remat_policy,
                        ],
                    ),
                ),
            ),
        )
    elif model_size == "Switch-XXL":
        # Num of parameters: 520B.
        num_layers=int(os.getenv("AXLEARN_NUM_LAYERS", 24))
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=num_layers,
                hidden_dim=64 * 128,
                ffn_dim=scaled_hidden_dim(scale=2.5, round_up_to_multiples_of=128),
                num_heads=64,
                num_kv_heads=8,
                num_experts=NUM_EXPERTS[model_size],
                train_capacity_factor=2.0,
                num_groups=num_groups,
                ffn_structure="hybridnorm",
                # MoE layer every 2 layers.
                ffn_layer_types=[
                    "dense",
                    "sparse",
                ],
                outer_batch_size=get_outer_batch_from_mesh(MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, neuron_mesh),
            ),
            learner_kwargs=dict(peak_lr=0.01, weight_decay=1e-4, lr_warmup_steps=5_000),
            max_sequence_length=max_sequence_length,
            train_batch_size=train_batch_size,
            max_step=250_000,  # Most of the evals were done at 100k steps in the paper.
            # TODO(kelvin-zou): not verified with real job.
            mesh_shape=mesh_shape_from_axes(fsdp=-1, expert=16, model=8),
            mesh_rules=(
                (
                    "neuron-(trn2|trn2n).48xlarge-64",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                # TP within the chip, FSDP across chips.
                                # Each TRN2 chip has 4 XLA cores.
                                mesh_shape=neuron_mesh
                            ),
                            *trn2_config.module_modifications,
                            *trn2_config.partition_spec_modifications,
                            remat_policy,
                        ],
                    ),
                ),
            ),
        )
    elif "Mistral" in model_size:
        num_layers=int(os.getenv("AXLEARN_NUM_LAYERS", 4))
        num_kv_heads = max(8, TP_DEGREE)
        if int(os.getenv("AXLEARN_NUM_KV_HEADS", -1)) != -1:
            num_kv_heads = int(os.getenv("AXLEARN_NUM_KV_HEADS"))
        if model_size == "Mistral-toy":
            num_heads = 32
            head_size = 32
            ffn_sparse_top_k=2
            ffn_scale_factor=3.5
        elif model_size == "Mistral-8x7B":
            # 32 layers gets to 47B
            num_layers = int(os.getenv("AXLEARN_NUM_LAYERS", 32))
            num_heads = 32
            head_size = 128
            ffn_sparse_top_k=2
            ffn_scale_factor=3.5
        elif model_size == "Mistral-16x10B":
            head_size = 128
            num_heads = 48
            num_layers = int(os.getenv("AXLEARN_NUM_LAYERS", 32))
            # 32 layers gets to 147B
            ffn_scale_factor=2.5
            ffn_sparse_top_k=4
        elif model_size == "Mistral-8x20B":
            head_size = 128
            num_heads = 64
            num_layers = int(os.getenv("AXLEARN_NUM_LAYERS", 44))
            ffn_scale_factor=2
            ffn_sparse_top_k=2
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=num_layers,
                hidden_dim=num_heads*head_size,
                ffn_dim=scaled_hidden_dim(scale=ffn_scale_factor, round_up_to_multiples_of=128),
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                num_experts=NUM_EXPERTS[model_size],
                train_capacity_factor=int(os.getenv("AXLEARN_CAP_FACTOR", ffn_sparse_top_k)),
                num_groups=num_groups,
                ffn_layer_types=ffn_layer_types,
                ffn_sparse_top_k=ffn_sparse_top_k,
                outer_batch_size=get_outer_batch_from_mesh(MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, neuron_mesh),
            ),
            learner_kwargs=dict(peak_lr=0.01, weight_decay=1e-4, lr_warmup_steps=5_000),
            max_sequence_length=int(os.getenv("AXLEARN_MAX_SEQ_LEN", max_sequence_length)),
            train_batch_size=train_batch_size,
            max_step=250_000,
            mesh_shape=mesh_shape_from_axes(fsdp=-1, model=8),
            mesh_rules=(
                (
                    "tpu-v5p-4096",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=16, model=8)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=True,
                                        policy=jax_remat_policies.nothing_saveable,
                                    ),
                                }
                            ),
                        ],
                    ),
                ),
                (
                    "neuron-(trn2|trn2n).48xlarge-64",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                # TP within the chip, FSDP across chips.
                                # Each TRN2 chip has 4 XLA cores.
                                mesh_shape=neuron_mesh
                            ),
                            *trn2_config.module_modifications,
                            *trn2_config.partition_spec_modifications,
                            remat_policy,
                        ],
                    ),
                ),
            ),
        )
    # pylint: enable=use-dict-literal
    else:
        raise NotImplementedError(f"Unknown model size {model_size}.")

    merged_trainer_kwargs = common_trainer_kwargs()
    callback_modify_trainer_kwargs_env_vars(trainer_kwargs)
    merged_trainer_kwargs.update(
        {k: v for k, v in trainer_kwargs.items() if k not in ("model_kwargs", "learner_kwargs")}
    )
    # Update the model_kwargs
    model_kwargs: dict[str, Any] = merged_trainer_kwargs.pop("model_kwargs")
    model_kwargs.update(trainer_kwargs.get("model_kwargs", {}))
    model_kwargs.setdefault("vocab_size", vocab_size)

    model_kwargs["attn_dense_fsdp_axis_names"] = attn_dense_fsdp_axis_names
    model_kwargs["attn_dense_seq_axis_names"] = attn_dense_seq_axis_names
    model_kwargs["dense_batch_axis_names"] = dense_batch_axis_names

    learner_kwargs: dict[str, Any] = merged_trainer_kwargs.pop("learner_kwargs")
    learner_kwargs.update(trainer_kwargs.get("learner_kwargs", {}))

    mesh_shape = merged_trainer_kwargs.get("mesh_shape", mesh_shape_from_axes(data=-1))
    merged_trainer_kwargs["model_cfg"] = model_config(
        flash_attention=flash_attention, mesh_shape=mesh_shape, **model_kwargs
    )
    # If a model is smaller than the base model, do not scale.
    linear_layer_lr_multiplier = min(_BASE_MODEL_HIDDEN_DIM / model_kwargs["hidden_dim"], 1.0)
    merged_trainer_kwargs["learner_cfg"] = adamw_decoupled_learner_config(
        max_step=trainer_kwargs["max_step"],
        # Enable mup-simple.
        adam_update_transformation=mup_simple_adam_update_transformation(
            linear_layer_lr_multiplier,
        ),
        **learner_kwargs,
    )

    return merged_trainer_kwargs


def model_config(
    *,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    num_experts: int,
    vocab_size: int,
    train_capacity_factor: float,
    num_groups: int,
    ffn_layer_types: Sequence[Literal["dense", "sparse"]],
    ffn_sparse_top_k: int = 2,
    ffn_dim: Union[int, config.FunctionConfigBase],
    dropout_rate: float = 0.0,
    flash_attention: bool = False,
    outer_batch_size: int = None,
    mesh_shape: Union[MeshShape, HybridMeshShape],
    attn_dense_fsdp_axis_names=None,
    attn_dense_seq_axis_names=None,
    dense_batch_axis_names=None,
    **kwargs,
) -> causal_lm.Model.Config:
    """Returns an LM model config based on the given hyperparams.

    Args:
        num_layers: The number of Transformer Layers.
        hidden_dim: The Transformer layer input/output dim.
        num_heads: The number of attention heads.
        num_kv_heads: The number of attention KV heads.
        num_experts: The number of experts in the MoE layer.
        vocab_size: The vocabulary size.
        train_capacity_factor: The train capacity factor for the MoE layer.
        ffn_layer_types: The types of layer in the feed-forward network, Options: [dense, sparse].
        dropout_rate: The dropout rate applied throughout the model.
            Defaults to 0.0 (i.e. no dropout).
        ffn_dim: The feed-forward dimension or config function.
            If None, defaults to a setting from https://arxiv.org/abs/2002.05202.
        flash_attention: If True, use flash attention implementation.
        mesh_shape: the mesh shape, used to infer the outer batch size.

    Returns:
        A causal LM config.
    """
    # Use RoPE by default.
    # RoPE <https://arxiv.org/abs/2104.09864> for positional encodings.
    # `CausalAttentionLogitBiasLayer` is already applied in the attention impl.
    attention_mask = None
    # RoPE embeddings: https://arxiv.org/abs/2104.09864.
    attention_qkv_linear = RoFormerQKVLinear.default_config().set(
        input_linear=FusedGroupedQKVLinear.default_config().set(
            num_kv_heads=num_kv_heads,
        ),
        rotary_value=False,
    )
    attention_qkv_linear.rope_pos_emb_layer.theta = 5e5
    norm_cfg = RMSNorm.default_config().set(eps=1e-5, forward_dtype=None)

    transformer_layer_cfg = TransformerLayer.default_config()
    if flash_attention:
        transformer_layer_cfg.self_attention.attention = flash_attention_config()
    else:
        transformer_layer_cfg.self_attention.attention = GroupedQueryAttention.default_config()
    transformer_layer_cfg.self_attention.attention.set(
        # Use q/k-norm in keeping with:
        # <https://arxiv.org/abs/2309.14322>
        query_scale=ScaleQuery.default_config().set(norm=norm_cfg.clone()),
        key_scale=ScaleKey.default_config().set(norm=norm_cfg.clone()),
    )
    if outer_batch_size is None:
        outer_batch_size = get_outer_batch_from_mesh(
            MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, mesh_shape
        )
    MOE_DIM_TO_MESH_AXIS_MAP = get_moe_dim_to_mesh_axis_map(EP_DEGREE, TP_DEGREE, SEQ_DEGREE)
    use_blockwise = int(os.getenv('AXLEARN_USE_BLOCKWISE', 1))
    if use_blockwise == 1:
        gating_type = TopKGatingGatherBlockwise
        gating_cfg = gating_type.default_config()
    elif use_blockwise == 2:
        gating_type = TopKGatingGatherBlockwiseV2
        gating_cfg = gating_type.default_config()
        gating_cfg.dim_to_mesh_axis_map=MOE_DIM_TO_MESH_AXIS_MAP
    else:
        gating_type = TopKGatingGather
        gating_cfg = gating_type.default_config()
    expert_config = TransformerFeedForwardMoE.default_config().set(
        outer_batch=outer_batch_size,
        num_experts=num_experts,
        input_dim=hidden_dim,
        num_groups=num_groups,
        dim_to_mesh_axis_map=MOE_DIM_TO_MESH_AXIS_MAP,
        gating=gating_cfg,
    )
    expert_config.gating.top_k = ffn_sparse_top_k
    expert_config.gating.train_capacity_factor = train_capacity_factor

    emb_cfg: TransformerTextEmbeddings.Config = TransformerTextEmbeddings.default_config().set(
        pos_emb=None
    )
    # emb_cfg.token_emb.param_partition_spec = (("expert", "fsdp", "seq"), "model")

    cfg = common_model_config(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
        # SwiGLU from https://arxiv.org/abs/2002.05202.
        activation_fn=("nn.silu", "linear"),
        ffn_dim=ffn_dim,
        normalization=norm_cfg,
        dropout_rate=dropout_rate,
        emb_cfg=emb_cfg,
        # Since we pass `layer_cfg`, this is already set.
        attention_cfg=None,
        attention_mask=attention_mask,
        attention_qkv_linear=attention_qkv_linear,
        layer_cfg=transformer_layer_cfg,
        ffn_layer_types=ffn_layer_types,
        expert_cfg=expert_config,
        fsdp_axis_names=attn_dense_fsdp_axis_names, 
        seq_axis_names=attn_dense_seq_axis_names,
        batch_axis_names=dense_batch_axis_names,
        **kwargs,
    )
    # if flash_attention:
    #     cfg.decoder.transformer.layer.remat_spec = RematSpec(
    #         prevent_cse=False, policy=jax_remat_policies.dots_saveable
    #     )
    return cfg


def trainer_configs(
    train_input_source: SourceBuilder,
    eval_input_sources: SourceBuilder,
) -> dict[str, TrainerConfigFn]:
    """Returns a mapping from config_name to TrainerConfigFn's.

    Args:
        train_input_source: A callable (vocab_size, max_sequence_length) -> input source config.
        eval_input_soruces: A callable (vocab_size, max_sequence_length) -> eval input sources.
    """
    arch = "envy"
    config_map = {}
    vocab_size = VOCAB_SIZE
    flash_attention = bool(int(os.getenv("AXLEARN_FLASH_ATTENTION", 1)))

    for model_size in MODEL_SIZES:
        seq_len = MAX_SEQUENCE_LENGTH[model_size]
        config_name = make_config_name(arch=arch, model_size=model_size)
        kwargs = get_trainer_kwargs(
            model_size,
            vocab_size=vocab_size,
            # Use default flash attention.
            flash_attention=flash_attention,
            max_sequence_length=seq_len,
        )

        # Test models sometimes override it to a very small length.
        seq_len = kwargs.pop("max_sequence_length", seq_len)

        # pylint: disable-next=unexpected-keyword-arg,missing-kwoa
        config_map[config_name] = get_trainer_config_fn(
            train_input_source=train_input_source(
                vocab_size=vocab_size, max_sequence_length=seq_len
            ),
            evalers=evaler_config_dict(
                eval_input_sources(vocab_size=vocab_size, max_sequence_length=seq_len),
            ),
            **kwargs,
        )

        # Only Switch-Base model size is runnable on a single node mode.
        if model_size == "Switch-Base":

            def make_single_host_config(base_config_name: str) -> SpmdTrainer.Config:
                """Make a single-host variant of the base config."""

                # pytype: disable=annotation-type-mismatch
                cfg: SpmdTrainer.Config = config_map[base_config_name]().clone()
                # pytype: enable=annotation-type-mismatch
                cfg.input.batcher.feed_batch_size = 8
                for evaler in cfg.evalers.values():
                    evaler.input.batcher.feed_batch_size = 8
                remat_modifier = (
                    RematSpecModifier.default_config()
                    .set(
                        remat_policies={
                            "model.decoder.transformer.layer": RematSpec(
                                prevent_cse=True,
                                policy=jax_remat_policies.nothing_saveable,
                            ),
                        }
                    )
                    .instantiate()
                )
                cfg = remat_modifier(cfg)
                return cfg

            # Make single-host config
            make_single_host_config_func = functools.partial(make_single_host_config, config_name)
            config_map[f"{config_name}-single-host"] = make_single_host_config_func

    return config_map
