# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/praxis:
# Copyright 2022 The Pax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Utils for tests for mixture_of_experts.py"""
import os
import glob
import shutil
import numpy as np
import json
from functools import partial, cache
from itertools import product
from contextlib import contextmanager
import math
from datetime import datetime
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, Mesh
from parse_pytest_results import parse_pytest_xml
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.decoder import Decoder
from axlearn.common.attention import TransformerLayer, GroupedQueryAttention, RoFormerQKVLinear, GroupedQKVLinear, ScaleKey, ScaleQuery, set_double_shard_weights_config
from axlearn.common.layers import RMSNorm
from axlearn.common.mixture_of_experts import (
    TopKGating,
    TransformerFeedForwardMoE,
    TopKGatingGather,
    TopKGatingGatherBlockwise,
    get_outer_batch_from_mesh
)

from axlearn.common.utils import PartitionSpec, infer_mesh_shape, cast_floats
from axlearn.experiments.text.gpt.common import MESH_AXIS_NAMES, mesh_shape_from_axes
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, DefaultInitializer, WeightInitializer
from axlearn.experiments.text.gpt.envy import MOE_OUTER_BATCH_AXIS_NAMES, get_moe_dim_to_mesh_axis_map

TEST_SUITE = os.environ.get("TEST_SUITE", 'presubmit').lower()

# FP32 test tolerances
TEST_TOLS_FP32 = {
    "atol": 5e-4,
    "rtol": 1e-2,
}
# BF16 test tolerances
TEST_TOLS_BF16 = {
    "atol": 8e-3,
    "rtol": 1e-3,
}

def get_mesh_dims_from_spec(mesh_spec):
    mesh = mesh_shape_from_axes(**mesh_spec)
    mesh = infer_mesh_shape(mesh)
    return mesh

def build_name(cfg, invoker_cfg):
    if invoker_cfg['mesh_spec']:
        #generating the testcase-name with required tp, ep, cp config 
        fsdp = invoker_cfg['mesh_spec']['fsdp']
        parts = [f"fsdp{fsdp}"]
        
        if 'model' in invoker_cfg['mesh_spec']:
            parts.append(f"tp{invoker_cfg['mesh_spec']['model']}")
        if 'expert' in invoker_cfg['mesh_spec']:
            parts.append(f"ep{invoker_cfg['mesh_spec']['expert']}")
        if 'seq' in invoker_cfg['mesh_spec']:
            parts.append(f"seq{invoker_cfg['mesh_spec']['seq']}")
            
        mesh_str = "".join(parts)
    else:
        mesh_str = ''

    if invoker_cfg['dtype'] == jnp.bfloat16:
        dtype_str = "bf16"
    elif invoker_cfg['dtype'] == jnp.float32:
        dtype_str = "fp32"
    else:
        dtype_str = f"{invoker_cfg['dtype']}"
    if hasattr(cfg, 'gating'):
        # MoE layer
        if hasattr(cfg.gating, 'block_size'):
            block_size_str = f'_blocksize{cfg.gating.block_size}'
        else:
            block_size_str = ''
        return f"MoE_i{cfg.input_dim}_h{cfg.hidden_dim}_e{cfg.num_experts}_topk{cfg.gating.top_k}_g{cfg.num_groups}_ec{cfg.gating.train_capacity_factor}{block_size_str}_b{invoker_cfg['batch_size']}_s{invoker_cfg['seq_len']}_mesh{mesh_str}_{dtype_str}"
    elif hasattr(cfg, "feed_forward"):
        return f"transformer_i{cfg.input_dim}_h{cfg.feed_forward.hidden_dim}_e{cfg.feed_forward.num_experts}_topk{cfg.feed_forward.gating.top_k}_g{cfg.feed_forward.num_groups}_ec{cfg.feed_forward.gating.train_capacity_factor}_b{invoker_cfg['batch_size']}_s{invoker_cfg['seq_len']}_mesh{mesh_str}_{dtype_str}"
    else:
        # Gating layer
        E = invoker_cfg['input_shape'][-1]
        G = invoker_cfg['input_shape'][1]
        if hasattr(cfg, 'block_size'):
            block_size_str = f'_blocksize{cfg.block_size}'
        else:
            block_size_str = ''
        return f"Gating_b{invoker_cfg['batch_size']}_s{invoker_cfg['seq_len']}_e{E}_topk{cfg.top_k}_g{G}_ec{cfg.train_capacity_factor}{block_size_str}_mesh{mesh_str}_{dtype_str}"

def _topkgather_to_topk(output, top_k, cf):
    tok_perm_idx, expert_index, exp_aff_mask = output.combine_tensor

    O, G, S, _ = tok_perm_idx.shape
    E = exp_aff_mask.shape[-1]

    expert_cap = jnp.int32(S*cf/E)

    exp_aff = jnp.take_along_axis(exp_aff_mask, expert_index, axis=-1)

    base = jnp.zeros((O, G, S, E * expert_cap), dtype=exp_aff_mask.dtype)

    idx_O, idx_G, idx_S = jnp.meshgrid(
        jnp.arange(O), 
        jnp.arange(G), 
        jnp.arange(S), 
        indexing='ij'
    )

    output_tensor = base.at[idx_O[..., None], idx_G[..., None], idx_S[..., None], tok_perm_idx].add(exp_aff)
    output_tensor = output_tensor.reshape(O, G, S, E, expert_cap)

    dispatch_tensor = output_tensor.astype(bool)

    return TopKGating.Output(
        combine_tensor=output_tensor,
        dispatch_tensor=dispatch_tensor,
        load_balance_loss=output.load_balance_loss,
        router_z_loss=output.router_z_loss
    )

class ModuleConfig():
    def __init__(self, cfg, invoker_cfg):
        self.cfg = cfg
        self.invoker_cfg = invoker_cfg
        self.dtype = invoker_cfg['dtype']
        self.input_shape = invoker_cfg['input_shape']
        self.layer = None
        self.mesh = None
        self.mesh_spec = invoker_cfg['mesh_spec']
        self.mesh_dims = None
        self.num_devices = None
        self.device = invoker_cfg['device']
        self.inputs = {}
        self.state = None
        self.testid = None
        self.atol = TEST_TOLS_BF16['atol'] if self.dtype in ["bfloat16", jnp.bfloat16] else TEST_TOLS_FP32['atol']
        self.rtol = TEST_TOLS_BF16['rtol'] if self.dtype in ["bfloat16", jnp.bfloat16] else TEST_TOLS_FP32['rtol']
        self.name = build_name(cfg, invoker_cfg)

    def to_dict(self):
        return {
            'cfg': self.cfg.to_dict(),
            'invoker_cfg': self.invoker_cfg,
            'jax': jax.__version__,
        }
    
    def matches_cached_config(self):
        # with open(os.path.join(self.golden_dump_path, 'golden_config_new.txt'), 'w') as f:
        #     f.write(f"{self.to_dict()}")
        try:
            with open(os.path.join(self.golden_dump_path, 'golden_config.txt'), 'r') as f:
                loaded_cfg = f.read()
            return f"{self.to_dict()}" == loaded_cfg
        except FileNotFoundError:
            return False

    @property
    def golden_dump_path(self):
        if hasattr(self, '_golden_dump_path'):
            return self._golden_dump_path
        else:
            GOLDENS_DIR = os.getenv('GOLDENS_DIR')
            if GOLDENS_DIR:
                testname = self.testid.split('.', 1)[-1]
                self.testname = testname
                self._golden_dump_path = os.path.join(GOLDENS_DIR, testname)
            else:
                self._golden_dump_path = None
            return self._golden_dump_path

    def dump_goldens(self, tensors):
        os.makedirs(self.golden_dump_path, exist_ok=True)
        with open(os.path.join(self.golden_dump_path, 'golden_config.txt'), 'w') as f:
            f.write(f"{self.to_dict()}")
        try:
            for k, v in tensors.items():
                jnp.save(os.path.join(self.golden_dump_path, f'{k}.npy'), v, allow_pickle=True)
            return True
        except OverflowError:
            # TODO: shard and write to disk
            print(self.testid, self.testname, 'OverflowError while saving tensors, skipping golden dump')
            return False

    def load_goldens(self, tensors):
        if not self.golden_dump_path or not os.path.exists(self.golden_dump_path):
            print(self.testid, self.testname, 'Could not find cache for test')
            return False
        if not self.matches_cached_config():
            print(self.testid, self.testname, 'Cached config does not match current config')
            return False
        try:
            for k in tensors.keys():
                tensor_path = os.path.join(self.golden_dump_path, f'{k}.npy')
                if not os.path.exists(tensor_path):
                    print('Incomplete cache, could not find', tensor_path)
                    return False
                tensors[k] = jnp.load(tensor_path, allow_pickle=True)
                if isinstance(tensors[k], np.ndarray):
                    val = tensors[k]
                    i = 0
                    for ck, v in np.ndenumerate(val):
                        if i == 0:
                            tensors[k] = v
                        i+=1
                    assert i == 1, f"Expected single value, got {i} values"
            return True
        except Exception as e:
            print(self.testid, self.testname, 'Error loading cached tensors:', e)
            return False

    @property
    def layer_type(self):
        if isinstance(self.cfg, TransformerFeedForwardMoE.Config):
            return "MoE"
        elif isinstance(self.cfg, TransformerLayer.Config):
            return "Transformer"
        else:
            return "Gating"
    
    @property
    def gating_type(self):
        if self.layer_type == "Transformer":
            return self.cfg.feed_forward.gating.__class__.__name__
        elif self.layer_type == "MoE":
            return self.cfg.gating.__class__.__name__
        else:
            return self.cfg.__class__.__name__

    @contextmanager
    def dump_for_spectometer(self):
        if self.device != "neuron":
            yield
            return

        testname = self.testid.split('.', 1)[-1]
        neuron_dump_path = os.path.join(os.environ.get('NEURON_DUMP_PATH'), testname)
        prev_flags = os.environ["NEURON_CC_FLAGS"]
        os.environ["NEURON_CC_FLAGS"] = os.environ["NEURON_CC_FLAGS"] + f" --dump={neuron_dump_path}"
        # Create dump folder if it doesn't exist
        os.makedirs(neuron_dump_path, exist_ok=True)
        # Create metadata JSON file for spectometer
        metadata = {
            "name": testname,
            "hlo_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "submitter_alias": "huilgolr",
            "compiler_flags": prev_flags,
            "target_instance_type": "trn2.48xl",
            "model_info": {
                "name": testname,
                "batch_size": self.invoker_cfg["batch_size"],
                "number_of_layers": 1,  # Assuming single layer for now
                "sequence_length": self.invoker_cfg["seq_len"],
                "hlo_url": f"s3://kaena-nn-models/spectometer-staging/training-moe-jax-integration-tests/{testname}/model.hlo_module.pb",
            },
            "software": {
                "jax": jax.__version__,
                "axlearn": os.getenv("GIT_COMMIT")
            },
            "source_code_url": "https://github.com/rahul003/axlearn/",
        }
        metadata_path = os.path.join(neuron_dump_path, "hlo_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        yield
        os.environ["NEURON_CC_FLAGS"] = prev_flags
        matches = glob.glob(os.path.join(neuron_dump_path, "**/*.code"))
        if matches:
            for match in matches:
                shutil.copyfile(
                    match,
                    os.path.join(neuron_dump_path, "model.hlo_module.pb"),
                )

    def reset(self):
        self.layer = None
        self.state = None
        self.inputs = None

class ExperimentConfig():
    def __init__(
            self, 
            test_cfg, 
            golden_cfg, 
            test_invoker_cfg, 
            golden_invoker_cfg,
            loss_fn = None, 
            conv_output = None,
            prefix = None
        ):
        self.test = ModuleConfig(test_cfg, test_invoker_cfg)
        self.golden = ModuleConfig(golden_cfg, golden_invoker_cfg) if golden_cfg else None
        self.loss_fn = loss_fn
        self.conv_output = conv_output
        self.prefix = prefix

    def print_summary(self):
        print('\n-------------------\nTest:', build_name(self.test.cfg, self.test.invoker_cfg))
        print('> Test device:', self.test.device, 'Layer:', self.test.layer_type, 'Gating:', self.test.gating_type)
        if self.golden:
            print('> Golden device:', self.golden.device, 'Layer:', self.golden.layer_type, 'Gating:', self.golden.gating_type)
    
    def instantiate(self, testid):
        self.test.testid = testid
        self.test.mesh_dims = get_mesh_dims_from_spec(self.test.invoker_cfg["mesh_spec"])
        self.test.num_devices = math.prod(self.test.mesh_dims)
        if self.golden:
            self.golden.testid = testid
            self.golden.mesh_dims = get_mesh_dims_from_spec(self.golden.invoker_cfg["mesh_spec"])
            self.golden.num_devices = math.prod(self.golden.mesh_dims)
        self.maybe_set_outer_batch()
        if self.golden:
            self.init_layer(self.golden)
        self.init_layer(self.test, state_to_copy=self.golden.state if self.golden else None)
        self.random_inputs_with_mesh(self.test.cfg.dim_to_mesh_axis_map if hasattr(self.test.cfg, 'dim_to_mesh_axis_map') else None)

    def maybe_set_outer_batch(self):
        test_outer_batch = get_outer_batch_from_mesh(MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, self.test.mesh_dims)
        if isinstance(self.test.cfg, TransformerFeedForwardMoE.Config):
            self.test.cfg.outer_batch = test_outer_batch
        self.test.outer_batch = test_outer_batch
        
        if self.golden:
            golden_outer_batch = get_outer_batch_from_mesh(MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, self.golden.mesh_dims)
            if isinstance(self.golden.cfg, TransformerFeedForwardMoE.Config):
                self.golden.cfg.outer_batch = golden_outer_batch
            self.golden.outer_batch = golden_outer_batch

    def init_layer(self, module_config, state_to_copy=None):
        devices = jax.devices(module_config.device)[:module_config.num_devices]
        module_config.mesh = Mesh(mesh_utils.create_device_mesh(module_config.mesh_dims, devices=devices), MESH_AXIS_NAMES) 
        with module_config.mesh:
            with jax.default_device(devices[0]):
                module_config.layer = module_config.cfg.instantiate(parent=None) 
                module_config.param_specs = module_config.layer.create_parameter_specs_recursively()
                for p, v in module_config.param_specs.items():
                    module_config.param_specs[p] = v if v else None
                param_partition_specs = jax.tree.map(lambda spec: spec.sharding, module_config.param_specs)
                module_config.param_partition_specs = param_partition_specs
                if state_to_copy:
                    module_config.state = {}
                    for key, value in state_to_copy.items():
                        # First put on a single device
                        module_config.state[key] = jax.device_put(value, param_partition_specs[key])
                else:
                    def _init_state(prng_key):
                        params = module_config.layer.initialize_parameters_recursively(prng_key)
                        return params
                    init_fn = jax.jit(_init_state, in_shardings=(None,), out_shardings=param_partition_specs)
                    module_config.state = init_fn(jax.random.PRNGKey(123))
                module_config.state = cast_floats(module_config.state, to_dtype=module_config.dtype)
                # TODO: Currently bf16 seeing expert index mismatch with f32. Setting routing to f32.
                if 'gate_weight' in module_config.state:
                    module_config.state['gate_weight'] = module_config.state['gate_weight'].astype(jnp.float32)
    
    def random_inputs_with_mesh(self, dim_to_mesh_axis_map): 
        
        # replace O and S from input shape with outer batch and seq
        if self.test.layer_type == "MoE":
            input_key = 'inputs'
            pspec = PartitionSpec(('data','fsdp'), 'model', None)
        elif self.test.layer_type == "Transformer":
            input_key = 'data'
            pspec = PartitionSpec(('data','fsdp'), ('expert', 'seq', 'model'), None)
        else:
            input_key = 'logits'
            pspec = dim_to_mesh_axis_map["ogse"]
            _, G, _, E = self.test.input_shape
            O = self.test.outer_batch
            S = (self.test.invoker_cfg["batch_size"] * self.test.invoker_cfg["seq_len"])//(O * G)
            self.test.input_shape = (O, G, S, E)
            if self.golden:
                _, G, _, E = self.golden.input_shape
                O = self.golden.outer_batch
                S = (self.golden.invoker_cfg["batch_size"] * self.golden.invoker_cfg["seq_len"])//(O * G)
                self.golden.input_shape = (O, G, S, E)
        
        in_shard_test = NamedSharding(mesh=self.test.mesh, spec=pspec)
        # create tensors on host to avoid OOM
        with jax.default_device(jax.devices("cpu")[0]):
            inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=self.test.input_shape, dtype=self.test.dtype)
        inputs = jax.device_get(inputs)   # device_put seg-faults without this
        self.test.inputs[input_key] = jax.device_put(inputs, in_shard_test)

        if self.golden:
            assert self.test.input_shape == self.golden.input_shape
            in_shard_golden = NamedSharding(mesh=self.golden.mesh, spec=pspec)
            self.golden.inputs[input_key] = jax.device_put(inputs, in_shard_golden)

class GridSpaceBuilder:
    def __init__(self, layer='moe', test=TopKGatingGather, golden=TopKGating, test_device="neuron", golden_device="cpu"):
        self.layer = layer
        self.test = test
        self.golden = golden
        self.test_device = test_device
        self.golden_device = golden_device

    def create_test_config(self, **kwargs):
        return create_test_config(
            test=self.test, golden=self.golden, test_device=self.test_device, golden_device=self.golden_device,
            layer=self.layer,
            **kwargs
        )
    
    def build_toy_grid_space(self):
        return self.create_test_config(
                input_dim=2048, hidden_dim=2048,
                n_experts=2, top_k=1, n_groups=1, capacity_factor=1,
                mesh_spec={},
                batch=1, seq=256, dtype=jnp.float32
            )
    
    def build_presubmit_grid_space(self):
        grid_space = []
        tp_4_mesh_spec = {"fsdp":-1, "model":4}
        tp_16_mesh_spec = {"fsdp":-1, "model":16}
        tp_64_mesh_spec = {"fsdp":-1, "model":64}
        kwargs={
            'dtype': jnp.bfloat16,
            'batch': 16,
        }
        # all tp4
        grid_space.extend([
            # 12b
            self.create_test_config(
                **kwargs, input_dim=2048, hidden_dim=7168, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, seq=4096, mesh_spec=tp_4_mesh_spec,
            ),
            # switch base
            self.create_test_config(
                **kwargs, input_dim=1536, hidden_dim=6144, n_experts=128, top_k=2, n_groups=1, capacity_factor=2, seq=8192, mesh_spec=tp_4_mesh_spec,
            ),
            # 50b
            self.create_test_config(
                **kwargs, input_dim=4096, hidden_dim=14336, n_experts=8, top_k=2, n_groups=4, capacity_factor=2, seq=8192, mesh_spec=tp_4_mesh_spec,
            ),
            # llama4 scout
            self.create_test_config(
                **kwargs, input_dim=5120, hidden_dim=8192, n_experts=16, top_k=1, n_groups=1, capacity_factor=4, seq=4096, mesh_spec=tp_4_mesh_spec,
            ),
            # gpt-oss main config
            self.create_test_config(
                **kwargs, input_dim=2880, hidden_dim=2880, n_experts=128, top_k=4, n_groups=64, capacity_factor=2, seq=4096, mesh_spec={"fsdp": -1, "expert": 64},
            ),
        ])
        
        if self.layer == "moe":
            # all tp16
            kwargs['batch'] = 4
            # switch large
            grid_space.append(
                self.create_test_config(
                **kwargs, input_dim=2048, hidden_dim=8192, n_experts=128, top_k=4, n_groups=1, capacity_factor=2, seq=2048, mesh_spec=tp_16_mesh_spec,
                )
            )
            # deepseek
            self.create_test_config(
                **kwargs, input_dim=7168, hidden_dim=2048, n_experts=256, top_k=8, n_groups=4, capacity_factor=4, seq=4096, mesh_spec=tp_4_mesh_spec,
            ),
            # dbrx
            grid_space.append(
                self.create_test_config(
                    **kwargs, input_dim=6144, hidden_dim=10752, mesh_spec=tp_16_mesh_spec,
                    n_experts=16, top_k=4, n_groups=1, capacity_factor=4, seq=4096
                )
            )
            # 16x10b
            grid_space.append(
                self.create_test_config(
                    **kwargs, input_dim=6144, hidden_dim=15360, mesh_spec=tp_16_mesh_spec,
                    n_experts=16, top_k=4, n_groups=1, capacity_factor=4, seq=4096
                )
            )
            # 8x20b
            grid_space.append(
                self.create_test_config(
                    **kwargs, input_dim=8192, hidden_dim=16384, mesh_spec=tp_16_mesh_spec,
                    n_experts=8, top_k=2, n_groups=1, capacity_factor=2, seq=2048
                )
            )
            # llama 4 maverick
            grid_space.append(
                self.create_test_config(
                    **kwargs, input_dim=5120, hidden_dim=6144, mesh_spec=tp_16_mesh_spec,
                    n_experts=128, top_k=1, n_groups=1, capacity_factor=2, seq=8192
                )
            )

            # tp64
            # 8x20b
            kwargs['batch'] = 1
            grid_space.append(
                self.create_test_config(
                    **kwargs, input_dim=8192, hidden_dim=16384, mesh_spec=tp_64_mesh_spec,
                    n_experts=8, top_k=2, n_groups=1, capacity_factor=2, seq=4096
                )
            )
            # switch xxl
            grid_space.append(
                self.create_test_config(
                    **kwargs, input_dim=8192, hidden_dim=20480, mesh_spec=tp_64_mesh_spec,
                    n_experts=64, top_k=2, n_groups=1, capacity_factor=2, seq=8192
                )
            )
            
        return grid_space

    def build_grid_space_input_hidden(self, input_dim=2048, hidden_dim=7168, min_seq=8*1024, max_seq=None, min_tp=None, max_tp=None, max_E=None, dtype=jnp.bfloat16, exclude_combinations=None, exclude_seq_lengths=None):
        # TODO: consider removing DP replicas of groups and parallelize different tests on different cores if possible
        # Grid space for testing
        grid_space = []
        batch_sizes = {
            1: 64,
            4: 16,
            8: 8,
            16: 4,
            32: 2,
            64: 1,
        }
        # All tp,cp,ep tuples
        tp_cp_ep_combinations = [
            #tp, cp, ep
            (4,4,4), (4,1,1),
            (1,16,4), 
            (16,1,1), (16,1,4)
        ]
        
        # Filter out excluded combinations
        if exclude_combinations:
            tp_cp_ep_combinations = [combo for combo in tp_cp_ep_combinations if combo not in exclude_combinations]
        
        kwargs={
            'dtype': dtype,
            'input_dim': int(input_dim),
            'hidden_dim': int(hidden_dim),
        }
        
        for tp_degree, cp_degree, ep_degree in tp_cp_ep_combinations:

            if min_tp is not None and ep_degree == 1 and tp_degree < min_tp:
                continue
            if max_tp is not None and ep_degree == 1 and tp_degree > max_tp:
                continue
                
            #mesh_spec
            mesh_spec = {"fsdp": -1}
            if tp_degree > 1:
                mesh_spec["model"] = tp_degree
            if ep_degree > 1:
                mesh_spec["expert"] = ep_degree
            if cp_degree > 1:
                mesh_spec["seq"] = cp_degree
            
            batch = batch_sizes[tp_degree]
            
            cf = 2
            
            for E in [1, 8, 16, 128]:
                if max_E and E > max_E:
                    break
                if E >= 64 and tp_degree < 16:
                    continue
                if E < ep_degree:  #making sure that E >= ep
                    continue
                    
                for K in [1, 2, 16]:
                    if K >= E//4:
                        break
                    
                    # n_groups based on parallelism type
                    if ep_degree > 1:
                        G_values = [ep_degree]  # n_groups = ep_degree for EP
                    else:
                        G_values = [1, 4]
                    
                    for G in G_values:
                        if G > E:
                            break
                        S = min_seq
                        while (max_seq and S <= max_seq) or (S <= 16*1024):
                            # Skip excluded sequence lengths
                            if exclude_seq_lengths and S in exclude_seq_lengths:
                                S = S * 2
                                continue
                            grid_space.append(self.create_test_config(**kwargs, n_experts=E, top_k=K, n_groups=G, capacity_factor=cf, seq=S, batch=batch, mesh_spec=mesh_spec))
                            S = S * 2
        return grid_space

    
    def build_grid_space_gpt_oss(self):
        kwargs={
            'dtype': jnp.bfloat16,
            'input_dim': 2880,
            'hidden_dim': 2880,
            'n_experts': 128,
            'capacity_factor': 2,
            'seq': 4096,
        }
        grid_space = []
        
        mesh_configs = [
            # TP=4 configurations
            ({"fsdp": -1, "model": 4}, 16, 1),
            
            # EP=64 configurations  
            ({"fsdp": -1, "expert": 64}, 16, 64),
            
            # Mixed TP+EP configurations
            ({"fsdp": -1, "model": 4, "expert": 16}, 16, 16),
        ]
        
        for mesh_spec, batch, n_groups in mesh_configs:
            for top_k in [4, 8]:
                test_kwargs = kwargs.copy()
                test_kwargs['n_groups'] = n_groups
                grid_space.append(self.create_test_config(**test_kwargs, top_k=top_k, batch=batch, mesh_spec=mesh_spec))
        
        return grid_space

    def build_grid_space_llama4_maverick(self):
        kwargs={
            'dtype': jnp.bfloat16,
            'input_dim': 5120,
            'hidden_dim': 6144,
            'n_experts': 128,
            'dtype': jnp.bfloat16,
            'seq': 8192,
            'capacity_factor': 2,
            'n_groups': 1,
        }

        # TODO: consider removing DP replicas of groups and parallelize different tests on different cores if possible
        # Grid space for testing
        grid_space = []
        
        mesh_configs = [
            # TP-only configurations
            ({"fsdp": -1, "model": 4}, 16, 1),      # TP=4, batch=16, n_groups=1
            ({"fsdp": -1, "model": 16}, 4, 1),      # TP=16, batch=4, n_groups=1
            ({"fsdp": -1, "model": 64}, 1, 1),      # TP=64, batch=1, n_groups=1
            
            # EP-only configurations
            ({"fsdp": -1, "expert": 16}, 16, 16),   # EP=16, batch=16, n_groups=16
            ({"fsdp": -1, "expert": 64}, 16, 64),   # EP=64, batch=16, n_groups=64
            
            # Mixed TP+EP configurations
            ({"fsdp": -1, "model": 4, "expert": 16}, 16, 16),  # TP=4×EP=16, batch=16, n_groups=16
        ]
        
        for mesh_spec, batch, n_groups in mesh_configs:
            for top_k in [1, 8]:
                test_kwargs = kwargs.copy()
                test_kwargs['n_groups'] = n_groups
                grid_space.append(self.create_test_config(**test_kwargs, top_k=top_k, batch=batch, mesh_spec=mesh_spec))
        return grid_space
    
    def build_grid_space_switch_xxl(self):
        kwargs={
            'dtype': jnp.bfloat16,
            'input_dim': 8192,
            'hidden_dim': 20480,
            'n_experts': 32,
            'dtype': jnp.bfloat16,
            'seq': 4096,
            'capacity_factor': 2,
            'n_groups': 1,
        }
        # TODO: consider removing DP replicas of groups and parallelize different tests on different cores if possible
        # Grid space for testing
        grid_space = []
        
        mesh_configs = [
            # TP-only configurations
            ({"fsdp": -1, "model": 4}, 16, 1),      # TP=4, batch=16, n_groups=1
            ({"fsdp": -1, "model": 16}, 4, 1),      # TP=16, batch=4, n_groups=1
            ({"fsdp": -1, "model": 64}, 1, 1),      # TP=64, batch=1, n_groups=1
            
            # EP-only configurations (EP <= n_experts)
            ({"fsdp": -1, "expert": 16}, 16, 16),   # EP=16, batch=16, n_groups=16
            # ({"fsdp": -1, "expert": 32}, 16, 32),   # EP=32 - NotImplementedError in mesh axis mapping
            
            # Mixed TP+EP configurations
            ({"fsdp": -1, "model": 4, "expert": 16}, 16, 16),  # TP=4×EP=16, batch=16, n_groups=16
        ]
        
        for mesh_spec, batch, n_groups in mesh_configs:
            for top_k in [1, 2]:
                test_kwargs = kwargs.copy()
                test_kwargs['n_groups'] = n_groups
                grid_space.append(self.create_test_config(**test_kwargs, top_k=top_k, batch=batch, mesh_spec=mesh_spec))
        return grid_space

    def build_grid_space_qwen3_235b(self):
        kwargs={
            'dtype': jnp.bfloat16,
            'input_dim': 4096,
            'hidden_dim': 12288,
            'n_experts': 128,
            'dtype': jnp.bfloat16,
            'seq': 16384,
            'capacity_factor': 2,
            'n_groups': 1,
        }
        # TODO: consider removing DP replicas of groups and parallelize different tests on different cores if possible
        # Grid space for testing
        grid_space = []
        
        mesh_configs = [
            # TP-only configurations
            ({"fsdp": -1, "model": 4}, 16, 1),      # TP=4, batch=16, n_groups=1
            ({"fsdp": -1, "model": 16}, 4, 1),      # TP=16, batch=4, n_groups=1
            ({"fsdp": -1, "model": 64}, 1, 1),      # TP=64, batch=1, n_groups=1
            
            # EP-only configurations
            ({"fsdp": -1, "expert": 16}, 16, 16),   # EP=16, batch=16, n_groups=16
            ({"fsdp": -1, "expert": 64}, 16, 64),   # EP=64, batch=16, n_groups=64
            
            # Mixed TP+EP configurations
            ({"fsdp": -1, "model": 4, "expert": 16}, 16, 16),  # TP=4×EP=16, batch=16, n_groups=16
        ]
        
        for mesh_spec, batch, n_groups in mesh_configs:
            for top_k in [1, 8]:
                test_kwargs = kwargs.copy()
                test_kwargs['n_groups'] = n_groups
                grid_space.append(self.create_test_config(**test_kwargs, top_k=top_k, batch=batch, mesh_spec=mesh_spec))
        
        return grid_space

    def build_grid_space_12B(self):
        grid_space = []
        kwargs = {
            'dtype': jnp.bfloat16,
            'input_dim': 2048,
            'hidden_dim': 7168,
        }
        
        #Keeping existing TP-only
        
        # Base test
        # grid_space.append(self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}))
        
        # Top-K variations (TP=4 only)
        grid_space.extend([
            self.create_test_config(**kwargs, n_experts=8, top_k=1, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # self.create_test_config(**kwargs, n_experts=8, top_k=4, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
        ])
        
        # Sequence length variations (TP=4 only)
        grid_space.extend([
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=256, mesh_spec={"fsdp":-1, "model":4}),
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=2048, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=8192, mesh_spec={"fsdp":-1, "model":4}),
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=16*1024, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=32*1024, mesh_spec={"fsdp":-1, "model":4}),
        ])
        
        # TP scaling tests (keep existing n_groups=2)
        grid_space.extend([
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=4, seq=4096, mesh_spec={"fsdp":-1, "model":16}),
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=1, seq=4096, mesh_spec={"fsdp":-1, "model":64}),
        ])
        
        # Expert/group variations (TP=4 only) - All 4 original tests
        grid_space.extend([
            self.create_test_config(**kwargs, n_experts=1, top_k=1, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=7, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=4, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
        ])
        
        #EP-specific tests
        ep_test_configs = [
            # Basic EP functionality
            ({"fsdp": -1, "expert": 16}, 16, 16, 16, 2),   # EP=16, n_experts=16, top_k=2
            
            # EP with different expert counts (similar to original n_experts variations)
            ({"fsdp": -1, "expert": 16}, 16, 16, 32, 2),  # EP=16, n_experts=32, top_k=2, is expert=32 allowed?
            
            # Mixed TP+EP test
            ({"fsdp": -1, "model": 4, "expert": 16}, 16, 16, 16, 2),  # TP×EP=64
            
            # EP=64 test
            ({"fsdp": -1, "expert": 64}, 16, 64, 128, 2),   # EP=64, n_experts=128, top_k=2
        ]
        
        for mesh_spec, batch, n_groups, n_experts, top_k in ep_test_configs:
            test_kwargs = kwargs.copy()
            test_kwargs['n_groups'] = n_groups
            grid_space.append(self.create_test_config(**test_kwargs, n_experts=n_experts, top_k=top_k, capacity_factor=2, batch=batch, seq=4096, mesh_spec=mesh_spec))
        
        return grid_space

    def build_grid_space_50B(self):
        # Grid space for testing
        grid_space = []
        kwargs={
            'dtype': jnp.bfloat16,
            'input_dim': 4096,
            'hidden_dim': 14336,
        }

        grid_space.extend([
            # base
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # topk changes
            self.create_test_config(**kwargs, n_experts=8, top_k=1, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # self.create_test_config(**kwargs, n_experts=8, top_k=4, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # seqlen changes
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=256, mesh_spec={"fsdp":-1, "model":4}),
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=2048, mesh_spec={"fsdp":-1, "model":4}),
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=8192, mesh_spec={"fsdp":-1, "model":4}),
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=16*1024, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=32*1024, mesh_spec={"fsdp":-1, "model":4}),

            # tp8
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=8, seq=4096, mesh_spec={"fsdp":-1, "model":8}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=4, seq=4096, mesh_spec={"fsdp":-1, "model":16}),
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=2, seq=4096, mesh_spec={"fsdp":-1, "model":32}),
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=1, seq=4096, mesh_spec={"fsdp":-1, "model":64}),

            # num experts
            self.create_test_config(**kwargs, n_experts=1, top_k=1, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=7, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # num groups
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=4, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
        ])

        #EP-specific tests 
        ep_test_configs = [
            # Basic EP functionality
            ({"fsdp": -1, "expert": 16}, 16, 16, 16, 2),   # EP=16, n_experts=16, top_k=2
            ({"fsdp": -1, "expert": 16}, 16, 16, 16, 4),   # EP=16, n_experts=16, top_k=4
            
            # Basic EP with 32 expert counts
            ({"fsdp": -1, "expert": 16}, 16, 16, 32, 2),   # EP=16, n_experts=32, top_k=2
            ({"fsdp": -1, "expert": 16}, 16, 16, 32, 4),   # EP=16, n_experts=32, top_k=4
            
            # Mixed TP+EP test
            ({"fsdp": -1, "model": 4, "expert": 16}, 16, 16, 16, 2),  # TP=4×EP=16, n_experts=16
            ({"fsdp": -1, "model": 4, "expert": 16}, 16, 16, 16, 4),  # TP=4×EP=16, n_experts=16, topk=4
            
            # EP=64 test
            ({"fsdp": -1, "expert": 64}, 16, 64, 64, 2),   # EP=64, n_experts=64, top_k=2
            ({"fsdp": -1, "expert": 64}, 16, 64, 64, 4),   # EP=64, n_experts=64, top_k=4
        ]
        
        for mesh_spec, batch, n_groups, n_experts, top_k in ep_test_configs:
            test_kwargs = kwargs.copy()
            test_kwargs['n_groups'] = n_groups
            grid_space.append(self.create_test_config(**test_kwargs, n_experts=n_experts, top_k=top_k, capacity_factor=2, batch=batch, seq=4096, mesh_spec=mesh_spec))
        
        return grid_space

    def build_grid_space_150B(self):
        # Grid space for testing
        kwargs={
            'dtype': jnp.bfloat16,
            'input_dim': 6144,
            'hidden_dim': 15360,
        }
        grid_space = []
        grid_space.extend([
            # base
            # self.create_test_config(**kwargs, n_experts=16, top_k=4, n_groups=1, capacity_factor=2, batch=4, seq=8192, mesh_spec={"fsdp":-1, "model":16}),
            # topk changes
            self.create_test_config(**kwargs, n_experts=16, top_k=1, n_groups=1, capacity_factor=2, batch=4, seq=8192, mesh_spec={"fsdp":-1, "model":16}),
            # self.create_test_config(**kwargs, n_experts=16, top_k=2, n_groups=1, capacity_factor=2, batch=4, seq=8192, mesh_spec={"fsdp":-1, "model":16}),
            self.create_test_config(**kwargs, n_experts=16, top_k=8, n_groups=1, capacity_factor=2, batch=4, seq=8192, mesh_spec={"fsdp":-1, "model":16}),
            # capf change
            self.create_test_config(**kwargs, n_experts=32, top_k=8, n_groups=1, capacity_factor=4, batch=4, seq=8192, mesh_spec={"fsdp":-1, "model":16}),

            # seqlen changes
            # using 8x20b
            # self.create_test_config(dtype=jnp.bfloat16, input_dim=8192, hidden_dim=16384, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=4, seq=256, mesh_spec={"fsdp":-1, "model":16}),
            # self.create_test_config(dtype=jnp.bfloat16, input_dim=8192, hidden_dim=16384, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=4, seq=2048, mesh_spec={"fsdp":-1, "model":16}),
            # self.create_test_config(dtype=jnp.bfloat16, input_dim=8192, hidden_dim=16384, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=4, seq=4096, mesh_spec={"fsdp":-1, "model":16}),
            self.create_test_config(dtype=jnp.bfloat16, input_dim=8192, hidden_dim=16384, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=4, seq=8192, mesh_spec={"fsdp":-1, "model":16}),
            # self.create_test_config(dtype=jnp.bfloat16, input_dim=8192, hidden_dim=16384, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=4, seq=16*1024, mesh_spec={"fsdp":-1, "model":16}),
            self.create_test_config(dtype=jnp.bfloat16, input_dim=8192, hidden_dim=16384, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=1, seq=32*1024, mesh_spec={"fsdp":-1, "model":64}),

            # tp changes
            # self.create_test_config(**kwargs, n_experts=16, top_k=2, n_groups=2, capacity_factor=2, batch=8, seq=4096, mesh_spec={"fsdp":-1, "model":8}),
            # self.create_test_config(**kwargs, n_experts=16, top_k=2, n_groups=1, capacity_factor=2, batch=16, seq=2048, mesh_spec={"fsdp":-1, "model":4}),
            # self.create_test_config(**kwargs, n_experts=16, top_k=2, n_groups=2, capacity_factor=2, batch=2, seq=4096, mesh_spec={"fsdp":-1, "model":32}),
            self.create_test_config(**kwargs, n_experts=16, top_k=2, n_groups=1, capacity_factor=2, batch=1, seq=4096, mesh_spec={"fsdp":-1, "model":64}),
            
            # num groups
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=4, capacity_factor=2, batch=4, seq=4096, mesh_spec={"fsdp":-1, "model":16}),

            # num experts
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}), # not-needed

            #batch per TP-group
            self.create_test_config(dtype=jnp.bfloat16, input_dim=8192, hidden_dim=16384, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=8, seq=4096, mesh_spec={"fsdp":-1, "model":16}),
            self.create_test_config(dtype=jnp.bfloat16, input_dim=8192, hidden_dim=16384, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":16}),
            ])
            
        #EP test cases
        ep_test_configs = [
            # EP=16 tests
            ({"fsdp": -1, "expert": 16}, 16, 16, 16, 4),   # EP=16, n_experts=16, top_k=4
            ({"fsdp": -1, "expert": 16}, 16, 16, 32, 2),   # EP=16, n_experts=32, top_k=2
            
            # Mixed TP+EP test  
            ({"fsdp": -1, "model": 4, "expert": 16}, 16, 16, 16, 4),  # TP=4×EP=16, n_experts=16
            
            # EP=64 test
            ({"fsdp": -1, "expert": 64}, 16, 64, 64, 2),   # EP=64, n_experts=64, top_k=2
            ({"fsdp": -1, "expert": 64}, 16, 64, 64, 4),   # EP=64, n_experts=64, top_k=4
            ]

        for mesh_spec, batch, n_groups, n_experts, top_k in ep_test_configs:
            test_kwargs = kwargs.copy()
            test_kwargs['n_groups'] = n_groups
            grid_space.append(self.create_test_config(**test_kwargs, n_experts=n_experts, top_k=top_k, capacity_factor=2, batch=batch, seq=8192, mesh_spec=mesh_spec))
        
        return grid_space

def get_gating_config(gating_cls, num_experts, top_k, train_capacity_factor, expert_capacity, block_size=None, name=None, mesh_spec=None):

    cfg = gating_cls.default_config()
    if name:
        cfg.set(name=name)
    cfg.top_k = top_k
    cfg.train_capacity_factor = train_capacity_factor
    cfg.expert_capacity = expert_capacity
    cfg.num_experts = num_experts
    if mesh_spec:
        cfg.dim_to_mesh_axis_map=get_moe_dim_to_mesh_axis_map(mesh_spec.get("expert", 1), mesh_spec.get("model", 1), mesh_spec.get("seq", 1))
    else:
        cfg.dim_to_mesh_axis_map=get_moe_dim_to_mesh_axis_map(1,1,1)
    if block_size is not None and isinstance(cfg, TopKGatingGatherBlockwise.Config):
        cfg.block_size = block_size
    return cfg

def create_moe_test_config(test, golden, test_device, golden_device, input_dim, hidden_dim, n_experts, top_k, n_groups, capacity_factor, mesh_spec, batch, seq, dtype, block_size, model_param_init, name=None):
    test_cfg = TransformerFeedForwardMoE.default_config().set(
            name="test" if name is None else name,
            param_init=model_param_init
        )
    test_cfg.input_dim = input_dim
    test_cfg.hidden_dim = hidden_dim
    if mesh_spec:
        test_cfg.dim_to_mesh_axis_map=get_moe_dim_to_mesh_axis_map(mesh_spec.get("expert", 1), mesh_spec.get("model", 1), mesh_spec.get("seq", 1))
    else:
        test_cfg.dim_to_mesh_axis_map=get_moe_dim_to_mesh_axis_map(1,1,1)
    test_cfg.activation = ("nn.silu","linear")
    test_cfg.num_experts = n_experts
    test_cfg.num_groups = n_groups
    # enabling nonorm gives us better check of the kernel logits, what's missing here is just add of residual

    test_cfg.structure = "nonorm"
    test_cfg.gating = get_gating_config(test, n_experts, top_k, capacity_factor, expert_capacity=None, block_size=block_size, mesh_spec=mesh_spec)

    if golden:
        golden_cfg = test_cfg.clone(name="golden" if name is None else name)
        golden_cfg.gating = get_gating_config(golden, n_experts, top_k, capacity_factor, expert_capacity=None, mesh_spec=mesh_spec)
    else:
        golden_cfg = None
    return test_cfg, golden_cfg

def create_test_config(test, golden, test_device, golden_device, input_dim, hidden_dim, n_experts, top_k, n_groups, capacity_factor, mesh_spec, batch, seq, dtype, block_size=512, layer='moe'):
    """
    Ensure any new param added here also shows up in the name to prevent multiple tests from having same name.
    You will see an exception calling that out if it happens.
    """

    model_param_init = DefaultInitializer.default_config().set(
        init_by_param_name={
            PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                fan="fan_in", distribution="normal"
            )
        }
    )

    if layer == "moe":
        conv_output = None
        test_cfg, golden_cfg = create_moe_test_config(
            test, golden, test_device, golden_device, input_dim, hidden_dim, n_experts, top_k, n_groups, capacity_factor, mesh_spec, batch, seq, dtype, block_size, model_param_init
        )
    elif layer == "transformer":
        test_cfg = TransformerLayer.default_config().set(
            name="test",
            param_init=model_param_init,
            input_dim=input_dim,
        )
        # RoPE embeddings: https://arxiv.org/abs/2104.09864.
        attention_qkv_linear = RoFormerQKVLinear.default_config().set(
            input_linear=GroupedQKVLinear.default_config().set(
                num_kv_heads=8,
            ),
            rotary_value=False,
        )
        attention_qkv_linear.rope_pos_emb_layer.theta = 5e5
        norm_cfg = RMSNorm.default_config().set(eps=1e-5, forward_dtype=None)

        if False: #flash_attention
            test_cfg.self_attention.attention = flash_attention_config()
        else:
            test_cfg.self_attention.attention = GroupedQueryAttention.default_config()
        test_cfg.self_attention.attention.set(
            # Use q/k-norm in keeping with:
            # <https://arxiv.org/abs/2309.14322>
            query_scale=ScaleQuery.default_config().set(norm=norm_cfg.clone()),
            key_scale=ScaleKey.default_config().set(norm=norm_cfg.clone()),
        )
        test_cfg.self_attention.attention.input_linear = attention_qkv_linear

        test_cfg.self_attention.attention.causal = True
        test_cfg.self_attention.attention.num_heads = 8
        batch_axis_names = ("data", "expert", "fsdp")
        set_double_shard_weights_config(
            test_cfg,
            batch_axis_names=batch_axis_names,
            fsdp_axis_names=("expert", "fsdp", "seq"),
            tp_axis_names="model",
            seq_axis_names="seq",
        )
        test_moe_cfg, golden_moe_cfg = create_moe_test_config(test, golden, test_device, golden_device, input_dim, hidden_dim, n_experts, top_k, n_groups, capacity_factor, mesh_spec, batch, seq, dtype, block_size, model_param_init, name="feed_forward")
        test_cfg.feed_forward = test_moe_cfg
        test_cfg.feed_forward.gating.dim_to_mesh_axis_map = test_moe_cfg.dim_to_mesh_axis_map
        if golden:
            golden_cfg = test_cfg.clone(name="golden")
            golden_cfg.feed_forward = golden_moe_cfg
        else:
            golden_cfg = None
        conv_output = None
    else:
        test_cfg = get_gating_config(test, n_experts, top_k, capacity_factor, expert_capacity=None, name="test", block_size=block_size, mesh_spec=mesh_spec)

        if golden:
            golden_cfg = get_gating_config(golden, n_experts, top_k, capacity_factor, expert_capacity=None, name="golden", block_size=block_size)
        else:
            golden_cfg = None
        if test == TopKGatingGather and golden == TopKGating:
            conv_output = partial(_topkgather_to_topk, top_k=top_k, cf=capacity_factor)
        else:
            conv_output = None
    
    test_invoker_cfg = {
        "batch_size": batch,
        "seq_len": seq,
        "input_dim": input_dim,
        "dtype": jnp.bfloat16 if dtype in ["bfloat16", jnp.bfloat16] else jnp.float32,
        "device": test_device,
        "mesh_spec": mesh_spec,
        "input_shape": (batch, seq, input_dim) if layer in ["moe", "transformer"] else ('O', n_groups, 'S', n_experts),
    }
    if golden:
        golden_invoker_cfg = dict(test_invoker_cfg)
        golden_invoker_cfg['device'] = golden_device
    else:
        golden_invoker_cfg = {}
    
    config = ExperimentConfig(
        test_cfg, 
        golden_cfg, 
        test_invoker_cfg, 
        golden_invoker_cfg,
        loss_fn=lambda x: jnp.mean(x)*1e2,
        conv_output=conv_output,
        prefix="_" + layer
    )
    return (build_name(test_cfg, test_invoker_cfg), config)

@cache
def get_gating_configs(test_suite="presubmit", layer='moe', test=TopKGatingGather, golden=TopKGating, test_device="neuron", golden_device="cpu"):
    builder = GridSpaceBuilder(layer=layer, test=test, golden=golden, test_device=test_device, golden_device=golden_device)
    if test_suite == 'presubmit':
        return builder.build_presubmit_grid_space()
    else:
        # dummy to avoid errors, can't have empty grid space
        return builder.build_presubmit_grid_space()[:1]


@cache
def get_training_configs(test_suite="presubmit", layer='moe', test=TopKGatingGather, golden=TopKGating, test_device="neuron", golden_device="cpu"):
    builder = GridSpaceBuilder(layer=layer, test=test, golden=golden, test_device=test_device, golden_device=golden_device)
    if test_suite == "toy":
        return builder.build_toy_grid_space()
    elif test_suite == 'presubmit':
        tests = builder.build_presubmit_grid_space()
    elif test_suite == '12b':
        # Fixed: Use tests= instead of return to enable proper partitioning and avoid test duplication
        tests = builder.build_grid_space_12B()
    elif test_suite == '50b':
        # Fixed: Use tests= instead of return for consistency with other test suites
        tests = builder.build_grid_space_50B()
    elif test_suite == '150b':
        tests = builder.build_grid_space_150B()
    elif test_suite == 'qwen3-30b':
        tests = builder.build_grid_space_input_hidden(input_dim=2048, hidden_dim=6144, max_E=128, exclude_combinations=[(16,1,1)], exclude_seq_lengths=[16384])
    elif test_suite == 'switch-base':
        tests = builder.build_grid_space_input_hidden(input_dim=1536, hidden_dim=6144, max_tp=16, exclude_combinations=[(16,1,1)], exclude_seq_lengths=[16384])
    elif test_suite == 'switch-large':
        tests = builder.build_grid_space_input_hidden(input_dim=2048, hidden_dim=8192, max_tp=16, max_E=128)
    elif test_suite == 'mixtral-50b':
        tests = builder.build_grid_space_input_hidden(input_dim=4096, hidden_dim=14336, max_E=16, max_tp=16)
    elif test_suite == 'llama4-scout':
        # llama4 scout (topk=1, E=16)
        tests = builder.build_grid_space_input_hidden(input_dim=5120, hidden_dim=8192, max_E=64, max_tp=16)
    elif test_suite == 'deepseek-v3':
        tests = builder.build_grid_space_input_hidden(input_dim=7168, hidden_dim=2048, max_E=128, max_tp=16, exclude_combinations=[(16,1,1)], exclude_seq_lengths=[16384])
    # below are too big, takes too long to run, and many tests go CPU OOM if we do grid like for above configs
    elif test_suite == 'qwen3-235b':
        tests = builder.build_grid_space_qwen3_235b()
    elif test_suite == 'switch-xxl':
        tests = builder.build_grid_space_switch_xxl()
    elif test_suite == 'llama4-maverick':
        tests = builder.build_grid_space_llama4_maverick()
    elif test_suite == 'gpt-oss':
        tests = builder.build_grid_space_gpt_oss()
    else:
        raise ValueError(f"Unknown test suite: {test_suite}")

    test_suite_part = int(os.getenv('TEST_SUITE_PART', 0))
    test_suite_parts = int(os.getenv('TEST_SUITE_PARTS', 1))
    print(f'Part[{test_suite_part}/{test_suite_parts}] of test suite {test_suite} with {len(tests)} tests')
    part_size = len(tests)//test_suite_parts
    tests = tests[part_size*test_suite_part:part_size*(test_suite_part+1)]
    print('Candidate tests', [x[0] for x in tests])
    TEST_LOG_DIR = os.getenv('TEST_LOG_DIR', None)
    # check if we need to resume tests
    # by looking for any xmls in path
    matches = glob.glob(os.path.join(TEST_LOG_DIR, test_suite, "integ_*.xml"))
    if matches:
        tests_to_resume = []
        failed_tests = set()
        all_tests = set()
        for m in matches:
            # load each xml result and list test names
            results = parse_pytest_xml(m)
            for r in results['failures']:
                if len(r[0].split('_MoE')) > 1:
                    failed_tests.add('MoE' + r[0].split('_MoE')[1])
                else:
                    failed_tests.add('MoE' + r[0])
            for r in results['all_tests']:
                if len(r[0].split('_MoE')) > 1:
                    all_tests.add('MoE' + r.split('_MoE')[1])
                else:
                    all_tests.add('MoE' + r)

        print('Failed tests', failed_tests)
        for t in tests:
            if t[0] in failed_tests or t[0] not in all_tests:
                tests_to_resume.append(t)
        tests = tests_to_resume
        print('Filtered tests', tests)
    else:
        print(f"No previous results found in {TEST_LOG_DIR} for {test_suite}, running all tests")
    if tests:
        return tests
    else:
        # dummy test as we can't return no test
        return []