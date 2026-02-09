#!/usr/bin/env bash
set -e
# Skip kernel module reload in Kaizen (containerized) environment
if [ -z "${WORKLOAD_OUTPUT_RW}" ]; then
    sudo rmmod neuron; sudo modprobe neuron
fi
if [ -z "$VENV_NAME" ]; then
	VENV_NAME=../jaxmoe
fi

source $VENV_NAME/bin/activate

export TEST_SUITE=${2:-"presubmit"}
export TEST_LOG_DIR=${3:-"test_artifacts/shell"}
export GOLDENS_DIR=${4:-"/shared/huilgolr/axlearn/test_goldens"}
export JAX_COMPILATION_CACHE_DIR=${5:-"test_artifacts/shell_jax_cc"}
rm -rf $JAX_COMPILATION_CACHE_DIR
rm -rf $TEST_LOG_DIR

# if defined
if [ -n "$1" ]; then
    mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
fi
mkdir -p "${GOLDENS_DIR}"

export TEST_ARTIFACTS_PATH=$TEST_LOG_DIR/$TEST_SUITE/artifacts
export NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
mkdir -p "$TEST_ARTIFACTS_PATH"

export USE_CACHED_GOLDENS=1
export CACHE_GOLDENS=0
export USE_SHARDMAP_FFN=1
export NEURON_HLO_ANALYZER=1
export XLA_FLAGS="--xla_cpu_use_thunk_runtime=false --xla_force_host_platform_device_count=64 --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives"

export GIT_COMMIT=$(git rev-parse --short HEAD)

HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump
export XLA_FLAGS="${XLA_FLAGS} --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*' --xla_dump_hlo_as_text --xla_dump_hlo_as_proto"
# export XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_snapshots"

# PJRT Flags 
export NEURON_FSDP_NUM_LAYER_EARLY_AG_SHIFT=1
export NEURON_FSDP_NUM_LAYER_LATE_RS_SHIFT=2
export NEURON_ENABLE_INT_MATMUL_DOWNCAST=1
export NEURON_FSDP=0
export NEURON_FSDP_NUM_LAYER_COALESCE=-1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1 # changed from 0
export NEURON_DISABLE_BOUNDARY_MARKER=1
export NEURON_COLLECTIVE_PERMUTE_TO_ALL_GATHER=1
# Neuron runtime flags
export NEURON_RT_DBG_CC_DMA_PACKET_SIZE=4096 && export NEURON_RT_DBG_DMA_PACKETIZATION_SIZE=104857
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0
export NEURON_RT_IO_RING_CACHE_SIZE=0
export NEURON_RT_ENABLE_MEMORY_METRICS=0
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="ERROR"
export NEURON_RT_ENABLE_INTERNODE_EXECUTION_BARRIER=1

# Neuron collectives flag
export FI_LOG_LEVEL="warn"
export OFI_NCCL_PROTOCOL=RDMA
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/:${LD_LIBRARY_PATH}"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1
export OFI_NCCL_MR_CACHE_DISABLE=1
# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-max-instruction-limit=20000000"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options=--verify-hlo"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --tensorizer-options='--enable-hoist-fsdp-collectives'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--remat-rope --verify-hlo'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --auto-cast=none" # --hbm-scratchpad-page-size=1024"
export TF_CPP_MIN_LOG_LEVEL=3

if [ "$1" = "unit" ]; then
    export JAX_PLATFORMS=cpu
    pytest -rsA --tb=short --junitxml=$TEST_LOG_DIR/$TEST_SUITE/unit.xml aws-testing/moe_layer_unit_test.py
elif [ "$1" = "integ" ]; then
    # breaking them up as we seem to leak memory across tests
    if [ "$2" = "50b" ]; then
        export TEST_SUITE_PARTS=1
    elif [ "$2" = "12b" ]; then
        export TEST_SUITE_PARTS=6
    elif [ "$2" = "gpt-oss" ]; then
        export TEST_SUITE_PARTS=3
    elif [ "$2" = "150b" ]; then
        export TEST_SUITE_PARTS=14
    elif [ "$2" = "deepseek-v3" ] || [ "$2" = "qwen3-30b" ] || [ "$2" = "switch-base" ]; then
        export TEST_SUITE_PARTS=15
    else
        export TEST_SUITE_PARTS=10
    fi
    echo "Splitting tests into $TEST_SUITE_PARTS parts"
    status=0
    set +e
    for ((part=0;part<TEST_SUITE_PARTS;part++)); do
        set -x
        if [ "$3" = "collect-only" ]; then
            TEST_SUITE_PART=$part pytest --collect-only -q aws-testing/moe_layer_integ_test.py -k "TestLayerOnTrn"
        else
            TEST_SUITE_PART=$part pytest -rsA --tb=short --junitxml=$TEST_LOG_DIR/$TEST_SUITE/integ_$part.xml aws-testing/moe_layer_integ_test.py -k "TestLayerOnTrn"
        fi
        status_part=$?
        status=$((status + status_part))
        set +x
    done
    if [ $status -ne 0 ]; then
        exit 1
    fi
elif [ "$1" = "dev" ]; then
    pytest -rsA -v aws-testing/gating_test.py -k "TestSwitchBaseGatingUnit or TestDev150bGatingUnit"
    # pytest -rsA -v aws-testing/moe_layer_integ_test.py -k "TestDev150bInteg and test_fwdbwd_blockwisev2"
    # pytest -rsA -v aws-testing/moe_layer_integ_test.py -k "TestDevSwitchBaseInteg and test_fwd_blockwise_ep4_seq4_model4" # or TestDevSwitchBaseInteg and test_fwdbwd_blockwise_ep4_seq16"
    # pytest -rsA -v aws-testing/transformer_layer_integ_test.py -k "TestDevSwitchBaseInteg and test_fwdbwd_transformer" # or TestDevSwitchBaseInteg and test_fwdbwd_blockwise_ep4_seq16"
fi
