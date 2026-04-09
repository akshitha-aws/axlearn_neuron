#!/bin/bash
# comment this out to run full model
# fsdp not implemented - need to do it for many nodes

export AXLEARN_NUM_LAYERS=1
export AXLEARN_REMAT_LAYER=selective
export AXLEARN_MODEL_NAME="envy-Switch-Base"
export AXLEARN_TP_DEGREE=4
export AXLEARN_EP_DEGREE=4
export AXLEARN_SEQ_DEGREE=4
export AXLEARN_TRAIN_BATCH_SIZE=4
# use v2 index calc
export AXLEARN_USE_BLOCKWISE=2
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_DISABLE_MOVEMENT_OF_SLICE_FROM_PARAM=1
# Force block size to be compatible with Neuron

# - 0 (dense) - 1 (sparse) - 2 (alternating)
export AXLEARN_MOE_LAYER_FREQ=1
export AXLEARN_FSDP_DEGREE=1
export EP_WITHIN_NODE=1
# export AXLEARN_PROFILE_MODE="tracerun"
# export NEURON_RT_LOCAL_CORE_DUMP_DIRECTORY=""

# export NEURON_PROFILE=/fsx/akshiaws/artifacts/$JOB_ID/ntff_output

if [ "${AXLEARN_SEQ_DEGREE:-0}" -gt 1 ]; then
    export AXLEARN_FLASH_ATTENTION=0
else
    export AXLEARN_FLASH_ATTENTION=1
fi

export AXLEARN_REPEATED=1
# it expects the env to be at ../$VENV_NAME
export VENV_NAME=akshiaws/jaxmoe3

# to simulate slurm job run
export SLURM_PROCID=0
# to output artifacts at this path ./artifacts/JOB_ID/
export JOB_ID=base
rm -rf /fsx/akshiaws/artifacts/$JOB_ID/

bash /fsx/akshiaws/axlearn_neuron/runner.sh
2>&1 | tee log_$JOB_ID.out
