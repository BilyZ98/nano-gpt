#!/bin/bash
#source /path/to/conda/etc/profile.d/conda.sh
echo $CONDA_DEFAULT_ENV
#python test_cuda.py
# python gpt.py
export CUDA_LAUNCH_BLOCKING=1
# Only use 4 because default number of visible cuda devices is 4
torchrun --standalone  --nproc-per-node 4 persona_gpt.py
