#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --job-name=mxnettest
#SBATCH --mem-per-cpu=500M
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/work/moisioa3/conv_lm/mlm-scoring/examples/asr-convfin-finbert/log/%x-%j.out

module purge
module load cuDNN
module load miniconda
module list

source activate mlm-scoring

# cd ../..
# pip install -e .
# pip list
# pip uninstall mxnet-cu110
# pip install mxnet-cu90
# pip list

# cd examples/asr-convfin-finbert

# source activate mlm-scoring3
# pip list

# cd mlm-scoring
# pip install --user -e .
# cd ..
# pip install --user mxnet-cu110

python mxnet_test.py

mlm rescore --help
