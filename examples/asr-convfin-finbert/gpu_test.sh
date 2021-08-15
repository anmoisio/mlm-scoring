#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --job-name=mxnettest
#SBATCH --mem-per-cpu=100M
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1


module load cuDNN
module load miniconda
module list

source activate mlm-scoring3
pip list

# cd mlm-scoring
# pip install --user -e .
# cd ..
# pip install --user mxnet-cu110

python mxnet_test.py
# mlm rescore --model "TurkuNLP/bert-base-finnish-cased-v1" \
#     mlm-scoring/examples/asr-librispeech-espnet/data/dev-other.am.json \
#     mlm-scoring/examples/demo/dev-other-3.lm.json \
#     > dev-other-3.lambda.json



# mlm score --model "TurkuNLP/bert-base-finnish-cased-v1" \
#     --mode hyp \
#     ../../../keskustelu2020/experiments/am/converse_fin/exp/chain/tdnn7q_sp_ensemble_vc/50best_devel_morph_finbert_nosp_4-gram/ac.ac0.1.json \
#     > log/scoretestdevel50.ac0.1.json

# for weight in 0.1 0.5 0.9 2
# do
#     echo weight $weight
#     mlm rescore --model "TurkuNLP/bert-base-finnish-cased-v1" \
#         --weight ${weight} \
#         ../keskustelu2020/experiments/am/converse_fin/exp/chain/tdnn7q_sp_ensemble_vc/50best_devel_morph_finbert_nosp_4-gram/ac.ac0.1.json \
#         scoretestdevel50.ac0.1.json \
#         > finberttestdevel50.ac0.1.${weight}.json
# done