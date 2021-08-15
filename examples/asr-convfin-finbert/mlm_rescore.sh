#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --job-name=mlmrescore
#SBATCH --mem-per-cpu=500M
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/work/moisioa3/conv_lm/mlm-scoring/examples/asr-convfin-finbert/log/%x-%j.out

module purge
module load cuDNN/7-CUDA-9.0.176
module load miniconda/4.9.2
module list

source activate mlm-scoring


project_dir="/scratch/work/moisioa3/conv_lm"
decode_set="eval"
n=100
# dir=../../../../keskustelu2020/experiments/am/converse_fin/exp/chain/tdnn7q_sp_ensemble_vc
# name=devel_morph_finbert_nosp_4-gram

am="tdnn7q_sp_ensemble2"
ngram_lm="_morph_nosp_4-gram"

# nbest_dir="../../../../keskustelu2020/experiments/am/converse_fin/exp/chain/tdnn7q_sp_ensemble_vc/${decode_set}${ngram_lm}"

# 2nd pass
lstm_lm="/theanolm-morph-42k/expt2-sampled-seq40"
nbest_name="lstm_rescored_${n}best_${decode_set}_${am}${ngram_lm}"
nbest_dir="${project_dir}/experiments${lstm_lm}/${nbest_name}"

finetune_set="web-dsp"
# bert_lm="${project_dir}/finbert-finetune/TurkuNLP/bert-base-finnish-cased-v1-finetuned-${finetune_set}/checkpoint-1000"


echo "${nbest_dir}"
echo "${name}"
for cp in 1000 10000
do
    bert_lm="${project_dir}/finbert-finetune/TurkuNLP/bert-base-finnish-cased-v1-finetuned-${finetune_set}/checkpoint-$cp"
    echo weight $bert_lm
    mlm rescore \
        --model $bert_lm \
        --weight 1 \
        ${nbest_dir}/ac_combined.ac_scale0.1.json \
        scored/${am}${ngram_lm}${lstm_lm}/${n}best_${decode_set}_finbert_ft_cased_${finetune_set}.combined.ac0.1.json \
        > scored/rescored_lstm_${nbest_name}_bert-base-finnish-cased-v1-finetuned-${finetune_set}-checkpoint-${cp}.combined.ac0.1.${weight}.json
done
# for weight in 0.5 1 1.5
# do
#     echo weight $weight
#     mlm rescore --model "TurkuNLP/bert-base-finnish-cased-v1" \
#         --weight ${weight} \
#         ${nbest_dir}/${name}/ac_combined.ac_scale0.1.json \
#         scored/${name}.combined.ac0.1.json \
#         > scored/rescored_${name}.combined.ac0.1.${weight}.json
# done


# for weight in 0.5 1 1.5
# do
#     echo weight $weight
#     mlm rescore --model "TurkuNLP/bert-base-finnish-cased-v1" \
#         --weight ${weight} \
#         ${dir}/${n}best_${name}/accombined.ac0.1.json \
#         scored/scoretestdevel${n}.combined.ac0.1.json \
#         > ${n}best_${name}.accombined0.1.${weight}.json
# done
