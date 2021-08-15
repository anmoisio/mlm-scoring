#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --job-name=mlmscore
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/work/moisioa3/conv_lm/mlm-scoring/examples/asr-convfin-finbert/log/%x-%j.out

module purge
module load cuDNN/7-CUDA-9.0.176
module load miniconda/4.9.2
module list

source activate mlm-scoring

stage=1

workdir="/scratch/work/moisioa3"
project_dir="${workdir}/conv_lm"
# decode_set="eval"
# lahjoitapuhetta-transcript-comparison
decode_set="momaf-movies-with-sub"
n=100

# 1st pass
am="tdnn7q_sp_ensemble_vc"
# ngram_lm="_morph_nosp_4-gram"
# ngram_lm="_morph_finbert_nosp_4-gram"
ngram_lm="_word_web_dsp_os_nosp"

nbest_dir="../../../../keskustelu2020/experiments/am/converse_fin/exp/chain/tdnn7q_sp_ensemble_vc/${n}best_${decode_set}${ngram_lm}"

# 2nd pass
# lstm_lm="/theanolm-morph-42k/expt2-sampled-seq40"
# nbest_name="lstm_rescored_${n}best_${decode_set}_${am}${ngram_lm}"
# nbest_dir="${project_dir}/experiments${lstm_lm}/${nbest_name}"

# 3rd pass
# finetune_set="web-dsp"
finetune_set="opensubtitles_all"

cased="cased"
# bert_model="TurkuNLP/bert-base-finnish-cased-v1"
bert_model="TurkuNLP/bert-base-finnish-${cased}-v1-finetuned-${finetune_set}"
# bert_lm="${project_dir}/finbert-finetune/${bert_model}/checkpoint-20500"

# train_step_checkpoints="40000 60000 82000"
train_step_checkpoints="30000 40000"
# ac_scales="0.01 0.03 0.08 0.1 0.13 0.2 0.3"
ac_scale="0.083"
weight=1

# TODO paste convert-nbest-mlmscoring
# get n-best from lattices
# if [ $stage -le 0 ]; then
# fi

# # convert n-best lists to mlm-scoring format
# if [ $stage -le 1 ]; then
# fi

# rescore n-best list with BERT
for train_steps in $train_step_checkpoints
do
    bert_lm="${project_dir}/finbert-finetune/${bert_model}/checkpoint-${train_steps}"
    score_output="scored/${am}${ngram_lm}${lstm_lm}/${n}best_${decode_set}_finbert_ft_${cased}_${finetune_set}-checkpoint-${train_steps}.combined.ac${ac_scale}.json"
    mkdir -p "scored/${am}${ngram_lm}${lstm_lm}"

    if [ $stage -le 1 ]; then
        echo using ${n}-best list from "$nbest_dir"
        echo scoring ${n}-best list using $bert_lm
        echo write output to $score_output
        mlm score \
            --model "${bert_lm}" \
            --mode hyp \
            "${nbest_dir}/ac_combined.ac_scale${ac_scale}.json" \
            > "$score_output"
    fi

    if [ $stage -le 2 ]; then
        
        echo "$nbest_dir"
        echo rescoring "$bert_lm"
        # for ac_scale in $ac_scales
        # do
        echo ac scale $ac_scale
        rescore_output="scored/rescored_${nbest_name}_bert-base-finnish-${cased}-v1-finetuned-${finetune_set}-checkpoint-${train_steps}.combined.ac${ac_scale}.${weight}.json"
        mlm rescore \
            --model "$bert_lm" \
            --weight $weight \
            "${nbest_dir}/ac_combined.ac_scale${ac_scale}.json" \
            "$score_output" \
            > "$rescore_output"
        # done
    fi

    # if [ $stage -le -3 ]; then
    #     module load kaldi-2020/5968b4c-GCC-6.4.0-2.28-OPENBLAS
    #     module list

    #     for ac_scale in $ac_scales
    #     do
    #         rescore_output="scored/rescored_lstm_${nbest_name}_bert-base-finnish-${cased}-v1-finetuned-${finetune_set}-checkpoint-${train_steps}.combined.ac${ac_scale}.${weight}.json"
    #         hypothesis_file="${rescore_output%%.json}-converted.trn"

    #         python3 ${workdir}/keskustelu2020/scripts/convert-nbest-mlm2kaldi.py \
    #             "$rescore_output" \
    #             "$hypothesis_file"
        
    #         ref_file="${workdir}/keskustelu2020/experiments/am/converse_fin/exp/chain/tdnn7q_sp_ensemble_vc/decode_${decode_set}${ngram_lm}/scoring_kaldi/test_filt.txt"
    #         output_file="${rescore_output%%.json}-wer.txt"

    #         if [[ -f "${output_file}" ]]
    #         then
    #             echo WER output file "${output_file}" exists, not calculating WER
    #         else
    #             compute-wer --text --mode=present \
    #                 ark:"${ref_file}" \
    #                 ark,p:"${hypothesis_file}" \
    #                 > "${output_file}"
    #         fi
    #     done
    # fi
done