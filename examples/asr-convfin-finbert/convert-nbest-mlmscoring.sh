#!/bin/bash -e

n=100
# lmwt with the best WER in 4-gram decoding phase
lmwt=11
dir=../../../../keskustelu2020/experiments/am/converse_fin/exp/chain/tdnn7q_sp_ensemble_vc

vocab=_morph_finbert_nosp
# vocab=_morph_nosp

# decode_set=devel
decode_set=eval
# decode_set=lahjoitapuhetta-transcript-comparison

nbest_dir=${dir}/${n}best_${decode_set}${vocab}_4-gram
# nbest_dir=${dir}/${n}best_${decode_set}${vocab}_4-gram_rnnlm_rescored_combined
# nbest_dir=/m/triton/scratch/work/moisioa3/conv_lm/experiments/theanolm-morph-42k/expt2-sampled-seq40/lstm_rescored_100best_eval_tdnn7q_sp_ensemble2_morph_nosp_4-gram

# refs=/m/triton/scratch/work/moisioa3/keskustelu2020/experiments/am/converse_fin/exp/chain/tdnn7q_sp_ensemble_vc/decode_lahjoitapuhetta-transcript-comparison_morph_nosp_4-gram_rnnlm_rescored_combined/scoring_kaldi1/test_filt.txt 
refs=/m/triton/scratch/work/moisioa3/conv_lm/data/lm-train/segmented-finbert/eval_ids.txt

for ac_scale in 0.01 0.03 0.08 0.1 0.13 0.2 0.3
do
    python3 /m/triton/scratch/work/moisioa3/keskustelu2020/scripts/convert-nbest.py \
        --ac_scale $ac_scale \
        --combine \
        "${nbest_dir}" \
        "${refs}"
done
