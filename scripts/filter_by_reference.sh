# 例: Qwen単体の2Mデータを、Ensemble版ハーフと同じ問題セットに絞り込む
python3 src/filter_by_reference.py \
    --reference_file data/prm_train_ensemble_1M.jsonl \
    --target_file data/numinamath_gen_30k.jsonl \
    --output_file data/numinamath_gen_15k.jsonl