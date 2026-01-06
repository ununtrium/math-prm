# Best Aggregation Method Summary

| Benchmark    | Samples    | PRM                                 | BEST_METHOD   |   BEST_SCORE |   MajVote |
|:-------------|:-----------|:------------------------------------|:--------------|-------------:|----------:|
| aime24       | samples_16 | orm_1.5b_raw_only_ckpt4054          | LAST          |        17.33 |     18    |
| aime24       | samples_16 | prm_1.5b_ensemble_raw_only_ckpt4054 | LAST          |        14.67 |     18    |
| aime24       | samples_16 | prm_1.5b_qwen_raw_only_ckpt4054     | LAST          |        13.33 |     18    |
| aime24       | samples_16 | prm_1.5b_llama_raw_only_ckpt4054    | MEAN          |        13.33 |     18    |
| aime24       | samples_16 | prm_1.5b_deepseek_raw_only_ckpt4054 | LAST          |        12    |     18    |
| aime24       | samples_64 | orm_1.5b_raw_only_ckpt4054          | MIN           |        20    |     16.67 |
| aime24       | samples_64 | prm_1.5b_llama_raw_only_ckpt4054    | MEAN          |        16.67 |     16.67 |
| aime24       | samples_64 | prm_1.5b_qwen_raw_only_ckpt4054     | MEAN          |        16    |     16.67 |
| aime24       | samples_64 | prm_1.5b_deepseek_raw_only_ckpt4054 | MEAN          |        16    |     16.67 |
| aime24       | samples_64 | prm_1.5b_ensemble_raw_only_ckpt4054 | MEAN          |        12.67 |     16.67 |
| aime25       | samples_16 | prm_1.5b_ensemble_raw_only_ckpt4054 | LAST          |        14.67 |     19.33 |
| aime25       | samples_16 | prm_1.5b_qwen_raw_only_ckpt4054     | LAST          |        14    |     19.33 |
| aime25       | samples_16 | orm_1.5b_raw_only_ckpt4054          | MIN           |        12.67 |     19.33 |
| aime25       | samples_16 | prm_1.5b_llama_raw_only_ckpt4054    | LAST          |        12    |     19.33 |
| aime25       | samples_16 | prm_1.5b_deepseek_raw_only_ckpt4054 | SUM           |        12    |     19.33 |
| math500      | samples_16 | orm_1.5b_raw_only_ckpt4054          | MEAN          |        61.2  |     62.12 |
| math500      | samples_16 | prm_1.5b_llama_raw_only_ckpt4054    | MEAN          |        60.72 |     62.12 |
| math500      | samples_16 | prm_1.5b_ensemble_raw_only_ckpt4054 | MEAN          |        60.6  |     62.12 |
| math500      | samples_16 | prm_1.5b_deepseek_raw_only_ckpt4054 | MEAN          |        60.44 |     62.12 |
| math500      | samples_16 | prm_1.5b_qwen_raw_only_ckpt4054     | LAST          |        60.36 |     62.12 |
| math500      | samples_64 | prm_1.5b_llama_raw_only_ckpt4054    | LAST          |        60.84 |     62.48 |
| math500      | samples_64 | prm_1.5b_deepseek_raw_only_ckpt4054 | MEAN          |        60.68 |     62.48 |
| math500      | samples_64 | orm_1.5b_raw_only_ckpt4054          | MIN           |        60.52 |     62.48 |
| math500      | samples_64 | prm_1.5b_ensemble_raw_only_ckpt4054 | LAST          |        60.16 |     62.48 |
| math500      | samples_64 | prm_1.5b_qwen_raw_only_ckpt4054     | MEAN          |        59.16 |     62.48 |
| numina_train | samples_16 | orm_1.5b_raw_only_ckpt4054          | LAST          |        43.25 |     44.35 |
| numina_train | samples_16 | prm_1.5b_ensemble_raw_only_ckpt4054 | MEAN          |        43.25 |     44.35 |
| numina_train | samples_16 | prm_1.5b_deepseek_raw_only_ckpt4054 | SUM           |        43    |     44.35 |
| numina_train | samples_16 | prm_1.5b_qwen_raw_only_ckpt4054     | MEAN          |        42.85 |     44.35 |
| numina_train | samples_16 | prm_1.5b_llama_raw_only_ckpt4054    | LAST          |        42.6  |     44.35 |