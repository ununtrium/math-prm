# PRM Evaluation Report

| Benchmark   | Samples    | Model                               |   Seeds | Avg@k             | MajVote           | BoN (min)         | BoN (mean)        | BoN (last)        | BoN (sum)         |
|:------------|:-----------|:------------------------------------|--------:|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| aime24      | samples_16 | orm_1.5b_raw_only_ckpt4054          |       5 | 10.25 (Max:10.42) | 18.00 (Max:20.00) | 16.00 (Max:20.00) | 12.67 (Max:16.67) | 17.33 (Max:20.00) | 8.67 (Max:13.33)  |
| aime24      | samples_16 | prm_1.5b_qwen_raw_only_ckpt4054     |       5 | 10.25 (Max:10.42) | 18.00 (Max:20.00) | 11.33 (Max:13.33) | 12.67 (Max:13.33) | 13.33 (Max:16.67) | 5.33 (Max:10.00)  |
| aime24      | samples_16 | prm_1.5b_deepseek_raw_only_ckpt4054 |       5 | 10.25 (Max:10.42) | 18.00 (Max:20.00) | 11.33 (Max:13.33) | 11.33 (Max:13.33) | 12.00 (Max:16.67) | 10.00 (Max:13.33) |
| aime24      | samples_16 | prm_1.5b_llama_raw_only_ckpt4054    |       5 | 10.25 (Max:10.42) | 18.00 (Max:20.00) | 11.33 (Max:13.33) | 13.33 (Max:13.33) | 12.00 (Max:16.67) | 7.33 (Max:10.00)  |
| aime24      | samples_16 | prm_1.5b_ensemble_raw_only_ckpt4054 |       5 | 10.25 (Max:10.42) | 18.00 (Max:20.00) | 11.33 (Max:13.33) | 13.33 (Max:13.33) | 14.67 (Max:20.00) | 4.67 (Max:6.67)   |
| aime25      | samples_16 | orm_1.5b_raw_only_ckpt4054          |       5 | 10.58 (Max:10.83) | 19.33 (Max:20.00) | 12.67 (Max:16.67) | 9.33 (Max:10.00)  | 12.00 (Max:13.33) | 9.33 (Max:10.00)  |
| aime25      | samples_16 | prm_1.5b_qwen_raw_only_ckpt4054     |       5 | 10.58 (Max:10.83) | 19.33 (Max:20.00) | 10.67 (Max:13.33) | 9.33 (Max:10.00)  | 14.00 (Max:16.67) | 10.00 (Max:13.33) |
| aime25      | samples_16 | prm_1.5b_deepseek_raw_only_ckpt4054 |       5 | 10.58 (Max:10.83) | 19.33 (Max:20.00) | 10.67 (Max:13.33) | 6.00 (Max:10.00)  | 6.00 (Max:10.00)  | 12.00 (Max:16.67) |
| aime25      | samples_16 | prm_1.5b_ensemble_raw_only_ckpt4054 |       5 | 10.58 (Max:10.83) | 19.33 (Max:20.00) | 8.00 (Max:10.00)  | 8.00 (Max:10.00)  | 14.67 (Max:16.67) | 8.67 (Max:13.33)  |
| aime25      | samples_16 | prm_1.5b_llama_raw_only_ckpt4054    |       5 | 10.58 (Max:10.83) | 19.33 (Max:20.00) | 8.00 (Max:10.00)  | 10.67 (Max:13.33) | 12.00 (Max:13.33) | 10.00 (Max:13.33) |
| math500     | samples_16 | prm_1.5b_llama_raw_only_ckpt4054    |       5 | 58.45 (Max:58.83) | 62.12 (Max:62.80) | 60.28 (Max:60.60) | 60.72 (Max:61.00) | 60.00 (Max:60.40) | 60.28 (Max:60.60) |
| math500     | samples_16 | orm_1.5b_raw_only_ckpt4054          |       5 | 58.45 (Max:58.83) | 62.12 (Max:62.80) | 60.04 (Max:60.40) | 61.20 (Max:61.60) | 59.76 (Max:60.20) | 60.40 (Max:61.00) |
| math500     | samples_16 | prm_1.5b_qwen_raw_only_ckpt4054     |       5 | 58.45 (Max:58.83) | 62.12 (Max:62.80) | 59.76 (Max:60.20) | 60.32 (Max:60.80) | 60.36 (Max:61.80) | 60.20 (Max:61.60) |
| math500     | samples_16 | prm_1.5b_ensemble_raw_only_ckpt4054 |       5 | 58.45 (Max:58.83) | 62.12 (Max:62.80) | 59.16 (Max:59.80) | 60.60 (Max:61.00) | 60.52 (Max:61.20) | 60.08 (Max:61.00) |
| math500     | samples_16 | prm_1.5b_deepseek_raw_only_ckpt4054 |       5 | 58.45 (Max:58.83) | 62.12 (Max:62.80) | 59.12 (Max:60.40) | 60.44 (Max:61.20) | 59.64 (Max:60.40) | 59.64 (Max:60.20) |