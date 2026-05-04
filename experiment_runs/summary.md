# Global Summary (2000-step run, 20 configs)

> Wall time: 3801.4s | Workers: 8

## Top-10 by validation accuracy

| rank | config | model | size(M) | val_acc | val_loss |
|------|--------|-------|---------|---------|----------|
| 1 | GQA_h8_kv4_l8_lr0.01 | GQA | 1.993 | 0.4781 | 1.7702 |
| 2 | GQA_h8_kv2_l8_lr0.01 | GQA | 1.928 | 0.4748 | 1.7730 |
| 3 | MHA_h8_l8_lr0.01 | MHA | 2.124 | 0.4716 | 1.7936 |
| 4 | MHA_h4_l4_lr0.01 | MHA | 1.075 | 0.4702 | 1.7914 |
| 5 | GQA_h8_kv4_l4_lr0.01 | GQA | 1.009 | 0.4677 | 1.7910 |
| 6 | GQA_h4_kv2_l4_lr0.01 | GQA | 1.009 | 0.4668 | 1.8043 |
| 7 | GQA_h16_kv8_l4_lr0.01 | GQA | 1.009 | 0.4651 | 1.8008 |
| 8 | MHA_h16_l4_lr0.01 | MHA | 1.075 | 0.4599 | 1.8130 |
| 9 | MHA_h8_l4_lr0.01 | MHA | 1.075 | 0.4551 | 1.8421 |
| 10 | MHA_h2_l4_lr0.01 | MHA | 1.075 | 0.4467 | 1.8655 |

## Generated plots
- `global_top10_size_vs_val_accuracy.png`
- `mha_vs_gqa_avg_accuracy.png`
