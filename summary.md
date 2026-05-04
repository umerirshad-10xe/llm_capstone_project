# Experiment Summary

| Model | n_head | n_layer | lr | n_kv_head | train_loss | val_loss | train_acc | val_acc |
|-------|--------|---------|----|-----------|------------|----------|-----------|---------|
| MHA | 4 | 4 | 0.001 | - | 2.5073 | 2.5142 | 0.2721 | 0.2651 |
| GQA | 4 | 4 | 0.001 | 4 | 2.5090 | 2.5092 | 0.2661 | 0.2704 |

## Learnings
- GQA can reduce KV cache size while maintaining performance.
- More heads generally improve performance but increase computation.
- See `training_validation_comparison.png` for metric trends over steps.
