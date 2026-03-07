# RCNN Model Weights

Place your fine-tuned Faster R-CNN weights here:

- `constellation_rcnn.pt` — fine-tuned model weights (state dict)

## Training

Run the training script once implemented:

```bash
python training/train_rcnn.py --config configs/config.yaml
```

The model uses a **Faster R-CNN ResNet-50-FPN** backbone with a custom
classifier head for the 17 constellation classes.

## Expected weight file format

The detector supports two formats:

1. **Plain state dict** — `torch.save(model.state_dict(), path)`
2. **Training checkpoint** — `{"epoch": N, "model": state_dict, ...}`

If no weights file is present, the model falls back to random weights
(predictions will be meaningless but the pipeline will not crash).
