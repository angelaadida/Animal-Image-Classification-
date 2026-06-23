# 📋 Project Summary — Animal Image Classification

## Overview
Animal image classifier for **10 categories** using PyTorch. Three models are trained and compared: custom CNN from scratch vs Transfer Learning with pretrained backbones.

---

## Source Files

| File in repo | Original file |
|-------------|--------------|
| datasets.py | CNN_datasets_20Nov.py |
| models.py | CNN_models_20Nov.py |
| train.py | CNN_train_Animal_MobilenetV2_25Nov.py |
| inference_image.py | CNN_Inference_image25Nov.py |
| inference_video.py | CNN_Inference_video25Nov.py |

---

## Models

### 1. AdvancedCNN — trained from scratch
- 5 convolutional blocks: Conv2d×2 + BatchNorm + ReLU + MaxPool
- Feature flow: 3×224×224 → 64×7×7 → FC(512) → FC(128) → FC(10)
- Dropout(0.5) before each FC layer
- ~3.2M parameters

### 2. ResNet18 — Transfer Learning
- Pretrained ResNet18 on ImageNet
- Replace `fc`: Linear(512 → 10)
- ~11.2M parameters

### 3. MobileNetV2 — Transfer Learning ✅ Best
- Pretrained MobileNetV2 on ImageNet
- Replace `classifier[1]`: Linear(1280 → 10)
- ~3.4M parameters

---

## Training Configuration

| Component | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.01) |
| Loss | CrossEntropyLoss |
| Input size | 224 × 224 |
| Batch size | 16 |
| Train transforms | ColorJitter + RandomAffine + ToTensor + Resize |
| Val transforms | ToTensor + Resize |
| Early stopping | 3 epochs patience |
| Checkpoints | best.pt + last.pt per model |
| Logging | TensorBoard |

---

## Training Flow

```
Epoch N:
  TRAIN:
    ColorJitter + RandomAffine augmentation applied
    forward → CrossEntropyLoss → backward → Adam.step
    log Train/Loss → TensorBoard

  VALIDATE (no augmentation):
    forward (no_grad) → loss + predictions
    compute accuracy_score, confusion_matrix
    log Val/Loss, Val/Accuracy, Confusion Matrix → TensorBoard
    save last.pt
    if best accuracy → save best.pt

  if no improvement for 3 epochs → Early Stopping
```

---

## Inference Pipeline

### Image (inference_image.py)
```
cv2.imread → BGR to RGB → resize(224,224) → /255
→ transpose(2,0,1) → [None,:,:,:] → FloatTensor
→ model forward → Softmax → argmax
→ cv2.imshow result
```

### Video (inference_video.py)
```
VideoCapture → frame by frame → same as image
→ putText on frame → write to output.mp4
```

---

## Model Comparison

| Model | Pretrained | Params | Best Val Acc | Notes |
|-------|-----------|-------:|:------------:|-------|
| AdvancedCNN | No | ~3.2M | — | Slowest convergence |
| ResNet18 | Yes | ~11.2M | — | Good Transfer Learning baseline |
| MobileNetV2 | Yes | ~3.4M | — | Best overall |

*(Update after training)*

---

## Implementation Details

| Topic | Detail |
|-------|--------|
| Image loading | PIL (not cv2) in __getitem__ |
| Train augmentation | ColorJitter + RandomAffine |
| Checkpoint | model.state_dict + optimizer.state_dict + epoch metadata |
| Resume | Not implemented in this version (checkpoint_dir is cleared each run) |
| TensorBoard | Loss, Accuracy, Confusion Matrix per epoch |
