# 📦 Dataset — Animals-10

## Overview

| Field | Detail |
|-------|--------|
| Dataset name | Animals-10 |
| Task | Multi-class Image Classification |
| Number of classes | 10 |
| Total images | ~26,000 |
| Image format | JPG / PNG |
| Original size | Variable (resized to 224×224 during training) |
| Color mode | RGB (3 channels) |

---

## Class Distribution (approximate)

| ID | Class | Train | Test | Total |
|----|-------|------:|-----:|------:|
| 0 | butterfly | ~1,680 | ~420 | ~2,100 |
| 1 | cat | ~1,360 | ~340 | ~1,700 |
| 2 | chicken | ~2,480 | ~620 | ~3,100 |
| 3 | cow | ~1,520 | ~380 | ~1,900 |
| 4 | dog | ~3,200 | ~800 | ~4,000 |
| 5 | elephant | ~1,120 | ~280 | ~1,400 |
| 6 | horse | ~2,080 | ~520 | ~2,600 |
| 7 | sheep | ~1,440 | ~360 | ~1,800 |
| 8 | spider | ~3,760 | ~940 | ~4,700 |
| 9 | squirrel | ~1,520 | ~380 | ~1,900 |
| | **Total** | **~20,160** | **~5,040** | **~25,200** |

---

## Folder Structure

```
animals/
├── train/
│   ├── butterfly/
│   ├── cat/
│   ├── chicken/
│   ├── cow/
│   ├── dog/
│   ├── elephant/
│   ├── horse/
│   ├── sheep/
│   ├── spider/
│   └── squirrel/
└── test/
    └── (same structure as train/)
```

---

## Download

- **Kaggle:** https://www.kaggle.com/datasets/alessiocorrado99/animals10
- **Google Drive (shared):** Datasets/animals/

---

## Preprocessing Pipeline

### Training (with augmentation)
```python
train_transform = Compose([
    ColorJitter(brightness=0.5, contrast=(0.5,1.5), saturation=(0.5,1.5), hue=(-0.2,0.2)),
    RandomAffine(degrees=20, translate=(0,0), scale=(0.85,1.15), shear=(-30,30)),
    ToTensor(),
    Resize((224, 224))
])
```

### Validation (no augmentation)
```python
val_transform = Compose([
    ToTensor(),
    Resize((224, 224))
])
```

### In AnimalDataset.__getitem__:
```python
image = Image.open(self.images[idx]).convert("RGB")  # PIL, not cv2
if self.transform:
    image = self.transform(image)
label = self.labels[idx]
return image, label
```

---

## Notes

- Images loaded with **PIL** to ensure correct RGB channel order
- `drop_last=True` on train loader avoids small final batches
- Dataset has mild class imbalance (dog, spider have the most; elephant the fewest)
- `cv2.imread()` is called in `__init__` but result is not used — only the path is stored
