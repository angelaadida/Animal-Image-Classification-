# 🐾 Animal Image Classification

A PyTorch image classification project that recognizes **10 animal categories** using CNN and Transfer Learning. Three models are trained and compared.

---

## 📌 Categories

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Butterfly | 5 | Elephant |
| 1 | Cat | 6 | Horse |
| 2 | Chicken | 7 | Sheep |
| 3 | Cow | 8 | Spider |
| 4 | Dog | 9 | Squirrel |

---

## 📁 Project Structure

```
animal-image-classification/
├── datasets.py            # AnimalDataset class (CNN_datasets_20Nov.py)
├── models.py              # MyNeuralNetwork, SimpleCNN, AdvancedCNN (CNN_models_20Nov.py)
├── train.py               # Training script (CNN_train_Animal_MobilenetV2_25Nov.py)
├── inference_image.py     # Single image inference (CNN_Inference_image25Nov.py)
├── inference_video.py     # Video inference (CNN_Inference_video25Nov.py)
├── notebook.ipynb         # Full pipeline notebook
├── requirements.txt
├── README.md
├── project_summary.md
├── dataset_description.md
└── .gitignore
```

Data and checkpoints (not in repo):
```
animals/
  train/  butterfly/ cat/ chicken/ cow/ dog/ elephant/ horse/ sheep/ spider/ squirrel/
  test/   (same structure)
train_models/
  AdvancedCNN/   best.pt  last.pt
  ResNet18/      best.pt  last.pt
  MobileNetV2/   best.pt  last.pt
tensorboard/
```

---

## ⚙️ Setup

```bash
git clone <repo-url>
cd animal-image-classification
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## 📥 Dataset

Download **Animals-10** and place at `animals/`

- **Kaggle:** https://www.kaggle.com/datasets/alessiocorrado99/animals10
- **Google Drive (shared):** Datasets/animals/

| Split | Images |
|-------|-------:|
| Train | ~21,000 |
| Test  | ~5,000  |
| Total | ~26,000 |

---

## 🚀 Training

```bash
python train.py \
  --data-path animals \
  --image-size 224 \
  --epochs 100 \
  --batch-size 16 \
  --lr 0.01 \
  --checkpoint_dir train_models \
  --log_dir tensorboard \
  --early_stopping_duration 3
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --data-path | animals | Path to dataset |
| --image-size | 64 | Resize all images |
| --epochs | 2 | Max training epochs |
| --batch-size | 2 | Batch size |
| --lr | 0.01 | Learning rate |
| --checkpoint_dir | train_models | Checkpoint directory |
| --log_dir | tensorboard | TensorBoard directory |
| --early_stopping_duration | 3 | Patience epochs |

---

## 🧠 Models

### AdvancedCNN — trained from scratch
```
5 blocks: Conv2d x2 + BN + ReLU + MaxPool
3x224x224 → 64x7x7 (3136)
FC(3136→512, Drop0.5) → FC(512→128, Drop0.5) → FC(128→10)
```

### ResNet18 — Transfer Learning
```
Pretrained ResNet18 (ImageNet)
Replace fc: Linear(512 → 10)
```

### MobileNetV2 — Transfer Learning ✅ Best
```
Pretrained MobileNetV2 (ImageNet)
Replace classifier[1]: Linear(1280 → 10)
```

---

## ✨ Data Augmentation (train only)

```python
ColorJitter(brightness=0.5, contrast=(0.5,1.5), saturation=(0.5,1.5), hue=(-0.2,0.2))
RandomAffine(degrees=20, scale=(0.85,1.15), shear=(-30,30))
ToTensor()
Resize((224, 224))
```

---

## 🔍 Inference

### Single Image
```bash
python inference_image.py \
  --image-path path/to/image.jpg \
  --checkpoint-path train_models/MobileNetV2/best.pt
```

### Video
```bash
python inference_video.py \
  --video-path path/to/video.mp4 \
  --checkpoint-path train_models/MobileNetV2/best.pt
```
Output saved to `output.mp4`

---

## 📈 TensorBoard

```bash
tensorboard --logdir tensorboard
```
Open http://localhost:6006

---

## 📝 Results

| Model | Pretrained | Params | Best Val Accuracy | Best Epoch |
|-------|-----------|-------:|:-----------------:|:----------:|
| AdvancedCNN | No | ~3.2M | — | — |
| ResNet18 | Yes (ImageNet) | ~11.2M | — | — |
| MobileNetV2 | Yes (ImageNet) | ~3.4M | — | — |

*(Update after training)*
