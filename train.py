import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import ToTensor, Compose, Resize, ColorJitter, RandomAffine
import warnings
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from tqdm.autonotebook import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Suppress warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument("--data-path", "-d", type=str, default="animals", help="path to data")
    parser.add_argument("--image-size", "-i", type=int, default=64, help="common size of image")  # Adjusted
    parser.add_argument("--epochs", "-e", type=int, default=2, help="Number of epochs to train")  # Adjusted
    parser.add_argument("--batch-size", "-b", type=int, default=2, help="batch size")  # Adjusted
    parser.add_argument("--lr", type=float, default=0.01, help="Optimizer learning rate")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="Momentum")
    parser.add_argument("--checkpoint_dir", "-c", type=str, default="train_models", help="Checkpoint save location")
    parser.add_argument("--early_stopping_duration", "-s", type=int, default=3, help="Stop after no improvement")
    parser.add_argument("--log_dir", "-g", type=str, default="tensorboard", help="Logging info save location")
    args = parser.parse_args()
    return args

# Import custom modules
from CNN_models_20Nov import AdvancedCNN
from CNN_datasets_20Nov import AnimalDataset

def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(8, 8))  # Adjusted size
    plt.imshow(cm, interpolation='nearest', cmap="Blues")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def train(args):
    train_transform = Compose([
        ColorJitter(brightness=0.5, contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.2, 0.2)),
        RandomAffine(degrees=20, translate=(0, 0), scale=(0.85, 1.15), shear=(-30, 30)),
        ToTensor(),
        Resize((args.image_size, args.image_size))  # Adjusted size
    ])
    val_transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size))  # Adjusted size
    ])
    # cho nay se phai thay doi
    train_dataset = AnimalDataset(root=args.data_path, is_train=True, transform=train_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # Adjusted for lightweight system
        drop_last=True
    )

    val_dataset = AnimalDataset(root=args.data_path, is_train=False, transform=val_transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,  # Adjusted for lightweight system
        drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=len(train_dataset.categories), bias=True)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if os.path.isdir(args.checkpoint_dir):
        shutil.rmtree(args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    best_accuracy = -1
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        train_loss = []
        for iter, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            predictions = model(images)
            loss_value = criterion(predictions, labels)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch {epoch + 1}/{args.epochs}. Loss {loss_value.item():0.4f}")
            train_loss.append(loss_value.item())
            writer.add_scalar("Train/Loss", np.mean(train_loss), global_step=epoch * len(train_dataloader) + iter)

        model.eval()
        all_labels, all_predictions, all_loss = [], [], []

        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, colour="green")
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss_value = criterion(outputs, labels)

                predictions = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
                all_loss.append(loss_value.item())

            accuracy = accuracy_score(all_labels, all_predictions)
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            plot_confusion_matrix(writer, conf_matrix, train_dataset.categories, epoch)

            avg_loss = np.mean(all_loss)
            print(f"Epoch {epoch + 1}/{args.epochs}. Loss {avg_loss:0.4f}. Accuracy {accuracy:0.4f}")

            writer.add_scalar("Val/Loss", avg_loss, global_step=epoch)
            writer.add_scalar("Val/Accuracy", accuracy, global_step=epoch)

            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_epoch": best_epoch,
                "best_accuracy": best_accuracy,
            }

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best.pt"))

            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last.pt"))

            if epoch - best_epoch > args.early_stopping_duration:
                print(f"Stop training process at epoch {epoch + 1}")
                break

if __name__ == '__main__':
    args = get_args()
    train(args)
