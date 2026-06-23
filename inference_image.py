import os.path
import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, mobilenet_v2, \
    MobileNet_V2_Weights
import numpy as np
import warnings
import cv2

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument("--image-path", "-p", type=str, help="path to an image", required=True)
    parser.add_argument("--image-size", "-s", type=int, default=224, help="Common size of image")
    parser.add_argument("--checkpoint-path", "-c", type=str, help="Path to trained checkpoint",
                        default="trained_models/best.pt")
    args = parser.parse_args()
    return args


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                  "squirrel"]
    # model = AdvancedCNN(num_classes=len(train_dataset.categories))
    # model = resnet18(weights=ResNet18_Weights)
    # model.fc = nn.Linear(in_features=512, out_features=len(train_dataset.categories), bias=True)
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=len(categories), bias=True)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = image/255.
    image = np.transpose(image, (2, 0, 1))
    # image = np.expand_dims(image, 0)
    image = image[None,:,:,:]
    image = torch.from_numpy(image).float().to(device)
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
        probs = softmax(output[0])
    predicted_prob, predicted_idx = torch.max(probs, dim=0)
    cv2.imshow("{}: {:0.2f} %".format(categories[predicted_idx], predicted_prob*100), ori_image)
    cv2.waitKey(0)



if __name__ == '__main__':
    args = get_args()
    inference(args)