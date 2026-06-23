import os.path
import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import warnings
import cv2

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument("--video-path", "-v", type=str, help="path to a video", required=True)
    parser.add_argument("--frame-size", "-s", type=int, default=224, help="Common size of image")
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
    cap = cv2.VideoCapture(args.video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_video = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (width, height))
    while cap.isOpened():
        flag, ori_frame = cap.read()
        if not flag:
            break
        frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (args.frame_size, args.frame_size))
        frame = frame/255.
        frame = np.transpose(frame, (2, 0, 1))
        # frame = np.expand_dims(frame, 0)
        frame = frame[None,:,:,:]
        frame = torch.from_numpy(frame).float().to(device)
        softmax = nn.Softmax()
        with torch.no_grad():
            output = model(frame)
            probs = softmax(output[0])
        predicted_prob, predicted_idx = torch.max(probs, dim=0)
        text = "{}: {:0.2f} %".format(categories[predicted_idx], predicted_prob*100)
        ori_frame = cv2.putText(ori_frame, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA) # xem them ve putText, toa do goc tren cung ben trai
        out_video.write(ori_frame)
    cap.release() # giai phong bo nho
    out_video.release()



if __name__ == '__main__':
    args = get_args()
    inference(args)



