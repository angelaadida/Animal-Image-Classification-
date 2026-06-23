import os.path
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision.transforms import ToTensor, Compose, Resize

class AnimalDataset(Dataset):

    def __init__(self, root, is_train, transform = None):
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
        if is_train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")

        self.transform = transform

        self.images = []
        self.labels = []


        for idx, category in enumerate(self.categories):
            category_path = os.path.join(data_path, category)

            for item in os.listdir(category_path):
                image_path = os.path.join(category_path, item)

                image = cv2.imread(image_path)
                self.images.append(image_path)

                self.labels.append(idx)

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):
        # image = cv2.imread(self.images[idx])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]

        return image, label

if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
      ])
    dataset = AnimalDataset(root="animals", is_train=False, transform=transform)
    # image, label = dataset[100]
    # print(image.shape)

    dataloader = DataLoader(
                            dataset=dataset,
                            batch_size=16,
                            shuffle=True,
                            num_workers=8,
                            drop_last=True
                            )

    for images, labels in dataloader:
        print(images.shape, labels.shape)
        print(labels)

# PIL thi ko dataloader duoc
# cv2 thi dung dataloader duoc, nhung co loi la anh ko cung kich thuot.
# animal moi anh co kich thuot khac nhau, ko stack vao voi nhau duoc. phai dua ve cung kich thuot
# collate: dung de gan cac anh lai voi nhau.

# ToTensor convert PIL or numpy array deu convert sang Tensor
# Sau do resize ve cung 1 kich thuot lu cdo mac dinh no moi ghep cac anh lai voi nhau.
# Bo cifar chung ta ko can transform vi anh da cung kich thuoc, va numpy array thi dataloader no doc duoc roi
# Bo animal anh la anh PIL, kich thuoc ko giong nhau nen phai dung 2 buoc ToTensor va Resize
