import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-4


class HouseSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_files = sorted(self.images_dir.glob("*.png"))

        self.image_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

        self.mask_resize = transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.masks_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_resize(mask)
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 0).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask


def get_model():
    model = fcn_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
    return model


def dice_score(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return ((2.0 * intersection + smooth) / (union + smooth)).item()


def iou_score(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    total = preds.sum() + targets.sum() - intersection
    return ((intersection + smooth) / (total + smooth)).item()


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            probs = torch.sigmoid(outputs)

            total_loss += loss.item()
            total_dice += dice_score(probs, masks)
            total_iou += iou_score(probs, masks)

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n, total_iou / n


def save_loss_curve(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig("loss_curve.png")
    plt.close()


def save_predictions(model, loader):
    model.eval()
    os.makedirs("predictions", exist_ok=True)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            images = images.to(DEVICE)
            outputs = model(images)["out"]
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            image = images[0].cpu().permute(1, 2, 0).numpy()
            true_mask = masks[0].squeeze().cpu().numpy()
            pred_mask = preds[0].squeeze().cpu().numpy()

            plt.figure()
            plt.imshow(image)
            plt.axis("off")
            plt.title("Aerial Image")
            plt.savefig(f"predictions/image_{idx}.png", bbox_inches="tight")
            plt.close()

            plt.figure()
            plt.imshow(true_mask, cmap="gray")
            plt.axis("off")
            plt.title("Ground Truth Mask")
            plt.savefig(f"predictions/true_mask_{idx}.png", bbox_inches="tight")
            plt.close()

            plt.figure()
            plt.imshow(pred_mask, cmap="gray")
            plt.axis("off")
            plt.title("Predicted Mask")
            plt.savefig(f"predictions/pred_mask_{idx}.png", bbox_inches="tight")
            plt.close()


def main():
    train_dataset = HouseSegmentationDataset(
        "segmentation_dataset/images/train",
        "segmentation_dataset/masks/train"
    )
    val_dataset = HouseSegmentationDataset(
        "segmentation_dataset/images/val",
        "segmentation_dataset/masks/val"
    )
    test_dataset = HouseSegmentationDataset(
        "segmentation_dataset/images/test",
        "segmentation_dataset/masks/test"
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = get_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

    torch.save(model.state_dict(), "house_segmentation_model.pth")
    save_loss_curve(train_losses, val_losses)

    test_loss, test_dice, test_iou = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice Score: {test_dice:.4f}")
    print(f"Test IoU Score: {test_iou:.4f}")

    save_predictions(model, test_loader)


if __name__ == "__main__":
    main()