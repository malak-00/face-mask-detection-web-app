"""
Robust face mask detector training script.

Improvements over the original Colab notebook:
- Stronger augmentation (RandomResizedCrop, RandomAffine, ColorJitter, blur)
  -> teaches the model to handle different framings, lighting, angles.
- Unfreezes the last 2 inverted-residual blocks of MobileNetV2
  -> learns features specific to "is there a mask on this face" instead of
     just "is this image from the mask folder".
- Cosine LR schedule, label smoothing, longer training.
- Saves best model on validation accuracy.

Run on GPU. Expects the Face Mask Dataset under data/ with Train/Validation/Test.
"""

import os, time, copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

DATA_ROOT = "data"        # original Kaggle Face Mask 12K dataset
OUT_PATH  = "app/mask_detector.pth"
EPOCHS    = 10
BATCH     = 32
LR        = 3e-4


def build_model(num_classes=2):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    # unfreeze last 2 blocks - those will learn mask-specific features
    for p in m.features[-2:].parameters():
        p.requires_grad = True
    # same classifier head shape as the original notebook so app.py keeps loading it
    m.classifier[1] = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return m


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", DEVICE)
    if DEVICE.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        print("vram:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

    eval_tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = ImageFolder(os.path.join(DATA_ROOT, "Train"), transform=train_tfm)
    val_ds   = ImageFolder(os.path.join(DATA_ROOT, "Validation"), transform=eval_tfm)
    test_ds  = ImageFolder(os.path.join(DATA_ROOT, "Test"), transform=eval_tfm)

    print("classes:", train_ds.classes)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(2).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,}")

    # class weights to balance the real dataset (it has 2x more WithoutMask)
    from collections import Counter
    counts = Counter([y for _, y in train_ds.samples])
    n_classes = len(train_ds.classes)
    weights = torch.tensor(
        [len(train_ds) / (n_classes * counts[i]) for i in range(n_classes)],
        dtype=torch.float32,
    ).to(DEVICE)
    print("class weights:", weights.tolist())

    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optim   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    def run_epoch(loader, train=True):
        model.train(train)
        total_loss, total_correct, total_n = 0.0, 0, 0
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            if train:
                optim.zero_grad()
            with torch.set_grad_enabled(train):
                out = model(imgs)
                loss = loss_fn(out, labels)
            if train:
                loss.backward()
                optim.step()
            total_loss   += loss.item() * imgs.size(0)
            total_correct += (out.argmax(1) == labels).sum().item()
            total_n      += imgs.size(0)
        return total_loss / total_n, total_correct / total_n

    print("\n--- training ---")
    best_val_acc = 0.0
    best_state = None
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        et = time.time()
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss,   val_acc   = run_epoch(val_loader,   train=False)
        sched.step()
        elapsed = time.time() - et
        print(f"epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}%  "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%  "
              f"lr={optim.param_groups[0]['lr']:.2e}  ({elapsed:.0f}s)", flush=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            print(f"           -> new best ({val_acc*100:.2f}%), saving", flush=True)

    elapsed_total = time.time() - t0
    print(f"\ntraining took {elapsed_total/60:.1f} minutes")
    print(f"best validation accuracy: {best_val_acc*100:.2f}%")

    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(test_loader, train=False)
    print(f"\nTEST accuracy: {test_acc*100:.2f}%")

    torch.save(best_state, OUT_PATH)
    print(f"saved best model -> {OUT_PATH}")


if __name__ == "__main__":
    main()
