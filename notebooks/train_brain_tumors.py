import argparse
import time

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage.morphology import binary_dilation

import torch
from tqdm import tqdm
from dataset import get_train_test_loaders
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import BCE_dice, EarlyStopping, dice_pytorch, iou_pytorch, plot_result, plot_score, save_test_samples, set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def training_loop(epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler):
    history = {'train_loss': [], 'val_loss': [], 'val_IoU': [], 'val_dice': []}
    early_stopping = EarlyStopping(patience=7)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        running_loss = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            img, mask = data
            img, mask = img.to(device), mask.to(device)
            predictions = model(img)
            predictions = predictions.squeeze(1)
            loss = loss_fn(predictions, mask)
            running_loss += loss.item() * img.size(0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        model.eval()
        with torch.no_grad():
            running_IoU = 0
            running_dice = 0
            running_valid_loss = 0
            for i, data in enumerate(valid_loader):
                img, mask = data
                img, mask = img.to(device), mask.to(device)
                predictions = model(img)
                predictions = predictions.squeeze(1)
                running_dice += dice_pytorch(predictions, mask).sum().item()
                running_IoU += iou_pytorch(predictions, mask).sum().item()
                loss = loss_fn(predictions, mask)
                running_valid_loss += loss.item() * img.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        val_loss = running_valid_loss / len(valid_loader.dataset)
        val_dice = running_dice / len(valid_loader.dataset)
        val_IoU = running_IoU / len(valid_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_IoU'].append(val_IoU)
        history['val_dice'].append(val_dice)
        print(f'Epoch: {epoch}/{epochs} | Training loss: {train_loss} | Validation loss: {val_loss} | Validation Mean IoU: {val_IoU} '
         f'| Validation Dice coefficient: {val_dice}')
        
        lr_scheduler.step(val_loss)
        if early_stopping(val_loss, model):
            early_stopping.load_weights(model)
            break
    model.eval()
    return history

def test(model, test_loader, loss_fn):
    with torch.no_grad():
        running_IoU = 0
        running_dice = 0
        running_loss = 0
        for i, data in enumerate(test_loader):
            img, mask = data
            img, mask = img.to(device), mask.to(device)
            predictions = model(img)
            predictions = predictions.squeeze(1)
            running_dice += dice_pytorch(predictions, mask).sum().item()
            running_IoU += iou_pytorch(predictions, mask).sum().item()
            loss = loss_fn(predictions, mask)
            running_loss += loss.item() * img.size(0)
        loss = running_loss / len(test_loader.dataset)
        dice = running_dice / len(test_loader.dataset)
        IoU = running_IoU / len(test_loader.dataset)
        
        print(f'Tests: loss: {loss} | Mean IoU: {IoU} | Dice coefficient: {dice}')

def main(args):
    
    # set seed
    set_seed(args.seed)
    
    # load train, test, valid loaders
    train_loader = None
    test_loader = None
    valid_loader = None
    
    train_loader, valid_loader, test_loader = get_train_test_loaders(args)
    
    print(f"---> loaded train loader: {len(train_loader)}")
    print(f"---> loaded valid loader: {len(valid_loader)}")
    print(f"---> loaded test loader: {len(test_loader)}")
    
    # load model
    model = smp.Unet(encoder_name="efficientnet-b7", 
                     encoder_weights="imagenet",
                     in_channels=3, 
                     classes=1,
                     activation='sigmoid')
    model.to(device);
    print(f"---> loaded model:")
    # print(model)
    loss_fn = BCE_dice
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = 60
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=2,factor=0.2)

    history = training_loop(epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler)
    
    plot_result(history, args.save_model_dir, suffix='unet')
    plot_score(history, args.save_model_dir, suffix='unet')
    
    test(model, test_loader, loss_fn)
    save_test_samples(model, test_loader, device, parent_path=args.save_model_dir, save_path="test_samples.png")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--encoder", type=str, default="efficientnet-b7")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--classes", type=int, default=1)
    parser.add_argument("--activation", type=str, default="sigmoid")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--save_model", type=str, default="model.pth")
    parser.add_argument("--save_model_dir", type=str, default="weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=54)
    args = parser.parse_args()
    main(args)