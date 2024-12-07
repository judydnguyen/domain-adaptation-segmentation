import argparse
import time

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage.morphology import binary_dilation

import torch
from tqdm import tqdm
from dataset import get_labeled_CT_datasets, get_train_test_loaders
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import BCE_dice, EarlyStopping, dice_pytorch, iou_pytorch, plot_result, plot_score, set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def training_loop(epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler):
    history = {'train_loss': [], 'val_loss': [], 'val_IoU': [], 'val_dice': []}
    early_stopping = EarlyStopping(patience=7)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        running_loss = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # img, mask = data
            img, mask = data["image"], data["mask"]
            img, mask = img.to(device), mask.to(device)
            # input_tensor = input_tensor.to(device, dtype=torch.float)
            mask = mask.to(device, dtype=torch.float32)
            predictions = model(img)
            predictions = predictions.squeeze(1)
            # import IPython; IPython.embed()
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
                # img, mask = data
                img, mask = data["image"], data["mask"]
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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import binary_dilation
import torch
import os
import numpy as np

def show_or_save_masks(model, test_loader, device, parent_path='', save_path=None):
    width = 3
    columns = 10
    n_examples = columns * width

    fig, axs = plt.subplots(columns, width, figsize=(7 * width, 7 * columns), constrained_layout=True)
    fig.legend(
        loc='upper right',
        handles=[
            mpatches.Patch(color='red', label='Ground truth (if available)'),
            mpatches.Patch(color='green', label='Predicted abnormality'),
        ]
    )
    i = 0
    with torch.no_grad():
        for data in test_loader:
            image = data
            # mask = mask[0]
            mask = None
            image = image.to(device)
            prediction = model(image).to('cpu')[0][0]
            prediction = torch.where(prediction > 0.5, 1, 0).numpy()

            # Prepare the base layer: Original image
            base_image = image[0].to('cpu').permute(1, 2, 0).numpy()

            # Create an alpha mask based on the prediction
            alpha_mask = np.zeros_like(prediction, dtype=np.float32)
            alpha_mask[prediction > 0] = 0.4  # Full opacity for the prediction
            alpha_mask[prediction <= 0] = 0.0  # Full transparency for areas outside

            if mask is not None or mask.byte().any():  # If ground truth is available
                prediction_edges = prediction - binary_dilation(prediction)
                ground_truth = mask - binary_dilation(mask)

                # Overlay ground truth in red
                base_image[:, :, 0][ground_truth.bool()] = 1

                # Overlay prediction edges in green
                base_image[:, :, 1][prediction_edges.bool()] = 1

                axs[i // width][i % width].imshow(base_image)
            else:  # If no ground truth is available
                axs[i // width][i % width].imshow(base_image, alpha=0.8, cmap='gray')
                axs[i // width][i % width].imshow(prediction, cmap='Oranges', alpha=alpha_mask)
                axs[i // width][i % width].set_title("Predicted Mask")

            if n_examples == i + 1:
                break
            i += 1

    # Show or save the figure
    if save_path:
        os.makedirs(parent_path, exist_ok=True)
        plt.savefig(f"{parent_path}/{save_path}")
    else:
        plt.show()

    plt.close(fig)


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import binary_dilation
import torch

def save_test_samples_with_masks(model, test_loader, device, parent_path='', save_path="test_samples.png"):
    width = 3  # Number of rows per figure
    columns = 3  # Three columns: input image, ground truth mask, predicted mask
    n_examples = columns * width

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(width, columns, figsize=(7 * columns, 7 * width), constrained_layout=True)

    i = 0  # To track number of plotted samples
    with torch.no_grad():
        for data in test_loader:
            image, mask = data["image"], data["mask"]
            # image, mask = data
            mask = mask[0]
            image = image.to(device)
            prediction = model(image).to('cpu')[0][0]
            print(f"Sum of prediction: {prediction.sum()}")
            prediction = torch.where(prediction > 0.2, 1, 0)
            print(f"Prediction shape: {prediction.shape}")
            print(f"Sum of prediction: {prediction.sum()}")

            # Plot the input image
            axs[i][0].imshow(image[0].to('cpu').permute(1, 2, 0), cmap="gray")
            axs[i][0].set_title("Input Image")
            axs[i][0].axis("off")

            # Plot the ground truth mask
            axs[i][1].imshow(mask.squeeze(), cmap="gray")
            axs[i][1].set_title("Ground Truth Mask")
            axs[i][1].axis("off")

            # Plot the predicted mask
            axs[i][2].imshow(prediction.squeeze(), cmap="gray")
            axs[i][2].set_title("Predicted Mask")
            axs[i][2].axis("off")

            # Move to the next row of the grid
            i += 1
            if i == width:  # Stop after filling the grid
                break

    # Save the figure to the specified path
    plt.savefig(f"{parent_path}/{save_path}")
    plt.close(fig)


def test(model, test_loader, loss_fn):
    
    with torch.no_grad():
        running_IoU = 0
        running_dice = 0
        running_loss = 0
        for i, data in enumerate(test_loader):
            # img, mask = data
            img, mask = data["image"], data["mask"]
            img, mask = img.to(device), mask.to(device)
            predictions = model(img)
            predictions = predictions.squeeze(1)
            # import IPython; IPython.embed()
            print(f"Prediction range: min={predictions[0].min()}, max={predictions[0].max()}")

            running_dice += dice_pytorch(predictions, mask).sum().item()
            print(f"Sum of predictions: {predictions.sum()}")
            print(f"Sum of masks: {mask.sum()}")
            print(f"running_dice: {running_dice}")
            running_IoU += iou_pytorch(predictions, mask).sum().item()
            loss = loss_fn(predictions, mask)
            running_loss += loss.item() * img.size(0)
        loss = running_loss / len(test_loader.dataset)
        dice = running_dice / len(test_loader.dataset)
        IoU = running_IoU / len(test_loader.dataset)
        
        print(f'Tests: loss: {loss} | Mean IoU: {IoU} | Dice coefficient: {dice}')

def save_test_samples(model, test_loader, device, parent_path='', save_path="test_samples.png"):
    width = 3
    columns = 1
    n_examples = columns * width

    fig, axs = plt.subplots(columns, width, figsize=(7 * width, 7 * columns), constrained_layout=True)
    axs = axs.ravel()  # Flatten axs for consistent indexing
    
    fig.legend(
        loc='upper right',
        handles=[
            mpatches.Patch(color='green', label='Ground truth'),
            mpatches.Patch(color='red', label='Predicted abnormality'),
        ]
    )
    
    i = 0
    with torch.no_grad():
        for data in test_loader:
            image, mask = data
            image, mask = data["image"], data["mask"]
            mask = mask[0]
            # Skip samples without any ground truth mask
            if not mask.byte().any():
                continue
            
            image = image.to(device)
            prediction = model(image).to('cpu')[0][0]
            prediction = torch.where(prediction > 0.5, 1, 0)
            
            prediction_edges = prediction - binary_dilation(prediction)
            ground_truth = mask - binary_dilation(mask)
            
            image[0, 0, ground_truth.bool()] = 1  # Highlight ground truth in red
            image[0, 1, prediction_edges.bool()] = 1  # Highlight prediction in green
            
            axs[i].imshow(image[0].to('cpu').permute(1, 2, 0))
            axs[i].axis('off')  # Turn off axes for a cleaner plot
            
            i += 1
            if i == n_examples:
                break

    # Save the figure to the specified path
    plt.savefig(f"{parent_path}/{save_path}")
    plt.close(fig)


def main(args):
    
    # set seed
    set_seed(args.seed)
    
    # load train, test, valid loaders
    train_loader = None
    test_loader = None
    valid_loader = None
    
    # train_loader, valid_loader, test_loader = get_train_test_loaders(args)
    # train_loader, valid_loader, test_loader = get_train_test_loaders(args) # MRI datasets
    train_loader, valid_loader, test_loader = get_labeled_CT_datasets(batch_size=args.batch_size, num_workers=args.num_workers)
    
    print(f"---> loaded train loader: {len(train_loader)}")
    print(f"---> loaded valid loader: {len(valid_loader)}")
    print(f"---> loaded test loader: {len(test_loader)}")
    
    # load model
    model = smp.Unet(encoder_name="efficientnet-b7", 
                     encoder_weights="imagenet",
                     in_channels=3, 
                     classes=1,
                     activation='sigmoid')
    model.load_state_dict(torch.load("weights/lgg-mri-segmentation.pth"))
    model.to(device);
    print(f"---> loaded model:")
    # print(model)
    loss_fn = BCE_dice
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=2,factor=0.2)

    history = training_loop(epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler)
    
    # # save model
    torch.save(model.state_dict(), f"{args.save_model_dir}/{args.save_model}")
    
    # plot_result(history, args.save_model_dir, suffix='unet')
    # plot_score(history, args.save_model_dir, suffix='unet')
    # model.load_state_dict(torch.load(f"{args.save_model_dir}/{args.save_model}"))
    # model.load_state_dict(torch.load("weights/lgg-mri-segmentation.pth"))
    # model.load_state_dict(torch.load("weights/w_aug_model.pth"))
    
    
    test(model, test_loader, loss_fn)
    save_test_samples(model, test_loader, device, parent_path=args.save_model_dir, save_path="test_samples_CT_mapped.png")
    save_test_samples_with_masks(model, test_loader, device, parent_path=args.save_model_dir, save_path="test_samples_CT_separated.png")
    
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