from glob import glob
import os
import random

import numpy as np
from sklearn.impute import SimpleImputer
import torch
from skimage.io import imread

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import albumentations as A

import cv2
import pandas as pd
import matplotlib.pyplot as plt

from utils import crop_sample, pad_sample, resize_sample, normalize_volume
from sklearn.model_selection import train_test_split


class BrainSegmentationDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 3
    out_channels = 1

    def __init__(
        self,
        images_dir,
        transform=None,
        image_size=256,
        subset="train",
        random_sampling=True,
        validation_cases=10,
        seed=42,
    ):
        assert subset in ["all", "train", "validation"]

        # read images
        volumes = {}
        masks = {}
        print("reading {} images...".format(subset))
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            image_slices = []
            mask_slices = []
            for filename in sorted(
                filter(lambda f: ".tif" in f, filenames),
                key=lambda x: int(x.split(".")[-2].split("_")[4]),
            ):
                filepath = os.path.join(dirpath, filename)
                if "mask" in filename:
                    mask_slices.append(imread(filepath, as_gray=True))
                else:
                    image_slices.append(imread(filepath))
            if len(image_slices) > 0:
                patient_id = dirpath.split("/")[-1]
                volumes[patient_id] = np.array(image_slices[1:-1])
                masks[patient_id] = np.array(mask_slices[1:-1])

        self.patients = sorted(volumes)

        # select cases to subset
        if not subset == "all":
            random.seed(seed)
            validation_patients = random.sample(self.patients, k=validation_cases)
            if subset == "validation":
                self.patients = validation_patients
            else:
                self.patients = sorted(
                    list(set(self.patients).difference(validation_patients))
                )

        print("preprocessing {} volumes...".format(subset))
        # create list of tuples (volume, mask)
        self.volumes = [(volumes[k], masks[k]) for k in self.patients]

        print("cropping {} volumes...".format(subset))
        # crop to smallest enclosing volume
        self.volumes = [crop_sample(v) for v in self.volumes]

        print("padding {} volumes...".format(subset))
        # pad to square
        self.volumes = [pad_sample(v) for v in self.volumes]

        print("resizing {} volumes...".format(subset))
        # resize
        self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]

        print("normalizing {} volumes...".format(subset))
        # normalize channel-wise
        self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]

        # probabilities for sampling slices based on masks
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        self.slice_weights = [
            (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        ]

        # add channel dimension to masks
        self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

        print("done creating {} dataset".format(subset))

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        self.random_sampling = random_sampling

        self.transform = transform

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        if self.random_sampling:
            patient = np.random.randint(len(self.volumes))
            slice_n = np.random.choice(
                range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
            )

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor, mask_tensor

class MriDataset(Dataset):
    def __init__(self, df, transform=None, mean=0.5, std=0.25):
        super(MriDataset, self).__init__()
        self.df = df
        self.transform = transform
        self.mean = mean
        self.std = std
        
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx, raw=False):
        row = self.df.iloc[idx]
        img = cv2.imread(row['image_filename'], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(row['mask_filename'], cv2.IMREAD_GRAYSCALE)
        if raw:
            return img, mask
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        img = T.functional.to_tensor(img)
        mask = mask // 255
        mask = torch.Tensor(mask)
        return img, mask
    
def get_file_row(path):
    """Produces ID of a patient, image and mask filenames from a particular path"""
    path_no_ext, ext = os.path.splitext(path)
    filename = os.path.basename(path)
    
    patient_id = '_'.join(filename.split('_')[:3]) # Patient ID in the csv file consists of 3 first filename segments
    
    return [patient_id, path, f'{path_no_ext}_mask{ext}']

def get_train_test_loaders(args):
    files_dir = 'datasets/2/lgg-mri-segmentation/kaggle_3m/'
    file_paths = glob(f'{files_dir}/*/*[0-9].tif')
    
    csv_path = 'datasets/2/lgg-mri-segmentation/kaggle_3m/data.csv'
    df = pd.read_csv(csv_path)

    # df.info()
    imputer = SimpleImputer(strategy="most_frequent")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # merge original df and filenames based on patient
    filenames_df = pd.DataFrame((get_file_row(filename) for filename in file_paths), columns=['Patient', 'image_filename', 'mask_filename'])
    df = pd.merge(df, filenames_df, on="Patient")

    # ### Split data into train, valid, and test
    train_df, test_df = train_test_split(df, test_size=0.3)
    test_df, valid_df = train_test_split(test_df, test_size=0.5)


    transform = A.Compose([
        A.ChannelDropout(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.3),
    ])

    train_dataset = MriDataset(train_df, transform)
    valid_dataset = MriDataset(valid_df)
    test_dataset = MriDataset(test_df)


    # ### DataLoaders

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

def plot_examples(dataset, n_samples=4, save_path="dataset_examples.png"):
    fig, axs = plt.subplots(n_samples, 3, figsize=(20, n_samples*7), constrained_layout=True)
    i = 0
    for ax in axs:
        while True:
            image, mask = dataset.__getitem__(i, raw=True)
            i += 1
            if np.any(mask): 
                ax[0].set_title("MRI images")
                ax[0].imshow(image)
                ax[1].set_title("Highlited abnormality")
                ax[1].imshow(image)
                ax[1].imshow(mask, alpha=0.2)
                ax[2].imshow(mask)
                ax[2].set_title("Abnormality mask")
                break
    # Save the figure instead of showing it
    plt.savefig(save_path)
    plt.close(fig)
    
# ----- LABELED CT DATASETS -----
# -------------------------------

import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class LabeledCTDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Initialize the dataset with directories for images and masks.

        Args:
            images_dir (str): Path to the directory containing the CT scan images.
            masks_dir (str): Path to the directory containing the corresponding masks (.nii.gz format).
            transform (callable, optional): Optional transforms to be applied on the images and masks.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Sort files to ensure alignment between images and masks
        self.image_filenames = sorted(os.listdir(images_dir))
        self.mask_filenames = sorted(os.listdir(masks_dir))

        # Ensure the lengths of images and masks match
        assert len(self.image_filenames) == len(self.mask_filenames), \
            "Number of images and masks must match."

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding mask by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image and the corresponding mask.
        """
        # Construct the full file paths
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])

        # Read the image using SimpleITK
        # image = sitk.ReadImage(image_path)
        # image = sitk.GetArrayFromImage(image)  # Convert to numpy array
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.moveaxis(image, 0, -1)  # Move channel to last axis if necessary

        # Read the mask using SimpleITK
        mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask)  # Convert to numpy array
        mask = np.moveaxis(mask, 0, -1)  # Move channel to last axis if necessar
        mask = np.squeeze(mask).astype(np.uint8)  # Remove unnecessary dimensions if needed

        # Apply transformations if specified
        if self.transform:
            # Apply the same transform to both image and mask
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return {
            "image": image,
            "mask": mask
        }


# Example of transforms
from torchvision.transforms import functional as F

class TransformWrapper:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    
    def __call__(self, image, mask):
        image = self.image_transform(image)
        # print shape of image and mask

        mask = self.mask_transform(mask).squeeze()
        mask = torch.where(mask > 0.001, 1, 0)  # Threshold at a small value to binarize
        # change type of mask to float32
        mask = mask.float()
        return {"image": image, "mask": mask}

from torch.utils.data import DataLoader, random_split

def get_labeled_CT_datasets(batch_size=16, num_workers=64):
    # Define the dataset and transformations
    transform = TransformWrapper()
    dataset = LabeledCTDataset(images_dir="datasets/ct_scans/images", masks_dir="datasets/ct_scans/masks", transform=transform)

    # Calculate the sizes for train, validation, and test splits
    train_size = int(0.7 * len(dataset))  # 70% for training
    valid_size = int(0.15 * len(dataset))  # 15% for validation
    test_size = len(dataset) - train_size - valid_size  # Remaining 15% for testing

    # Split the dataset
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # Create DataLoaders for each split
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Example usage (optional)
    for batch in train_dataloader:
        images, masks = batch["image"], batch["mask"]
        print(f"Train batch: images {images.shape}, masks {masks.shape}")
        break

    for batch in valid_dataloader:
        images, masks = batch["image"], batch["mask"]
        print(f"Validation batch: images {images.shape}, masks {masks.shape}")
        break

    for batch in test_dataloader:
        images, masks = batch["image"], batch["mask"]
        print(f"Test batch: images {images.shape}, masks {masks.shape}")
        break

    return train_dataloader, valid_dataloader, test_dataloader

