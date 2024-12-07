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

from albumentations.pytorch import ToTensorV2

import cv2
import pandas as pd
import matplotlib.pyplot as plt

from utils import crop_sample, pad_sample, resize_sample, normalize_volume, set_seed
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
    def __init__(self, pair_transforms=False):
        mean, std = [0.23276818, 0.23326994, 0.23333506], [0.22573704, 0.22619095, 0.22634698]
        self.pair_transforms = pair_transforms
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            # change contrast and brightness
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
            # transforms.Normalize(mean=mean, std=std),
            
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        self.transforms = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            ToTensorV2()
        ])
    
    def __call__(self, image, mask):
        if self.pair_transforms:
            augmented = self.transforms(image=image, mask=mask)
            # mask = torch.tensor(image["mask"])
            image, mask = augmented['image'], augmented['mask']
            image = image.float()
            mask = torch.where(mask > 0.001, 1, 0)
            mask = mask.float()
            return {"image": image, "mask": mask}
        image = self.image_transform(image)
        # print shape of image and mask

        mask = self.mask_transform(mask).squeeze()
        mask = torch.where(mask > 0.002, 1, 0)  # Threshold at a small value to binarize
        # change type of mask to float32
        mask = mask.float()
        return {"image": image, "mask": mask}

from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset, random_split, ConcatDataset

def get_labeled_CT_datasets(batch_size=16, num_workers=64):
    set_seed(42)
    # Define the dataset and transformations
    transform = TransformWrapper(pair_transforms=False)
    dataset = LabeledCTDataset(images_dir="datasets/ct_scans/images", masks_dir="datasets/ct_scans/masks", transform=transform)

    # Calculate the sizes for train, validation, and test splits
    train_size = int(0.7 * len(dataset))  # 70% for training
    valid_size = int(0.15 * len(dataset))  # 15% for validation
    test_size = len(dataset) - train_size - valid_size  # Remaining 15% for testing
    
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size).tolist()  # Shuffle indices
    # Split indices
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    # Print sizes
    print(f"Train size: {len(train_indices)}, Validation size: {len(valid_indices)}, Test size: {len(test_indices)}")

    # Create subsets for original dataset
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)

    # Load the transformed datasets
    train_dataset_transformed = LabeledCTDataset(
        images_dir="datasets/ct_scans/transformed/images",
        masks_dir="datasets/ct_scans/transformed/masks",
        transform=transform
    )
    train_dataset_transformed = Subset(train_dataset_transformed, range(5))
    # Combine the original and transformed train datasets
    combined_train_dataset = ConcatDataset([train_dataset, train_dataset_transformed])

    # Create a DataLoader for the combined dataset
    train_dataloader = DataLoader(
        combined_train_dataset, 
        # train_dataset,
        batch_size=batch_size, 
        shuffle=True,  # Keep shuffle=True for training
        num_workers=num_workers
    )
    
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # No need to shuffle for validation
        num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # No need to shuffle for testing
        num_workers=num_workers
    )
    
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

# define a function to load datasets
import torchio as tio
from pathlib import Path

def get_datasets(training_transform, validation_transform):
    # Define dataset paths
    dataset_dir = Path("datasets/ct_scans")
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "masks"

    # Collect file paths
    image_paths = sorted(images_dir.glob("*"))  # Adjust extension if necessary
    label_paths = sorted(labels_dir.glob("*.nii.gz"))
    print(f"Number of images: {len(image_paths)}")
    print(f"Number of labels: {len(label_paths)}")
    assert len(image_paths) == len(label_paths), "Mismatch between image and label counts!"

    # Create subjects
    subjects = []
    for image_path, label_path in zip(image_paths, label_paths):
        
        # ct=tio.ScalarImage(image_path)
        # print shape of ct
        # print(f"shape of ct: {ct.shape}")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(f"shape of image: {image.shape}")
        # convert image to torch tensor
        image_tensor = torch.tensor(image).unsqueeze(0).transpose(0, 3).transpose(1, 2).float()
        image_tensor /= 255.0
        print(f"shape of image tensor: {image_tensor.shape}")
        ct = tio.ScalarImage(tensor = image_tensor)
        brain=tio.LabelMap(label_path)
        # print(f"shape of brain: {brain.shape}")
        # # print range of brain
        # print(f"range of brain: {brain.tensor.min()}, {brain.tensor.max()}")
        # # print unique values of brain
        # print(f"unique values of brain: {np.unique(brain.tensor)}")
        # # print unique values of ct
        # print(f"unique values of ct: {np.unique(ct.tensor)}")
        # print shape of ct
        if ct.shape[0]!=3:
            ct= tio.ScalarImage(tensor = ct.tensor.repeat(3,1,1,1))
            print(f"shape of ct: {ct.shape}")
        else:
            ct= tio.ScalarImage(tensor = ct.tensor)
            print('untouched')
        brain= tio.LabelMap(tensor = brain.tensor)
        subject = tio.Subject(
            image=ct,
            mask=brain,
        )
        subjects.append(subject)

    # Create dataset
    dataset = tio.SubjectsDataset(subjects)
    print('Dataset size:', len(dataset), 'subjects')
    
    # Create dataset
    training_split_ratio = 0.7
    dataset = tio.SubjectsDataset(subjects)
    num_subjects = len(dataset)
    num_training_subjects = int(training_split_ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_validation_subjects
    training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects, generator=torch.Generator().manual_seed(42))
    validation_subjects, test_subjects = torch.utils.data.random_split(validation_subjects, [num_validation_subjects // 2, num_validation_subjects // 2], generator=torch.Generator().manual_seed(42))

    training_set = tio.SubjectsDataset(
        training_subjects, transform=training_transform)

    validation_set = tio.SubjectsDataset(
        validation_subjects, transform=validation_transform)
    
    test_set = tio.SubjectsDataset(test_subjects, 
                                   transform=validation_transform)

    print('Training set:', len(training_set), 'subjects')
    print('Validation set:', len(validation_set), 'subjects')
    print('Test set:', len(test_set), 'subjects')
    return training_set, validation_set, test_set

def get_transformed_loader():
    set_seed(11)
    training_transform = tio.Compose([
        tio.ToCanonical(),
        #tio.Resample(1),
        #tio.CropOrPad((48, 60, 48)),
        tio.Resize((256,256,1)),
        # tio.RandomMotion(p=0.2),
        # tio.RandomBiasField(p=0.3),
        # tio.RandomNoise(p=0.5),
        # tio.RandomFlip(),
        # tio.OneOf({
        #     tio.RandomAffine(): 0.6,
        # #  tio.RandomElasticDeformation(): 0.2,
        # }),
        # to tensor
        # tio.ToTensor(),
        # tio.OneHot(),
    ])

    validation_transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resize((256,256,1)),
        # tio.ToTensor(),
        #tio.Resample(ct),
        #tio.CropOrPad((48, 60, 48)),
        # tio.OneHot(),
    ])
    
    training_set, validation_set, test_set = get_datasets(training_transform, validation_transform)
    
    # turn into dataloaders
    train_dataloader = DataLoader(training_set, batch_size=1, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
    
    return train_dataloader, valid_dataloader, test_dataloader

