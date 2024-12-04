import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import gui

#from downloaddata import fetch_data as fdata

# Convert a SimpleITK image to a NumPy array
def sitk_to_numpy(image):
    return sitk.GetArrayViewFromImage(image)

# Function to plot two images side by side
def plot_two_images(image1, title1, image2, title2):
    # Convert the images to NumPy arrays
    image1_np = sitk_to_numpy(image1)
    image2_np = sitk_to_numpy(image2)

    # Plot side by side using subplots
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    # First image
    axes[0].imshow(image1_np, cmap="gray")
    axes[0].set_title(title1)
    axes[0].axis("off")  # Hide the axes

    # Second image
    axes[1].imshow(image2_np, cmap="gray")
    axes[1].set_title(title2)
    axes[1].axis("off")  # Hide the axes

    # Display the side-by-side plot
    plt.tight_layout()
    plt.show()


# Function to load data by paths and plot the images
def load_images(image_paths=[], dtype=sitk.sitkInt8):
    # Load the images
    images = []
    for image_path in image_paths:
        # check format of image whether it is .nrrd or .png
        if image_path.endswith(".nrrd"):
            image = sitk.ReadImage(image_path, imageIO="NrrdImageIO")   # fetch data from the path
        elif image_path.endswith(".jpg"):
            image = sitk.ReadImage(image_path, imageIO="JPEGImageIO")
        # image = sitk.ReadImage(image_path)
        # plt.imshow(sitk_to_numpy(image), cmap="gray")
        # plt.axis("off")
        images.append(image)
    return images

def disp_images(images, fig_size, wl_list=None):
    if images[0].GetDimension() == 2:
        gui.multi_image_display2D(
            image_list=images, figure_size=fig_size, window_level_list=wl_list
        )
    else:
        gui.MultiImageDisplay(
            image_list=images, figure_size=fig_size, window_level_list=wl_list
        )
    
# Example usage
if __name__ == "__main__":
    # Create the original grid image
    grid_image = sitk.GridSource(
        outputPixelType=sitk.sitkUInt16,
        size=(512, 512),
        sigma=(0.1, 0.1),
        gridSpacing=(20.0, 20.0),
    )

    # The spatial definition of the images we want to use in a deep learning framework (smaller than the original).
    new_size = [100, 100]
    reference_image = sitk.Image(new_size, grid_image.GetPixelIDValue())
    reference_image.SetOrigin(grid_image.GetOrigin())
    reference_image.SetDirection(grid_image.GetDirection())
    reference_image.SetSpacing(
        [
            sz * spc / nsz
            for nsz, sz, spc in zip(new_size, grid_image.GetSize(), grid_image.GetSpacing())
        ]
    )

    # Resample without any smoothing
    resampled_image = sitk.Resample(grid_image, reference_image)

    # Call the function to plot the original and resampled images
    plot_two_images(grid_image, "Original Grid Image", resampled_image, "Resampled without Smoothing")
