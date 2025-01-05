import h5py
import numpy as np
import os
from tqdm import tqdm
import cv2
import argparse

# Define VC Mask
image_size=(256, 256)
# Load mask and define VC and foreground masks
mask = np.load("NSD_full_brain/vc_roi.npz")['images'][0] # [H, W]
H, W = mask.shape
vc_mask = mask == 1 # [H, W]
fg_mask = (mask == 1) | (mask == -1) # [H, W]

# Generate grid coordinates
x = np.linspace(0, W-1, W)
y = np.linspace(0, H-1, H)
xx, yy = np.meshgrid(x, y)
grid = np.stack([xx, yy], axis=0) # [2, H, W]

# Determine bounding box for VC region
gird_ = grid * vc_mask[np.newaxis]
x1 = min(int(gird_[0].max()) + 1, W)
y1 = min(int(gird_[1].max()) + 10, H)
gird_[gird_ == 0] = 1e6
x0 = max(int(gird_[0].min() - 1), 0)
y0 = max(int(gird_[1].min() - 10), 0)

# Define bounding box coordinates
vc_mask = vc_mask
fg_mask = fg_mask
coord = [x0, x1, y0, y1]

# Crop and resize the VC mask for later transformations
crop_msk = vc_mask[coord[2]:coord[3] + 1, coord[0]:coord[1] + 1]
cmask = cv2.resize(crop_msk * 1., (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)

def transform(image):
    """
    Crop and resize the input image based on the VC mask.
    Set values outside the VC mask to 0.
    """
    image = np.array(image)
    image = image[coord[2]:coord[3] + 1, coord[0]:coord[1] + 1]
    image = cv2.resize(image, (image_size[1], image_size[0]))
    image[cmask == 0] = 0 # Set values outside VC region to 0
    return image


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract and save VC region from full brain data.")
    parser.add_argument("--sub", type=int, default=1, help="Subject ID (default: 1)")
    parser.add_argument("--use_vc", action="store_true", help="If set, only process VC region")
    args = parser.parse_args()

    sub = args.sub
    use_vc = args.use_vc

    # Define directories
    hdf5_dir = 'NSD_full_brain/fmri'  # Directory containing HDF5 files
    out_dir = 'NSD_full_brain/fmri_npy'  # Output directory for .npy files
    os.makedirs(out_dir, exist_ok=True)

    # Set HDF5 file path and output directory for the subject
    hdf5_file_path = os.path.join(hdf5_dir, '0{}_norm.h5'.format(sub))
    save_path = os.path.join(out_dir, '0{}_norm'.format(sub))
    os.makedirs(save_path, exist_ok=True)

    # Process HDF5 file
    samples = []
    labels = []
    cont = 0
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        data_array = hdf5_file['images'] # Load full brain data
        labels_array = np.array(hdf5_file['labels']) # Load labels

        # Save labels as a .npy file
        np.save(os.path.join(out_dir, '0{}_label'.format(sub)), labels_array)

        for i in tqdm(range(data_array.shape[0]), desc="Saving Images"):
            # Extract full cortex data
            fc_data = data_array[i]
            surface = np.array(fc_data)

            # Apply VC mask transformation if specified
            if use_vc:
                surface = transform(surface)

            # Print for shape comparison
            # print(f"Sample {i}: fc_data shape: {fc_data.shape}, surface shape: {surface.shape}")
            
            # Save the processed surface as .npy
            surface_filename = f'surf_{i:06d}'
            np.save(os.path.join(save_path, surface_filename), surface)
