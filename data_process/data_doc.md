# Data Processing

This section includes data processing code, along with instructions for handling .h5 data, full-brain ROI masks, extracting the 256 × 256 VC region from the full-brain surface, and custom ROI extraction and swapping.

## Data Preparation

Download the full-brain data from [Hugging Face](https://huggingface.co/Fudan-fMRI/neuropictor/tree/main/NSD_full_brain). Place the folder `./NSD_full_brain` in the root directory `./neuropictor`.

## Converting h5 Data to Numpy

The file `NSD_full_brain/fmri/0{i}_norm.h5` contains the full-brain fMRI data for subject {i}. 

To extract the 256 × 256 visual cortex (VC) surface as in the paper, run: 
```bash
python convert_brain.py --sub {subid} --use_vc
```
Alternatively, to extract the full-brain fMRI surface, run: 
```bash
python convert_brain.py --sub {subid}
```

The converted files will be saved in `.npy` format in the `NSD_full_brain/fmri_npy` folder. Note that the full-brain fMRI surface `.npy` files for all 8 subjects will occupy about 700GB of space.

## Visualizing ROIs

Section 1 and 2 of `roi_visualize.ipynb` demonstrate how to handle full-brain ROI masks. The corresponding 360 ROI names are listed in `HCP_ROIs.csv`. You can refer to "Cortical Division" in Table 1 of [this paper](https://link.springer.com/article/10.1007/s00429-021-02421-6) for more details on these ROI names.

Sections 3, 4, and 5 of `roi_visualize.ipynb` show the process for extracting Visual Cortex (VC) ROIs, as well as how to crop and resize the VC ROI mask to generate a 256 × 256 surface image. This part will generate `vc_roi.npy` and `vc_masks.mpy`. You can also change the ROI names to extract other regions of interest and create your own surface masks.

## Exchanging ROIs

`exchange_roi.ipynb` provides an example of how to swap specific regions between two fMRI surfaces. You can modify the ROI names to customize which regions you want to exchange.