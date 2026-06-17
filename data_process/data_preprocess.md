# Data Preprocessing

This document describes how to obtain 2D full-brain fMRI surface data in the `1023 x 2514` cortex image format used by NeuroPictor.

There are two common workflows:

1. Start from processed NSD `fsaverage` data.
2. Start from raw fMRI data such as `.nii.gz` files.

## Option 1: Processed NSD `fsaverage` Data

If you already have processed NSD `fsaverage` data, such as NSD beta files in `.mgh` format, you can use the scripts provided in the following repository:

- NSD beta example:
  `https://natural-scenes-dataset.s3.amazonaws.com/index.html#nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/`
- Conversion scripts:
  `https://github.com/cnwangy/cortex_image`

## Option 2: Raw fMRI Data to 2D Surface Images

This workflow converts raw fMRI data into `dtseries` files first, and then transforms the `dtseries` files into 2D cortical surface images.

### Requirements

Install the following tools before running the pipeline:

- fMRIPrep: `https://fmriprep.org/en/stable/installation.html`
- Connectome Workbench: `https://www.humanconnectome.org/software/get-connectome-workbench#download`

### Step 1: Organize the Input Data in BIDS Format

Prepare a valid BIDS directory, for example following datasets such as BOLD5000:

`https://openneuro.org/datasets/ds001499/versions/1.3.0`

Your directory should look like this:

```text
BIDS/
  sub*/
    anat/
    func/
      *.nii.gz
```

### Step 2: Run fMRIPrep to Generate `dtseries` Data

Run fMRIPrep on the raw data to generate surface-aligned outputs. Depending on dataset size, this step may take many hours or even 1-2 days.

```bash
~/.local/bin/fmriprep-docker path/to/BIDS path/to/BIDS2 participant --skip_bids_validation \
  --participant-label 01 -w path/to/fmriprep_BIDS2/tmp --nthreads 32 --omp-nthreads 32 \
  --output-spaces MNI152NLin6Asym:res-2 fsaverage5 --cifti-output --use-aroma --ignore slicetiming sbref t2w \
  --fs-subjects-dir path/to/Test/freesurfer --fs-license-file path/to/license.txt
```

After preprocessing, you should obtain files similar to:

```text
sub-*_task-*_run-*_space-fsLR_den-91k_bold.dtseries.nii
```

### Step 3: Convert `dtseries` Files to 2D Surface Images

Place the `dtseries` files you want to process into a directory such as:

```text
path/to/dtseries/sub01
```

Then run:

```bash
python nii_to_surf_dnv.py \
  --surface_images_dir path/to/dtseries/sub01 \
  --data_dir path/to/DNV/sub01 \
  --wb_command path/to/workbench/bin_linux64/wb_command
```

If you want to disable image normalization, add:

```bash
--normalize_images false
```

### Output

The script will automatically:

1. Scan all `.nii` files in `path/to/dtseries/sub01`.
2. Convert each `dtseries.nii` file into left and right hemisphere GIFTI files.
3. Project the surface data into the 2D cortical image representation.
4. Save the final outputs as `.h5` files under `path/to/DNV/sub01/hdf5`.

Intermediate GIFTI files will be saved under:

```text
path/to/DNV/sub01/gii
```

The final `.h5` files contain the 2D surface representation of the fMRI data in the NeuroPictor image format.
