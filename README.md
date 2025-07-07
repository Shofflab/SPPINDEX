# SPPINDEX
This is a data respository for data collected in Dr. Shoffstall's lab related to the study of dexamethasone sodium phosphate loaded platelet-inspired nanoparticles (SPPINDEX) to reduce neuroinflammation associated with intracortical device implantation

## Code Description and Usage

This repository contains analysis code for processing histological images to quantify neuron density at various distances from implanted devices. The raw data and pre-trained models are provided as zipped files on Zenodo (https://zenodo.org/records/15603085).


### Analysis Workflow

#### Step 1: Neuron Segmentation
Run `notebooks/cellpose_prediction.ipynb` to perform automated neuron segmentation using our custom, pre-trained Cellpose model.
- Input: Raw histological images (PNG format)
- Output: Neuron masks saved as TIFF files in `output/` directory

#### Step 2: Create SECOND Masks
Use `src/app.py` GUI application to manually define regions of interest:
- Launch: `python src/app.py` or download the GUI executable from Zenodo. 
- Define the implant hole boundary (red outline)
- Add exclusion regions (yellow outlines) for artifacts or damaged tissue
- Save configuration as H5 files (required for distance analysis)

#### Step 3: Distance Analysis
Run `notebooks/dist_analysis.ipynb` to perform distance-based neuron density analysis:
- Input: Neuron masks from Step 1 and SECOND masks from Step 2
- Output: CSV files with density, count, and area data at binned distances