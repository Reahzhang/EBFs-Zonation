# HEBF: A Vegetation–Environment Zoning Framework

This repository contains Google Earth Engine (GEE) and Python scripts used in the modelling of fine-scale vegetation–environment relationships and zoning of evergreen broad-leaved forests (EBFs) in complex terrain.

---

## 📁 Directory Structure

All scripts are currently stored in the main directory. They can be grouped into:

### 🔷 GEE Script
- **images_correction.js**  
  Used in Google Earth Engine (GEE) to preprocess Landsat-8 imagery by applying scaling factors and filtering by region and time.

### 🔶 Python Scripts for Vegetation–Environment Modelling

These scripts handle data preprocessing, feature selection, and model training using various machine learning and statistical approaches:

- **Clean_Data.py** – Cleans and prepares input datasets.
- **clip.py** – Spatial clipping of input raster layers to study area.
- **draw.py** – Visualization and plotting of modelling results.
- **feature.py** – Extraction or generation of features from spatial or tabular data.
- **GBT.py** – Gradient Boosting Tree model training.
- **GWR.py** – Geographically Weighted Regression modelling.
- **Logic.py** – Logical conditions or filters applied during preprocessing.
- **RF.py** – Random Forest model training and validation.
- **SVM.py** – Support Vector Machine classifier implementation.
- **VIF.py** – Variance Inflation Factor analysis to reduce multicollinearity.
- **maxmin.py** – Normalization or feature scaling using min-max methods.
- **person.py** – Pearson correlation computation.
- **pperson.py** – Partial Pearson correlation analysis.
- **pointbiserialr.py** – Point-biserial correlation analysis.

---

## 🔧 How to Use

1. Clone this repository
2. Install dependencies listed in `requirements.txt`
3. Run preprocessing and modelling scripts as needed:
```bash
python Clean_Data.py
python RF.py
```

---

## 📊 Requirements

Install Python libraries via:
```bash
pip install -r requirements.txt
```

Typical dependencies include:
- `numpy`, `pandas`
- `scikit-learn`, `matplotlib`
- `rasterio`, `geopandas`

---

## 📬 Contact

**Shiqi Zhang**  
Email: shiqi.zhang@email.edu

---

## 🔗 Citation

If you use this code in your research, please cite the associated paper:  
“From Vegetation Classification to Zonation: A Multi-Source Modelling Framework for Evergreen Broad-Leaved Forests in Complex Terrain”
