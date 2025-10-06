# Food Classification & Nutrient Estimation Framework

This project aims to classify food items in images and estimating nutritional information.  
The workflow integrates image segmentation, classical computer vision, and LLM-based nutrient estimation.  

**Segmentation Types:**  
This framework uses **instance segmentation**, which identifies and separates individual objects (food items) in an image, unlike semantic segmentation which only labels pixels by class.

General Workflow:

- **Obtain datasets** (from FoodInsSeg) and preprocess them  
- **Annotate & clean data** using Roboflow  
- **Train YOLOv11 models** with Ray Tune and WandB logging  
- **Estimate food areas** using a reference coin detected separately using OpenCV2  
- **Connect to external nutritional data** via USDA API (legacy)  
- **Estimate calories and nutrients** using a Llama-based LLM  

By following this workflow, you can quickly analyze images of meals and compute approximate nutritional values.

---

## Workflow Info

---

### `dataset`
- **Source**: Downloaded dataset from **FoodInsSeg**.  
- **Cleaning & annotations**: Used **Roboflow** to correct masks, add missing annotations, and standardize data.  
- **Split**: Train/validation/test splits; defaults to `80/10/10`. In the latest setup, the project used `80/20` with no separate validation set and instead ran cross-validation.

---

### `training`
- **Model**: YOLOv11 for instance segmentation. 
- **Preprocessing**: All images resized to 640 x 640 and various data augmentations applied. 
- **Tuning**: Used **Ray Tune** for hyperparameter search using various search algs (e.g. BayesOpt, RandomSearch).
- **Logging**: Used **WandB** for training metrics and visualization.    
- **Freeze layers**: In latest model, the backbone is frozen (i.e. up to layer 11). This is subject to change.

---

### `area_estimation`
- **Reference coin**: Used a known coin (e.g., nickel) for scale calibration. Used **OpenCV2** to detect the coin and estimate area. 

---

### `nutrition_estimation`
- **USDA API** (Obsolete): Previously fetched nutrient information based on recognized food items. Currently **deprecated**, included for reference.  
- **LLM-based estimation** (Mandatory): Used **Llama** to estimate calories and nutrients from computed food areas and item labels.  

---

### `notes`
- The current model sucks due to imbalance data and messy annotation. Better model in the works.
