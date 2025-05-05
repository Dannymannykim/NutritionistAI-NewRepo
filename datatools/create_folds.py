import yaml
import pandas as pd
from collections import Counter
import random
from sklearn.model_selection import KFold
from pathlib import Path
import datetime
import shutil
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold

def create_folds(
      ksplit, 
      yaml_file, 
      source, 
      dest, 
      render_class_dist=False, 
      only_dist=False, 
      stratified=False, 
      seed=0, 
      name=""
   ):

   """
   Splits a dataset into k-folds for cross-validation, organizing images and labels accordingly.

   Args:
      ksplit (int):
         Number of folds to create in the dataset split.
      yaml_file (str or Path):
         File path to the dataset YAML file. This YAML should contain metadata including dataset paths and class definitions.\n
         Example: "./datasets/full/coco_seg.yaml"
      source : Path
         Path to the original dataset directory containing subdirectories 'images' and 'labels'.
         Example: Path("./datasets/full")
      dest : Path
         Destination path where the k-fold directories will be created and saved.
         Each fold will be a subdirectory here.
         Example: Path("./datasets/cv")
      render_class_dist : bool, optional (default=False)
         If True, displays class distribution and validation-to-training ratio statistics for each fold.
      only_dist : bool, optional (default=False)
         If True, only display class distribution heatmap without creating folds.
      seed : int, optional (default=0)
         Random seed for reproducibility of the dataset splitting process.
      name : str, optional (default="")
         Name for fold directory "cv_{kfold}_{split}{name}".

   Returns
   -------
   None
      The function creates the folder structure and copies/moves files but does not return any value.
   """
   
   # retrieve classes (names) from source yaml file to write into new .yaml's later
   with open(yaml_file, encoding="utf8") as y:
      classes = yaml.safe_load(y)["names"] 
   cls_idx = sorted(classes.keys())
   
   # retrieve all the annotation (label .txt) files and creates df that indicates number of class instances for each label.
   # this is needed to ensure correctness when labels start at arbitrary numbers.
   labels = sorted(source.rglob("labels/**/*.txt"))
   index = [label.stem for label in labels]  # uses base filename as ID (no extension)
   labels_df = pd.DataFrame([], columns=cls_idx, index=index)
   
   if render_class_dist:
      for label in labels:
         lbl_counter = Counter()
         with open(label) as lf:
            lines = lf.readlines()
         for line in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(line.split(" ", 1)[0])] += 1
         
         labels_df.loc[label.stem] = lbl_counter

   labels_df = labels_df.fillna(0.0)

   # creates a df indicating train or val for each label in each fold.
   random.seed(seed)  # for reproducibility
   kf = KFold(n_splits=ksplit, shuffle=True, random_state=seed) 
   
   kfolds = list(kf.split(labels_df))

   if stratified:
      skf = StratifiedKFold(n_splits=ksplit, shuffle=True, random_state=20)
      kfolds = list(skf.split(labels_df, labels_df.idxmax(axis=1)))
      raise ImportError # think abt which one to use
      #multi_label_df = labels_df.copy()
      #multi_label_df[multi_label_df > 0] = 1 
   
      #X = multi_label_df.values  # features: binary presence of each class
      #y = X.copy()               # same as X since labels = features in multi-label case

      #mskf = MultilabelStratifiedKFold(n_splits=ksplit, shuffle=True, random_state=seed)

      #kfolds = list(mskf.split(X, y))

   folds = [f"split_{n}" for n in range(1, ksplit + 1)]
   folds_df = pd.DataFrame(index=index, columns=folds)
   
   for i, (train, val) in enumerate(kfolds, start=1):
      folds_df.loc[labels_df.iloc[train].index, f"split_{i}"] = "train"
      folds_df.loc[labels_df.iloc[val].index, f"split_{i}"] = "val"

   # gather img files -> loop through folds -> Create necessary directories and .yaml files
   supported_extensions = [".jpg", ".jpeg", ".png"]

   images = [] # store image file paths

   for ext in supported_extensions: # loop through supported extensions and gather image files
      images.extend(sorted((source / "images").rglob(f"*{ext}")))
   
   save_path = Path(dest / f"cv_{ksplit}_{seed}{name}")
   save_path.mkdir(parents=True, exist_ok=False)
   ds_yamls = []

   # creates df of the val to train ratio of each class for each label.
   if render_class_dist:
      fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)
      for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
         train_totals = labels_df.iloc[train_indices].sum()
         val_totals = labels_df.iloc[val_indices].sum()
         # To avoid division by zero, we add a small value (1E-7) to the denominator
         ratio = val_totals / (train_totals + 1e-7)
         fold_lbl_distrb.loc[f"split_{n}"] = ratio
      
      plt.figure(figsize=(10, 6))
      sns.heatmap(fold_lbl_distrb.astype(float), annot=True, fmt=".2f", cmap="coolwarm")
      plt.title("Validation-to-Training Class Ratio per Fold")
      plt.xlabel("Class ID")
      plt.ylabel("Fold")
      plt.tight_layout()
      plt.savefig(save_path / "class_distribution_heatmap.png", dpi=300)
      #plt.show()

   for split in folds_df.columns:
      # Create directories
      split_dir = save_path / split
      split_dir.mkdir(parents=True, exist_ok=True)
      (split_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
      (split_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
      (split_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
      (split_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

      # Create dataset YAML files
      dataset_yaml = split_dir / f"{split}_dataset.yaml"
      ds_yamls.append(dataset_yaml)

      with open(dataset_yaml, "w") as ds_y:
         yaml.safe_dump(
               {
                  "path": split_dir.as_posix(),
                  "train": "images/train",
                  "val": "images/val",
                  "names": classes,
               },
               ds_y,
         )

   # loop over images one by one and copy-paste to new directories accordingly.
   for image, label in tqdm(zip(images, labels), total=len(images), desc="Copying files"):
      for split, k_split in folds_df.loc[image.stem].items():
         # Destination directory
         img_to_path = save_path / split / "images" / k_split
         lbl_to_path = save_path / split / "labels" / k_split
         
         # Copy image and label files to new directory (SamefileError if file already exists)
         shutil.copy(image, img_to_path / image.name)
         shutil.copy(label, lbl_to_path / label.name)


if __name__ == "__main__":
   dataset_pth = Path("./datasets/full")
   dataset_config_yaml = str(next(dataset_pth.glob("*.yaml"), None))
   dest_pth = Path("./datasets/cv")
   
   create_folds(
      ksplit=5, 
      yaml_file=dataset_config_yaml, 
      source=dataset_pth, 
      dest=dest_pth,
      render_class_dist=True, 
      stratified=True, 
      name="-stratified-visual-test"
   )