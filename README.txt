anotations_coco - contains COCO annotation of train and test of FoodInsSeg 
		instance segmentation dataset (see: https://github.com/jamesjg/FoodInsSeg)
anotations_yolo - contains Yolo annotation of the COCO annotation; this is the format needed to use Yolo's model

datasets -> cv - different cross-validation folds of the dataset, containing both images and annotations
	    full - the full dataset
            partial - parts of the dataset; mostly for experimentation
	    
runs/segment -> training/tuning/validation/prediction results

true_masks - pngs of segmentation masks over food images; done using visualize.py

classifier.py - legacy code for naive food image classification with webcam using resnet model





