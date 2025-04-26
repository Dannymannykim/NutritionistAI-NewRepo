anotations_coco - contains COCO annotation of train and test of FoodInsSeg 
		instance segmentation dataset (see: https://github.com/jamesjg/FoodInsSeg)
anotations_yolo - contains Yolo annotation of the COCO annotation; this is the format needed to use Yolo's model

datasets -> cv - different cross-validation folds of the dataset, containing both images and annotations
	    full - the full dataset
            partial - parts of the dataset; mostly for experimentation

runs/segment -> training/tuning/validation/prediction results

true_masks - pngs of segmentation masks over food images; done using visualize.py

classifier.py - legacy code for naive food image classification with webcam using resnet model



Three main types of segmentation: semantic, instance, and panoptic.
Briefly, semantic segmentation does not distinguish objects of the same class. In other words, each class in
an image has exactly one mask.
Instance segmentation creates mask for each instance of an object, regardless of the class.
Panoptic is the best of both worlds.

In this project, we use instance segmentation for the following reasons:
	- Yolo model is trained on instance segmentation.
	- more info to work with to estimate volume: bounding boxes + counts + masks
Cons:
	- training intensive
	- we don't exactly need counts of each food item since
	all we want is the area to predict the relative volume/amount.
	- annotated data is inconsistent; some images separate instances while others dont

semantic segmentation has no bounding boxes. So it may be difficult
to estimate cases where an object is overlapped by another object.



