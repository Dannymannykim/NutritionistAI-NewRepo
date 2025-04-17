import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import time
from torchvision.models import resnet18, ResNet18_Weights

# Goal: 1) Webcam (to act as camera since iphone to pc is a hassle) image -> 2) individual item detector (NOT single classifier)
# Note: Food item detection involves object detection (identifying and localizing food items in images with bounding boxes or
# segmentation masks) rather than strict image classification (assigning a single label to an image) for a single meal.
# For (2), consider Yolo v8 model's pretrained option on ImageNet. Fine-tune with own images.
    # - ImageNet vs COCO dataset: ImageNet is tailored towards more general items and classificaiton. COCO is better for 
    #   object detection, though it may lack images of food ingredients specifically.
    #       - YOLO's pretrained COCO models that support segmentation: yolov8n-seg.pt, yolo11n-seg.pt
    #       - Might need to train on Yolov8 format instead of COCO json format (COCO format is not anywhere near universal and so you may find yourself needing to convert it to another format for a model)
    # - Bounding boxes are not accurate as they are rectangular, so segmentation may be a better option.
    # - Fine tune with images: Model takes in image and txt file indicating where the object is in the overall image.
# Other facts:
#   - Mosaiced dataset images are often used to train to introduce more variety and scenes.
#   - Look into transfer learning, which may explain why fine-tuning benefits from pretrained models even if they weren't trained on the focused data.
#   - look out for class imbalance during fine-tuning
#   - yolo11x-seg has 23 layers. freeze=11 for all backbone. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11-seg.yaml

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

with open("imagenet_classes.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Transform
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

cap = cv2.VideoCapture(0)

last_prediction_time = 0
prediction = "..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Only predict once per second
    if time.time() - last_prediction_time > 1:
        center_crop = frame[
            frame.shape[0]//2 - 112 : frame.shape[0]//2 + 112,
            frame.shape[1]//2 - 112 : frame.shape[1]//2 + 112
        ]
        img = Image.fromarray(cv2.cvtColor(center_crop, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = outputs.max(1)
            prediction = labels[pred.item()]
            last_prediction_time = time.time()

    # Draw the prediction
    cv2.putText(frame, prediction, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.rectangle(frame,
                  (frame.shape[1]//2 - 112, frame.shape[0]//2 - 112),
                  (frame.shape[1]//2 + 112, frame.shape[0]//2 + 112),
                  (0, 255, 0), 2)

    cv2.imshow("Live Classification", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):  # Press 's' to save
        cv2.imwrite("snapshot.jpg", frame)
        print("Snapshot taken!")
    elif key == ord('q'):  # Press 'q' to quit
        break


cap.release()
cv2.destroyAllWindows()