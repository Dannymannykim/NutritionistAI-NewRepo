from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())

    
    model = YOLO('runs/segment/trains/train28/weights/best.pt', task='segment') # 8x works best
    model2 = YOLO('runs/segment/trains/train37/weights/best.pt', task='segment')
    #results = model7.predict(source="datasets/partial/images/train", conf=0.2, save=True, show_boxes=False, project="runs/segment/predicts")
    #results = model2.val(data="coco_seg.yaml", imgsz=640, plots=True, project="runs/segment/vals", split="val", conf=0.34)

    results2 = model2.predict("datasets/full/images/train", save=True, imgsz=640, visualize=False, show=False, save_txt=False, save_crop=True, show_boxes=False, stream=True, project="runs/segment/predicts")
    # getting numbers ofingredients often wrong. consider adding peanutbutter
    # look into f1 confidence curve.

    for r in results2: # stream=True returns a generator which you can loop over
        pass
