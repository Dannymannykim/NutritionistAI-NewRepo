from ultralytics import YOLO
import cv2
import torch

def inference(model_pth, img_pth, display=False):
    
    model = YOLO(model_pth, task='segment')

    results = model.predict(
        img_pth, 
        save=False, 
        imgsz=640, 
        visualize=False, 
        show=False, # 
        show_labels=True, 
        save_txt=False, 
        show_boxes=True, 
        project="runs/segment/predicts",
        conf=0.52,
        #iou=0.4
    )
    
    if display:
        img = results[0].plot()
        
        cv2.imshow("YOLO Prediction", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # reference; for mask area comptutation it should only be single image
    for result in results: # stream=True returns a generator which you can loop over
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        #result.show()  # display to screen
    
    areas = {}

    # masks are calculated from resized image, not original; adjust dimensions for coin mask
    #masks = result.masks.data          # shape (N, H, W)
    #print("orig shape (h,w):", getattr(result, "orig_shape", None))
    #print("masks.shape:", masks.shape)
    #print("total image pixels:", masks.shape[1] * masks.shape[2])
    
    for cls_id in set(boxes.cls.int().tolist()): # boxes.cls is a float tensor of class ids
        # collect masks belonging to class cls_id
        class_masks = [m for cid, m in zip(boxes.cls, masks.data) if int(cid.item()) == cls_id]
        
        if not class_masks:
            continue
        
        # stack masks -> (num_masks, h, w)
        stacked = torch.stack(class_masks)

        # merge via OR (any pixel covered counts once)
        merged = (stacked > 0.5).any(dim=0)  # bool mask of shape (h, w)
        
        # area = count of True pixels
        areas[cls_id] = merged.sum().item()

    # convert to dict of class_name : area
    class_names = result.names
    areas_named = {class_names[k]: v for k, v in areas.items()}
    
    return areas_named

if __name__ == '__main__':
    # for testing and debugging

    #model = YOLO('runs/segment/trains/train28/weights/best.pt', task='segment') # 8x works best
    #model2 = YOLO('runs/segment/trains/train37/weights/best.pt', task='segment')
    model3 = YOLO('runs/segment/trains/train41/weights/best.pt', task='segment')
    #results = model7.predict(source="datasets/partial/images/train", conf=0.2, save=True, show_boxes=False, project="runs/segment/predicts")
    #results = model2.val(data="coco_seg.yaml", imgsz=640, plots=True, project="runs/segment/vals", split="val", conf=0.34)

    #results2 = model2.predict("datasets/full/images/train", save=True, imgsz=640, visualize=False, show=False, save_txt=False, save_crop=True, show_boxes=False, stream=True, project="runs/segment/predicts")
    
    results = model3.predict("datasets/test/true/food10.jpg", save=False, imgsz=640, show=False, project="runs/segment/predicts")
    
    img = results[0].plot()

    cv2.imshow("YOLO Prediction", img)
    cv2.waitKey(0)    
    cv2.destroyAllWindows()

