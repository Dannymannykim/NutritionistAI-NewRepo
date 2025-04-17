from ultralytics.data.converter import convert_coco

# For keypoints data (like person_keypoints_val2017.json)
convert_coco(
    labels_dir="./anotations_coco/",  # Directory containing your json file
    save_dir="./anotations_yolo/",
    use_segments=True,
    cls91to80=False,
    use_keypoints=False  # Since you're using keypoints data
)