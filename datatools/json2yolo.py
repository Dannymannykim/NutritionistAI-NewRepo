from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="./anotations_coco/",  # directory containing your json file
    save_dir="./anotations_yolo/",
    use_segments=True,
    cls91to80=False,
    use_keypoints=False  # since you're using keypoints data
)