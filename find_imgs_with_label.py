import os
import shutil
import numpy as np

def get_labels_from_file(file_path):
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():  # skip empty lines
                first_num = line.split()[0]  # first value in row
                labels.append(int(first_num))  # or float(first_num) if needed
    return labels

if __name__ == "__main__":

    # Example usage:
    #folder = "anotations_yolo/labels/foodins_train"
    target_class = 99  # class you’re searching for
    move = True
    copy = False
    folder_images = "datasets/partial/images/train"
    folder_labels = "datasets/partial/labels/train"
    folder_dest = "datasets/partial/removed/" + str(target_class)
    os.makedirs(folder_dest, exist_ok=True)

    counts = np.zeros(103, dtype=int)
    for txt_file in os.listdir(folder_labels):
        if txt_file.endswith(".txt"):
            path_txt = os.path.join(folder_labels, txt_file)

            labels = get_labels_from_file(path_txt)

            for lbl in labels:
                counts[lbl] += 1
            
            if target_class in labels:
                print(f"Class {target_class} found in {txt_file}")

                base_name = os.path.splitext(txt_file)[0]

                # prepare corresponding files
                path_npy = os.path.join(folder_images, base_name + ".npy")
                path_jpg = os.path.join(folder_images, base_name + ".jpg")
                #mg_file_png = os.path.join(folder_images, base_name + ".png")

                if move:
                    shutil.move(path_txt, os.path.join(folder_dest, txt_file))

                    if os.path.exists(path_npy):
                        shutil.move(path_npy, os.path.join(folder_dest, base_name + ".npy"))
                    if os.path.exists(path_jpg):
                        shutil.move(path_jpg, os.path.join(folder_dest, base_name + ".jpg"))
                
                elif copy:
                    shutil.copy(path_txt, os.path.join(folder_dest, txt_file))

                    if os.path.exists(path_npy):
                        shutil.copy(path_npy, os.path.join(folder_dest, base_name + ".npy"))
                    if os.path.exists(path_jpg):
                        shutil.copy(path_jpg, os.path.join(folder_dest, base_name + ".jpg"))

                
print(counts)

            