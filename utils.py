import os
import cv2

def load_images_from_folder(folder_path):
    images = []
    paths = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(folder_path, file)
            img = cv2.imread(full_path)

            if img is None:
                continue

            h, w = img.shape[:2]
            if h * w > 2000 * 2000:
                img = cv2.resize(img, (800, 600))

            images.append(img)
            paths.append(full_path)

    return images, paths
