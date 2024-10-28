import cv2
import random
import os
import imutils
import numpy as np
import shutil
import pandas as pd

def read_yolo_label(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    labels = []
    for line in lines:
        label = line.strip().split()
        labels.append([int(label[0])] + [float(x) for x in label[1:]])
    return labels


def write_yolo_label(label_path, labels):
    with open(label_path, 'w') as file:
        for label in labels:
            file.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")


def random_crop(image_path, label_path, out_image_path, out_label_path):
    image = cv2.imread(image_path)
    labels = read_yolo_label(label_path)

    original_width, original_height = image.shape[1], image.shape[0]
    x_center, y_center = original_width // 2, original_height // 2

    x_left = random.randint(0, x_center // 2)
    x_right = random.randint(original_width - x_center // 2, original_width)
    y_top = random.randint(0, y_center // 2)
    y_bottom = random.randint(original_height - y_center // 2, original_height)

    cropped_image = image[y_top:y_bottom, x_left:x_right]
    cropped_image = cv2.resize(cropped_image, (original_width, original_height))

    new_labels = []
    for label in labels:
        class_id, x_center, y_center, w, h = label
        x_center = (x_center * original_width - x_left) / (x_right - x_left)
        y_center = (y_center * original_height - y_top) / (y_bottom - y_top)
        w = w * original_width / (x_right - x_left)
        h = h * original_height / (y_bottom - y_top)
        new_labels.append([class_id, x_center, y_center, w, h])

    cv2.imwrite(out_image_path, cropped_image)
    write_yolo_label(out_label_path, new_labels)

def random_crop(image_path, label_path, out_image_path, out_label_path):
    image = cv2.imread(image_path)
    labels = read_yolo_label(label_path)

    original_width, original_height = image.shape[1], image.shape[0]
    x_center, y_center = original_width // 2, original_height // 2

    # Xác định vị trí crop
    x_left = random.randint(0, x_center // 2)
    x_right = random.randint(original_width - x_center // 2, original_width)
    y_top = random.randint(0, y_center // 2)
    y_bottom = random.randint(original_height - y_center // 2, original_height)

    # Cắt ảnh và resize lại kích thước ban đầu
    cropped_image = image[y_top:y_bottom, x_left:x_right]
    cropped_image = cv2.resize(cropped_image, (original_width, original_height))

    new_labels = []
    for label in labels:
        class_id, x_center, y_center, w, h = label

        # Cập nhật tọa độ x_center, y_center, width, height
        x_center = (x_center * original_width - x_left) / (x_right - x_left)
        y_center = (y_center * original_height - y_top) / (y_bottom - y_top)
        w = w * original_width / (x_right - x_left)
        h = h * original_height / (y_bottom - y_top)

        # Chỉ thêm các label hợp lệ (kích thước lớn hơn 0)
        if w > 0 and h > 0 and x_center>0 and y_center > 0 and w < 1 and h < 1 and x_center < 1 and y_center < 1 :
            new_labels.append([class_id, x_center, y_center, w, h])
    if not new_labels:
        print(f"{image_path}Error: No valid labels after cropping.")
        return  # hoặc có thể xử lý khác tùy theo yêu cầu
    # Lưu ảnh và nhãn sau khi cắt
    cv2.imwrite(out_image_path, cropped_image)
    write_yolo_label(out_label_path, new_labels)

def change_brightness(image_path, output_path, value):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, img)


def augment_images_and_labels(image_path, label_path, output_image_dir, output_label_dir):
    base_name = os.path.basename(image_path).split('.')[0]

    # Augmentation 1: Random Crop
    random_crop(image_path, label_path,
                f"{output_image_dir}/{base_name}_crop.jpg",
                f"{output_label_dir}/{base_name}_crop.txt")

    # Augmentation 1: Random Crop
    random_crop(image_path, label_path,
                f"{output_image_dir}/{base_name}_crop_1.jpg",
                f"{output_label_dir}/{base_name}_crop_1.txt")

    # Augmentation 2: Change Brightness
    change_brightness(image_path,
                      f"{output_image_dir}/{base_name}_bright.jpg", 50)
    # Brightness change không thay đổi nhãn, nên sao chép nhãn gốc
    shutil.copy(label_path, f"{output_label_dir}/{base_name}_bright.txt")

    # Augmentation 2: Change Brightness
    change_brightness(image_path,
                      f"{output_image_dir}/{base_name}_bright_1.jpg", -50)
    # Brightness change không thay đổi nhãn, nên sao chép nhãn gốc
    shutil.copy(label_path, f"{output_label_dir}/{base_name}_bright_1.txt")


# Ví dụ chạy thử
image_path = "D:/WorkSpace/PycharmProjects/NhanDangBienSo/datasets/resized/images/train"
label_path = "D:/WorkSpace/PycharmProjects/NhanDangBienSo/datasets/resized/labels/train"
output_image_dir = "../NhanDangBienSo/datasets/augment/images/train"
output_label_dir = "../NhanDangBienSo/datasets/augment/labels/train"


for filename in os.listdir(image_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Thay đổi theo định dạng tệp của bạn
        img_path = os.path.join(image_path, filename)
        lbl_path = os.path.join(label_path, f"{filename[:-4]}.txt")

        if os.path.exists(lbl_path):
            augment_images_and_labels(img_path, lbl_path, output_image_dir, output_label_dir)
