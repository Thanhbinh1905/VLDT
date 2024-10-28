import cv2
import random
import os
import imutils
import numpy as np

# Hàm đọc nhãn từ file YOLO format
def read_yolo_label(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    labels = []
    for line in lines:
        label = line.strip().split()
        labels.append([int(label[0])] + [float(x) for x in label[1:]])
    return labels

# Hàm ghi nhãn vào file YOLO format
def write_yolo_label(label_path, labels):
    with open(label_path, 'w') as file:
        for label in labels:
            file.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")

# Thay đổi kích thước ngẫu nhiên ảnh và cập nhật nhãn
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

    # Cập nhật nhãn
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

# Xoay ảnh ngẫu nhiên và cập nhật nhãn
def rotate_image(image_path, label_path, range_angle, out_image_path, out_label_path):
    image = cv2.imread(image_path)
    labels = read_yolo_label(label_path)

    original_height, original_width = image.shape[:2]
    angle = random.randint(-range_angle, range_angle)
    rotated_image = imutils.rotate_bound(image, angle)

    # Cập nhật nhãn
    new_labels = []
    for label in labels:
        class_id, x_center, y_center, w, h = label
        x_center = x_center * original_width
        y_center = y_center * original_height
        box = np.array([[x_center - w * original_width / 2, y_center - h * original_height / 2],
                        [x_center + w * original_width / 2, y_center - h * original_height / 2],
                        [x_center - w * original_width / 2, y_center + h * original_height / 2],
                        [x_center + w * original_width / 2, y_center + h * original_height / 2]])
        # Xoay tọa độ box
        rot_matrix = cv2.getRotationMatrix2D((original_width / 2, original_height / 2), angle, 1)
        rotated_box = np.dot(box, rot_matrix[:, :2].T) + rot_matrix[:, 2]

        x_min = rotated_box[:, 0].min()
        y_min = rotated_box[:, 1].min()
        x_max = rotated_box[:, 0].max()
        y_max = rotated_box[:, 1].max()

        new_x_center = (x_min + x_max) / 2 / rotated_image.shape[1]
        new_y_center = (y_min + y_max) / 2 / rotated_image.shape[0]
        new_w = (x_max - x_min) / rotated_image.shape[1]
        new_h = (y_max - y_min) / rotated_image.shape[0]
        new_labels.append([class_id, new_x_center, new_y_center, new_w, new_h])

    cv2.imwrite(out_image_path, rotated_image)
    write_yolo_label(out_label_path, new_labels)

# Thay đổi độ sáng ngẫu nhiên (không cần cập nhật nhãn)
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

# Tạo ảnh và nhãn mới sau khi augment
def augment_images_and_labels(image_path, label_path, output_image_dir, output_label_dir):
    base_name = os.path.basename(image_path).split('.')[0]

    # Augmentation 1: Random Crop
    random_crop(image_path, label_path,
                f"{output_image_dir}/{base_name}_crop.jpg",
                f"{output_label_dir}/{base_name}_crop.txt")

    # Augmentation 2: Rotate Image
    rotate_image(image_path, label_path, 15,
                 f"{output_image_dir}/{base_name}_rotate.jpg",
                 f"{output_label_dir}/{base_name}_rotate.txt")

    # Augmentation 3: Change Brightness
    change_brightness(image_path,
                      f"{output_image_dir}/{base_name}_bright.jpg", 50)
    # Brightness change không thay đổi nhãn, nên sao chép nhãn gốc
    os.system(f"cp {label_path} {output_label_dir}/{base_name}_bright.txt")

# Ví dụ chạy thử
image_path = "../NhanDangBienSo/datasets/resized/images/train"
label_path = "../NhanDangBienSo/datasets/resized/labels/train"
output_image_dir = "../NhanDangBienSo/datasets/augment/labels/train"
output_label_dir = "../NhanDangBienSo/datasets/augment/labels/train"

augment_images_and_labels(image_path, label_path, output_image_dir, output_label_dir)
