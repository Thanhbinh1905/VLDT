import cv2
import pandas as pd

# Đọc ảnh
img = cv2.imread("D:/WorkSpace/PycharmProjects/NhanDangBienSo/datasets/augment/images/train/0055_01981_b_crop.jpg")
if img is None:
    print("Error: Could not read the image.")
else:
    # Kích thước ảnh
    img_height, img_width = img.shape[:2]

    # Đọc file nhãn với dấu phân cách là khoảng trắng
    label = pd.read_csv(
        r"D:/WorkSpace/PycharmProjects/NhanDangBienSo/datasets/augment/labels/train/0055_01981_b_crop.txt",
        sep=r"\s+", header=None)

    # Kiểm tra xem có ít nhất 5 cột không
    if label.shape[1] < 5:
        print("Error: The label file does not have enough columns.")
    else:
        # Lặp qua từng nhãn để vẽ hình chữ nhật
        for i in range(len(label)):
            # Tọa độ trung tâm (YOLO format)
            x_center = label.iloc[i, 1] * img_width
            y_center = label.iloc[i, 2] * img_height

            # Kích thước (YOLO format)
            w = label.iloc[i, 3] * img_width
            h = label.iloc[i, 4] * img_height

            # Tính toán tọa độ của góc trên bên trái và góc dưới bên phải của hình chữ nhật
            x = int(x_center - w / 2)
            y = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            # Vẽ hình chữ nhật lên ảnh
            cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)

        # Hiển thị ảnh với hình chữ nhật đã vẽ
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
