import numpy as np
import pytesseract
from ultralytics import YOLO
from PIL import  Image
import cv2
import os
pytesseract.pytesseract.tesseract_cmd = r'D:/Tesseract-OCR/tesseract.exe'
testing_images = "D:/WorkSpace/PycharmProjects/NhanDangBienSo/datasets/augment/test/images"
testing_labels = "D:/WorkSpace/PycharmProjects/NhanDangBienSo/datasets/augment/test/labels"
img_test_path = "D:/AT Energy/test2.jpg"
# class = {0: '0', 1: '1', 2: '10', 3: '11', 4: '12', 5: '13', 6: '14', 7: '15', 8: '16', 9: '17', 10: '18', 11: '19', 12: '2', 13: '20', 14: '21', 15: '22', 16: '23', 17: '24', 18: '25', 19: '26', 20: '27', 21: '28', 22: '29', 23: '3', 24: '30', 25: '31', 26: '32', 27: '33', 28: '34', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9'}

if __name__ == '__main__':
    model = YOLO("D:/WorkSpace/runs/detect/train24/weights/best.pt")
    results = model.predict(img_test_path)
    print("orig_img", results)
    for r in results:
        original_image = r.orig_img
    #     im_arr = r.plot()
    #     print(im_arr)
    #     im = Image.fromarray(im_arr[..., ::-1])
    #     # im.show()
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == 27:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                plate_img = original_image[y1:y2, x1:x2]
                # Hiển thị ảnh đã cắt
                # im = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                # im.show()
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                mean_threshold = np.mean(gray)
                _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                smoothed_image = cv2.GaussianBlur(binary_image, (3, 3), 0)
                im = Image.fromarray(plate_img)
                im.show()
                # # print(smoothed_image)
                # num_labels, labels_im = cv2.connectedComponents(binary_image.astype(np.uint8), connectivity=8)
                # print(f"Số lượng thành phần liên kết: {num_labels}")
                # # Tạo một ảnh màu để hiển thị các thành phần
                # colored_labels = np.zeros((labels_im.shape[0], labels_im.shape[1], 3), dtype=np.uint8)
                #
                # for label in range(1, num_labels):
                #     colored_labels[labels_im == label] = np.random.randint(0, 255,
                #                                                            size=3)  # Màu ngẫu nhiên cho mỗi thành phần
                #
                # # Hiển thị kết quả
                # cv2.imshow("Original Image", plate_img)
                # cv2.imshow("Smoothed Image", smoothed_image)
                # cv2.imshow("Connected Components", colored_labels)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    # for filename in os.listdir(testing_images):
    #     if filename.endswith(".jpg") or filename.endswith(".png"):
    #         results = model.predict(filename)


        # print(r.names)
        # print("Bounding box:", r.boxes)

        # # print(im_arr)
        # print(im_arr[...,::-1])

        # print(r.boxes)
        # r.show_labels()
        # for box in r.boxes:
            # print(box.cls[0])
            # print(box.xyxy[0])
            # x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # # print(x1, y1, x2, y2)
            # # Cắt ảnh dựa trên tọa độ hộp giới hạn
            # cropped_image = original_image[y1:y2, x1:x2]
            # # Hiển thị ảnh đã cắt
            # im = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            # im.show()
# for filename in os.listdir(image_path):
#     if filename.endswith(".jpg") or filename.endswith(".png"):  # Thay đổi theo định dạng tệp của bạn
#         img_path = os.path.join(image_path, filename)
#         lbl_path = os.path.join(label_path, f"{filename[:-4]}.txt")
#
#         if os.path.exists(lbl_path):
#             augment_images_and_labels(img_path, lbl_path, output_image_dir, output_label_dir)
