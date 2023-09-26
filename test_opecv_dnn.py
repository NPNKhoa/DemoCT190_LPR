import cv2
import numpy as np
import time
import pytesseract
from skimage.filters import threshold_local

def image_preprocessing(image):
    # Fix DPI (if needed)
    def fix_dpi(image):
        new_width = int(image.shape[1] * 300 / 72) # 72 DPI là giá trị mặc định
        new_height = int(image.shape[0] * 300 / 72)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized_image
    
    # Fix text size (e.g. 12pt)
    def fix_text_size(image):
        rescaled_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        return rescaled_image
    def apply_local_threshold(image):
        # Chuyển đổi không gian màu từ BGR sang HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Tách kênh V
        _, _, v = cv2.split(hsv)
        # Áp dụng phương pháp ngưỡng cục bộ
        binary = threshold_local(v, 35, offset=5, method="gaussian")
        binary = (v > binary).astype("uint8") * 255
        return binary
    def fix_image(image):
        # Chuyển đổi sang không gian màu HSV và lấy kênh độ sáng (Value)
        V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]

        # Tách ngưỡng ảnh V sử dụng adaptive thresholding
        T = threshold_local(V, 35, offset=5, method="gaussian")
        thresh = (V > T).astype("uint8") * 255

        # Đảo ngược ngưỡng: đen thành trắng và ngược lại
        thresh = cv2.bitwise_not(thresh)

        # Tìm các thành phần liên thông
        _, labels = cv2.connectedComponents(thresh)

        # Tạo mask để chứa các khu vực cần thiết
        mask = np.zeros(thresh.shape, dtype="uint8")
        total_pixels = thresh.shape[0] * thresh.shape[1]
        lower = total_pixels // 120
        upper = total_pixels // 20

        # Duyệt qua các nhãn thành phần liên thông và lấy các khu vực cần thiết vào mask
        for label in np.unique(labels):
            if label == 0:
                continue
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            if numPixels > lower and numPixels < upper:
                mask = cv2.add(mask, labelMask)
        return mask
    def fix_illumination(image):
        # Convert image to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Split LAB image into L, A, and B channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        # Apply histogram equalization to L channel
        equalized_l_channel = cv2.equalizeHist(l_channel)
        # Merge equalized L channel with original A and B channels
        equalized_lab_image = cv2.merge([equalized_l_channel, a_channel, b_channel])
        # Convert equalized LAB image back to BGR format
        equalized_bgr_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2BGR)
        return equalized_bgr_image
    
    def fix_scale(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_cnt = cnt

        rect = cv2.minAreaRect(max_cnt)
        ((cx, cy), (cw, ch), angle) = rect

        M = cv2.getRotationMatrix2D((cx, cy), angle - 90, 1)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, thresh_rotated = cv2.threshold(gray_rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours_rotated, _ = cv2.findContours(thresh_rotated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area_rotated = 0
        max_cnt_rotated = None
        for cnt in contours_rotated:
            area = cv2.contourArea(cnt)
            if area > max_area_rotated:
                max_area_rotated = area
                max_cnt_rotated = cnt

        x, y, w, h = cv2.boundingRect(max_cnt_rotated)
        cropped = rotated[y:y + h, x:x + w]
        return cropped
    # Thực hiện các bước xử lý hình ảnh
    processed_image = fix_dpi(image)
    processed_image = fix_text_size(processed_image)
    processed_image = fix_illumination(processed_image)
    processed_image = fix_image(processed_image)

    # Trả về hình ảnh đã được xử lý
    return processed_image


def build_tesseract_options(psm=6, whitelist=None, allow_multiline=True):
    # tell Tesseract to only OCR alphanumeric characters by default
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    if whitelist:
        # add additional characters to the whitelist
        options += whitelist
    # Add dot and hyphen characters to the whitelist
    options += ".-"
    # set the PSM mode
    options += " --psm {}".format(psm)
    if allow_multiline:
        # enable multiple lines for OCR
        options += " --oem 1"
    # return the built options string
    return options
# Load model YOLOv4-tiny pre-trained trên COCO dataset
net = cv2.dnn.readNet('custom-yolov4-tiny-detector_best.weights', 'custom-yolov4-tiny-detector.cfg')

# List các tên lớp trong COCO dataset
classes = []
with open('obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Khởi tạo font chữ
font = cv2.FONT_HERSHEY_SIMPLEX

# Màu sắc cho tất cả các đối tượng
color = (0, 255, 0)

# Đọc video từ camera
cap = cv2.VideoCapture(0)
desired_width = 960
desired_height = 720

# Khởi tạo biến thời gian để tính fps và độ trễ
prev_time = 0
delay_time = 0
options = build_tesseract_options(psm=6)
while True:
    # Đọc frame từ camera
    ret, frame = cap.read()

    #cấu hình kích thước đầu ra video
    frame = cv2.resize(frame, (desired_width, desired_height))
    
    # Chuyển frame thành blob để đưa vào mạng YOLOv4-tiny
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Lấy thông tin output từ mạng YOLOv4-tiny
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)
    
    # Khởi tạo các list để lưu thông tin về đối tượng
    boxes = []
    confidences = []
    class_ids = []
    color = (0, 255, 0)
    # Xử lý output để lấy thông tin về đối tượng
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Áp dụng non-max suppression để loại bỏ các box trùng nhau
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Vẽ hình chữ nhật và viết chữ trên hình chữ nhật cho các đối tượng còn lại
    # Màu sắc cho tất cả các đối tượng
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            plate = frame[y:y+h, x:x+w].copy()
            if plate.any():
                processed_image = image_preprocessing(plate)
                lpText = pytesseract.image_to_string(processed_image, config=options)
                lpText = lpText.replace("\n", "")
                text_width, text_height = cv2.getTextSize(lpText, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.putText(frame, lpText, (x, y + text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (266, 106, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f'{label} {confidence:.2f}'
            cv2.putText(frame, text, (x, y - 5), font, 1, color, 2)

    # Tính fps và độ trễ
    current_time = time.time()
    elapsed_time = current_time - prev_time
    prev_time = current_time
    fps = 1 / elapsed_time
    delay_time = max(1, int((1000/fps) - elapsed_time))

    fps_text = f'FPS: {fps:.2f}'
    cv2.putText(frame, fps_text, (10, 50), font, 1, color, 2)

    delay_text = f'Delay: {delay_time} ms'
    cv2.putText(frame, delay_text, (10, 100), font, 1, color, 2)

    cv2.imshow('frame', frame)
    # Thoát khỏi vòng lặp khi nhấn phím ESC
    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()