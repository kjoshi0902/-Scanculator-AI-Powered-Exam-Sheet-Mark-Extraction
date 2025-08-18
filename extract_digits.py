import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

model = load_model("red_digit_model.h5")

def extract_red_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return cv2.bitwise_or(mask1, mask2)

def detect_digits_and_sum(image_path):
    image = cv2.imread(image_path)
    mask = extract_red_mask(image)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []
    debug_img = image.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 10 < w < 80 and 20 < h < 100:
            roi = debug_img[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28))
            norm = resized.astype("float32") / 255.0
            input_img = np.expand_dims(norm, axis=(0, -1))
            pred = model.predict(input_img, verbose=0)
            digit = int(np.argmax(pred))
            digits.append(digit)

            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_img, str(digit), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    digits.sort()
    total = sum(digits)

    filename = os.path.basename(image_path)
    processed_filename = f"processed_{filename}"
    processed_path = os.path.join("static/processed", processed_filename)
    cv2.imwrite(processed_path, debug_img)

    return processed_filename, digits, total
