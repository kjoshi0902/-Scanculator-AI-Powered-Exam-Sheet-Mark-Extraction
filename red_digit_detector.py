import cv2
import numpy as np
from tensorflow.keras.models import load_model

def detect_digits_and_sum(image_path, output_path):
    model = load_model("red_digit_model.h5")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    debug_img = image.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 10 < w < 80 and 20 < h < 100:
            pad = 5
            x1, y1 = max(x - pad, 0), max(y - pad, 0)
            x2, y2 = min(x + w + pad, image.shape[1]), min(y + h + pad, image.shape[0])
            roi = image[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28))
            norm = resized.astype("float32") / 255.0
            input_img = np.expand_dims(norm, axis=(0, -1))
            pred = model.predict(input_img, verbose=0)
            digit = int(np.argmax(pred))
            digits.append((y, x, digit))
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_img, str(digit), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    digits.sort(key=lambda x: (x[0] // 20, x[1]))  # Sort by row then x
    digit_values = [d[2] for d in digits]
    total = sum(digit_values)

    # Save debug image
    cv2.imwrite(output_path, debug_img)

    return digit_values, total
