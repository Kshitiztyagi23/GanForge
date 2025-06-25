import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/mask_cnn_model.h5")
labels = ["Masked", "Unmasked", "Partially Masked"]

def preprocess_frame(frame):
    resized = cv2.resize(frame, (128, 128))
    norm = resized / 255.0
    return np.expand_dims(norm, axis=0)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_img = frame.copy()
    input_data = preprocess_frame(face_img)
    pred = model.predict(input_data)[0]
    label = labels[np.argmax(pred)]
    confidence = np.max(pred) * 100

    # Overlay label and confidence
    cv2.putText(frame, f'{label} ({confidence:.2f}%)', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
