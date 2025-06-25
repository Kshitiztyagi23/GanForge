from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np

img_size = 128
batch_size = 32

model = load_model("model/mask_cnn_model.h5")

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val = val_datagen.flow_from_directory(
    'dataset/images',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

predictions = model.predict(val)
y_pred = np.argmax(predictions, axis=1)
y_true = val.classes

print("CLASS INDICES:", val.class_indices)
print(classification_report(y_true, y_pred, target_names=val.class_indices.keys()))
