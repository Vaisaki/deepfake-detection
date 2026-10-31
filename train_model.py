import tensorflow as tf
from model import build_model

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal")
])

train_data = train_data.map(
    lambda x, y: (augmentation(x, training=True), y)
)

model = build_model()

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

model.save("deepfake_model.keras")

print("Model training completed!")
print("Model saved successfully!")