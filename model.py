import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential

train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.5,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=1,
    label_mode="binary"
)

val_data = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.5,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=1,
    label_mode="binary"
)

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

model.save("deepfake_model.keras")

print("Model saved successfully!")