import json
import matplotlib.pyplot as plt
import tensorflow as tf
from src.data_utils import (
    ARTIFACTS_DIR, REPORTS_DIR, MODEL_PATH, ensure_directories,
    load_fashion_mnist, save_json
)

def build_model():
    """Create a CNN classifier for 28x28 grayscale images."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    ensure_directories()
    tf.keras.utils.set_random_seed(42)

    print("Loading Fashion-MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    print(f"Training data: {x_train.shape}; Test data: {x_test.shape}")

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH), monitor="val_loss",
            save_best_only=True, verbose=1
        ),
    ]

    print("\nTraining CNN...")
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
        verbose=1,
    )

    # Save the best restored model in Keras format.
    model.save(MODEL_PATH)
    history_data = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    save_json(history_data, ARTIFACTS_DIR / "training_history.json")

    # Plot training and validation accuracy/loss.
    epochs = range(1, len(history_data["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history_data["accuracy"], label="Training accuracy")
    axes[0].plot(epochs, history_data["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history_data["loss"], label="Training loss")
    axes[1].plot(epochs, history_data["val_loss"], label="Validation loss")
    axes[1].set_title("CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "training_curves.png", dpi=150)
    plt.close(fig)

    print("\nEvaluating on the held-out test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    save_json(
        {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)},
        REPORTS_DIR / "evaluation_metrics.json"
    )
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Training curve: {REPORTS_DIR / 'training_curves.png'}")
    print("Next: run `python evaluate.py` for detailed classification metrics.")

if __name__ == "__main__":
    main()
