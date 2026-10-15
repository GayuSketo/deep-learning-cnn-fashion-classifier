import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from src.data_utils import (
    MODEL_PATH, REPORTS_DIR, ensure_directories, load_fashion_mnist,
    CLASS_NAMES, save_json
)

def main():
    ensure_directories()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train.py` first."
        )

    print("Loading trained CNN...")
    model = tf.keras.models.load_model(MODEL_PATH)
    (_, _), (x_test, y_test) = load_fashion_mnist()

    probabilities = model.predict(x_test, batch_size=256, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1_score": float(f1),
    }
    save_json(metrics, REPORTS_DIR / "evaluation_metrics.json")

    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, zero_division=0
    )
    (REPORTS_DIR / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Fashion-MNIST CNN — Confusion Matrix")
    fig.colorbar(image, ax=ax)
    ticks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(ticks, CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(ticks, CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print("\nTest metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("\nClassification report:")
    print(report)
    print(f"Saved reports in: {REPORTS_DIR}")

if __name__ == "__main__":
    main()
