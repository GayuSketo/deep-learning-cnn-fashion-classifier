# Deep Learning Major Project: Fashion Product Image Classification Using CNN

## 1. Project overview
Build an end-to-end **Deep Learning** project that classifies a grayscale clothing image into one of 10 Fashion-MNIST categories using a Convolutional Neural Network (CNN).

This project is deliberately limited to deep learning. It does **not** use Generative AI, LLMs, prompt engineering, RAG, vector databases, or pretrained generative models.

### Learning outcomes
- Understand image tensors, normalization, training/validation/test splits.
- Build a CNN with convolution, pooling, dropout, and dense layers.
- Apply ReLU and Softmax activation functions.
- Train with categorical cross-entropy and the Adam optimizer.
- Use early stopping and model checkpointing.
- Evaluate with accuracy, precision, recall, F1-score, and a confusion matrix.
- Save a trained model and use it for inference through a Streamlit interface.

## 2. Dataset
**Fashion-MNIST** contains 70,000 grayscale images, each 28 × 28 pixels:
- 60,000 training images
- 10,000 test images
- 10 clothing categories

TensorFlow/Keras downloads the dataset automatically the first time training is run. An internet connection may be required for this initial download.

| Label | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## 3. Project structure
```text
deep_learning_cnn_fashion_classifier/
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   └── data_utils.py
├── artifacts/          # model and training outputs are saved here
└── reports/            # evaluation reports and charts are saved here
```

## 4. Requirements
Recommended:
- Windows 10/11
- Python 3.11 (64-bit) or Python 3.10
- Visual Studio Code
- VS Code Python extension
- Internet connection for first-time dataset download and package installation

CPU training is supported. A GPU is optional; do not install CUDA separately for this starter project unless you specifically configure a compatible TensorFlow GPU environment.

## 5. Run in VS Code — step by step

### Step 1: Extract the ZIP
Extract `deep_learning_cnn_fashion_classifier.zip` to a folder you can access, for example:
`C:\Users\YourName\Documents\deep_learning_cnn_fashion_classifier`

### Step 2: Open the project
1. Open VS Code.
2. Select **File → Open Folder**.
3. Select the extracted `deep_learning_cnn_fashion_classifier` folder.
4. Install the **Python** extension from Microsoft if it is not already installed.

### Step 3: Open a terminal
Select **Terminal → New Terminal**. Confirm that the terminal is inside the project folder.

### Step 4: Create a virtual environment
On Windows PowerShell:
```powershell
py -3.11 -m venv .venv
```
If Python 3.11 is not installed, install Python 3.11 or use:
```powershell
py -3.10 -m venv .venv
```

Activate it in PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt in VS Code:
```bat
.venv\Scripts\activate.bat
```

### Step 5: Select the virtual environment in VS Code
1. Press `Ctrl+Shift+P`.
2. Search for **Python: Select Interpreter**.
3. Choose the interpreter whose path contains `.venv`.

### Step 6: Install the libraries
Upgrade pip:
```bash
python -m pip install --upgrade pip
```
Install project dependencies:
```bash
pip install -r requirements.txt
```

### Step 7: Train the CNN
```bash
python train.py
```
The first run downloads Fashion-MNIST. Training may take several minutes on a CPU, depending on your computer. When training finishes, the model and reports are saved in `artifacts/` and `reports/`.

Expected generated files include:
- `artifacts/fashion_mnist_cnn.keras`
- `artifacts/training_history.json`
- `reports/training_curves.png`
- `reports/confusion_matrix.png`
- `reports/classification_report.txt`
- `reports/evaluation_metrics.json`

The exact accuracy and training time depend on the environment and training run; no fixed score is guaranteed.

### Step 8: Evaluate the saved model
After training completes:
```bash
python evaluate.py
```
This loads the saved model, evaluates the test set, and refreshes the evaluation report and confusion matrix.

### Step 9: Launch the interactive application
```bash
streamlit run app.py
```
A local web page should open in your browser. If it does not, open the `Local URL` printed in the terminal (typically `http://localhost:8501`).

Use the slider to choose a test image, then click **Predict image** to view the predicted class, confidence, and class probabilities. The app requires a trained model, so run Step 7 first.

Stop the application with `Ctrl+C` in the terminal.

## 6. Libraries used
| Library | Purpose |
|---|---|
| `tensorflow` / `keras` | CNN layers, training, model saving, Fashion-MNIST dataset |
| `numpy` | Array operations and numerical processing |
| `matplotlib` | Training curves and confusion matrix visualization |
| `scikit-learn` | Precision, recall, F1-score, confusion matrix, classification report |
| `streamlit` | Local interactive web application |
| `pillow` | Image rendering support used by Streamlit |

`json`, `pathlib`, and other listed modules from Python's standard library do not need separate installation.

## 7. CNN architecture
1. Input: 28 × 28 × 1 grayscale image
2. Conv2D (32 filters, 3 × 3) + ReLU
3. MaxPooling2D (2 × 2)
4. Conv2D (64 filters, 3 × 3) + ReLU
5. MaxPooling2D (2 × 2)
6. Flatten
7. Dense (128 units) + ReLU
8. Dropout (0.30)
9. Dense (10 units) + Softmax

The model is trained with the Adam optimizer and sparse categorical cross-entropy loss.

## 8. Troubleshooting
- **`python` or `py` not found:** install Python 3.10/3.11 (64-bit), then reopen VS Code.
- **Wrong interpreter / `ModuleNotFoundError`:** select `.venv` using `Python: Select Interpreter`, activate it, and rerun `pip install -r requirements.txt`.
- **Dataset download fails:** check the internet connection and rerun `python train.py`.
- **`No trained model found`:** run `python train.py` before `python evaluate.py` or `streamlit run app.py`.
- **PowerShell activation blocked:** use the Command Prompt activation command shown above.
- **Training is slow:** close other heavy applications; CPU training is expected to take longer than GPU training.

## 9. Suggested student extensions
1. Compare two CNN architectures and document the difference.
2. Experiment with dropout rates and batch sizes.
3. Compare SGD and Adam optimizers.
4. Add data augmentation and compare results.
5. Analyze the most frequently confused classes.
6. Write a short model card covering limitations, intended use, and possible errors.

## 10. Academic submission checklist
- Source code and requirements file
- Dataset description and preprocessing explanation
- CNN architecture diagram or layer summary
- Training and validation curves
- Test metrics and confusion matrix
- Screenshots of the Streamlit app
- Conclusion, limitations, and future enhancements

**Important limitation:** Fashion-MNIST consists of small, centered, grayscale clothing images. This model is intended for this dataset and is not expected to classify arbitrary real-world photos reliably.
