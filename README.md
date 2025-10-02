# 👗 Fashion MNIST Image Classification

A deep learning project that implements and evaluates neural network models on the **Fashion MNIST** dataset using **PyTorch**.  
The goal of this project was to classify images of clothing (10 categories) such as T-shirts, trousers, dresses, shoes, etc., and achieve high accuracy with simple but effective architectures.

---

## 🚀 Project Highlights
- **Frameworks & Tools:** PyTorch, NumPy, Matplotlib, scikit-learn  
- **Dataset:** Fashion MNIST (60,000 training images, 10,000 test images, 28x28 grayscale)  
- **Model Architecture:** Custom `nn.Module` with Dense layers, Dropout regularization, and training loop with `torch.optim`.  
- **Accuracy:** Achieved ~90.39% test accuracy across 10 fashion categories.  
- **Evaluation:** Classification report, confusion matrix, and learning curves to visualize model performance.  
- **Code Structure:** Notebook organized into data loading, preprocessing, model training, and evaluation for reproducibility.

---

## 📊 Results
- Training and validation curves show good generalization with Dropout regularization.  
- Confusion matrix highlights most common misclassifications (e.g., shirt vs. T-shirt).  
- Overall test accuracy: **≈90.39%**.  

---

## 🔧 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fashion-mnist-classification.git
   cd fashion-mnist-classification
   
2. Install dependencies:
  ``` bash
  pip install torch torchvision matplotlib scikit-learn
```

3. Open and run the notebook:
``` bash
jupyter notebook Fashion_MNIST_image_Classification.ipynb
```
