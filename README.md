# Neural Network from Scratch (MNIST Digit Recognition)

This repository contains a raw implementation of a 2-layer Neural Network built entirely from scratch using **Python** and **NumPy**. No deep learning frameworks (like TensorFlow or PyTorch) were used. The goal of this project is to demystify the "black box" of neural networks by implementing the underlying mathematics, forward propagation, and backpropagation algorithms manually.

The model is trained on the MNIST dataset to recognize handwritten digits with an accuracy of **~90.35%**.

## 🧠 Project Overview

* **Language:** Python
* **Libraries:** NumPy (Linear Algebra), Pandas (Data Loading)
* **Dataset:** MNIST (Digit Recognition)
* **Architecture:** 2-Layer Perceptron (Input -> Hidden -> Output)
* **Optimization:** Gradient Descent

## 🏗️ Architecture
The network consists of an input layer (784 features), one hidden layer, and an output layer (10 classes).

| Layer | Nodes | Activation Function | Shape |
| :--- | :--- | :--- | :--- |
| **Input** | 784 | N/A | (784, m) |
| **Hidden (Layer 1)** | 10 | ReLU | (10, m) |
| **Output (Layer 2)** | 10 | Softmax | (10, m) |

*Note: `m` represents the number of training examples.* 

## 🧮 The Math (Under the Hood)

This implementation follows standard Forward and Backward propagation derivation. Below are the equations implemented in the code, derived from the project's handwritten notes.

### 1. Forward Propagation
The data flows from input to output to generate a prediction.

$$Z^{[1]} = W^{[1]} X + B^{[1]}$$

$$A^{[1]} = \text{ReLU}(Z^{[1]})$$

$$Z^{[2]} = W^{[2]} A^{[1]} + B^{[2]}$$

$$A^{[2]} = \text{Softmax}(Z^{[2]})$$


### 2. Backward Propagation
We calculate the gradients to update weights using the Chain Rule.

$$dZ^{[2]} = A^{[2]} - Y$$

$$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$$

$$dB^{[2]} = \frac{1}{m} \sum dZ^{[2]}$$

$$dZ^{[1]} = W^{[2]T} dZ^{[2]} \odot g'(Z^{[1]})$$

$$dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T$$

$$dB^{[1]} = \frac{1}{m} \sum dZ^{[1]}$$

### 3. Parameter Updates (Gradient Descent)
$$W^{[i]} = W^{[i]} - \alpha \cdot dW^{[i]}$$

$$B^{[i]} = B^{[i]} - \alpha \cdot dB^{[i]}$$

*Where $\alpha$ is the learning rate.* 

## 📂 File Structure

* `main.py`: Contains all functions for initialization, propagation, training, and prediction.
* `data.csv`: The MNIST dataset (formatted as pixel values with labels).

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MrToshith/Neural-networks-from-scratch.git](https://github.com/MrToshith/Neural-networks-from-scratch.git)
    cd Neural-networks-from-scratch
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy pandas
    ```

3.  **Run the script:**
    Ensure you have the `data.csv` file in the root directory.
    ```python
    python main.py
    ```

## 📊 Results

After training for **1000 iterations** with a learning rate of **0.1**:
**Final Accuracy:** 90.35% 



## 👨‍💻 Author

**P. Toshith** 
