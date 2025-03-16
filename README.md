# Graph Convolutional Network (GCN) - Node Classification

## Overview
This Jupyter Notebook demonstrates the implementation of a **Graph Convolutional Network (GCN)** for node classification using the **Karate Club dataset**. The goal is to classify nodes into different groups using **graph-based deep learning** techniques.

### **Key Features:**
- Converts **graph data** into **structured tensors**
- Uses **GCNConv layers** to process graph-based relationships
- **Leverages dropout, LeakyReLU, and batch normalization** for stable training
- **Optimizes model performance** through tuning hidden layers and learning rates
- **Visualizes embeddings and training progress**

---

## **How It Works**
### **1️⃣ Load and Preprocess the Dataset**
- The dataset used is **KarateClub**, a well-known graph dataset in PyTorch Geometric.
- Each node represents a **person**, and edges represent **connections** (friendships).
- Features, edges, and labels are stored in PyTorch **tensor format**.

### **2️⃣ Define the GCN Model**
We implement a **Graph Convolutional Network (GCN)**, which:
- Aggregates information from **neighboring nodes**.
- Passes it through multiple **GCNConv layers**.
- Applies **LeakyReLU activation** to prevent dead neurons.
- Uses a **softmax output layer** for classification.

#### **Mathematical Formula (Graph Convolution Operation)**
Each layer in GCN is computed as:

$H^{(l+1)} = \sigma( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$

Where:
- $H^{(l)}$ = Node feature matrix at layer \( l \)
- $\tilde{A}$ = **Adjacency matrix** with self-loops
- $\tilde{D}$ = **Degree matrix** of \( \tilde{A} \)
- $W^{(l)}$ = Learnable **weight matrix** for layer \( l \)
- $\sigma$ = Activation function (LeakyReLU)

This operation allows nodes to learn from their neighbors recursively!

### **3️⃣ Train the Model**
- Uses **CrossEntropy Loss** for classification.
- Optimized using **Adam optimizer** with `lr=0.005`.
- Tracks loss and accuracy **over 200 epochs**.

### **4️⃣ Visualize Results**
- **Graph visualization**: Shows node colors based on predicted classes.
- **Training plots**: Displays loss and accuracy trends over epochs.
- **3D Embedding Visualization**: Uses final hidden layer embeddings for a **3D plot**.

---

## 🧠 **Why This Approach Works**
1. **Graph Convolution Captures Node Relationships**
   - Normal CNNs work on **grid data (e.g., images)**, but GCNs allow message passing **over irregular graph structures**.

2. **Multiple GCN Layers Learn Deeper Graph Features**
   - One layer learns **direct neighbors**, multiple layers capture **multi-hop connections**.

3. **LeakyReLU Prevents Dead Neurons**
   - Unlike ReLU, **LeakyReLU (0.1)** prevents neurons from being inactive.

4. **Dropout Regularizes the Model**
   - Prevents **overfitting** by randomly dropping units during training.

