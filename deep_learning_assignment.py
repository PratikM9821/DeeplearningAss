
# Q1: What is TensorFlow 2.0?
# Answer: TensorFlow 2.0 is an improved version of TensorFlow with eager execution, integration with Keras, and a more intuitive API.

# Q2: How to install TensorFlow 2.0
# pip install tensorflow

# Q3: tf.function usage
import tensorflow as tf

@tf.function
def add(a, b):
    return a + b

# Q4: Model class in TensorFlow 2.0
# Used to build models using subclassing API

# Q5: Create a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Q6: Importance of Tensor Space
# Tensors represent data in N-dimensional arrays used in computation.

# Q7: TensorBoard Integration
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
# model.fit(x, y, epochs=5, callbacks=[tensorboard_callback]) # Example

# Q8: TensorFlow Playground
# Web tool to visualize neural network behaviors

# Q9: Netron
# Viewer for deep learning model structures

# Q10: TensorFlow vs PyTorch
# TF: Static computation, PyTorch: Dynamic

# Q11: Install PyTorch
# pip install torch torchvision

# Q12: Basic PyTorch network
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        return self.fc(x)

# Q13: Significance of tensors
# Core data structures in PyTorch for computation

# Q14: torch.Tensor vs torch.cuda.Tensor
# torch.Tensor: CPU; torch.cuda.Tensor: GPU

# Q15: torch.optim
# Module contains optimizers like Adam, SGD

# Q16: Activation Functions
# ReLU, Sigmoid, Tanh, Softmax

# Q17: torch.nn.Module vs torch.nn.Sequential
# Module: Custom; Sequential: Stack of layers

# Q18: Monitor training
# Use callbacks or TensorBoard

# Q19: Keras in TF 2.0
# Integrated as tf.keras

# Q20: Example Project
# MNIST digit classification

# Q21: Advantage of pre-trained models
# Save time and data via transfer learning

# Q22: Verify TensorFlow install
print(tf.__version__)

# Q23: Addition function in TF
@tf.function
def add_fn(a, b):
    return a + b

# Q24: Neural network with 1 hidden layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Q25: Visualize training with Matplotlib
# history = model.fit(...)
# import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])

# Q26: Verify PyTorch install
print(torch.__version__)
print(torch.cuda.is_available())

# Q27: Simple PyTorch NN
simple_model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Q28: Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)

# Q29: Custom loss function
def custom_loss(output, target):
    return torch.mean((output - target)**2)

# Q30: Save and load TensorFlow model
# model.save("model.h5")
# loaded_model = tf.keras.models.load_model("model.h5")
