import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Loading MNIST data
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)
X = X/255

# One-hot encode
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

y_one_hot = one_hot_encode(y, 10)

# Split test train
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

print(f" Training samples: {X_train.shape}")
print(f" Test samples: {X_test.shape}")

# Activation functions
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Fixed: Use randn (normal distribution) instead of rand (uniform)
        # Fixed: Proper bias initialization
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))  # Fixed: was b1, should be b2

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = softmax(self.z2)  # Fixed: was softmax(self.a2), should be softmax(self.z2)
        return self.a2    

    def backward(self, X, y, output):  # Fixed: parameter was 'yu', should be 'y'
        m = X.shape[0]
        dL_do = output - y
        dL_dW2 = (self.a1.T @ dL_do) / m
        dL_db2 = np.sum(dL_do, axis=0, keepdims=True) / m
        dL_dh = dL_do @ self.W2.T
        dL_dz1 = dL_dh * relu_derivative(self.z1)
        dL_dW1 = (X.T @ dL_dz1) / m
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True) / m  # Fixed: missing /m
        return dL_dW1, dL_db1, dL_dW2, dL_db2
    
    def update(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs, batch_size, learning_rate):
        print(f"🏋️ Training: {epochs} epochs, batch size {batch_size}")
        print("=" * 60)
        
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                output = self.forward(X_batch)
                dW1, db1, dW2, db2 = self.backward(X_batch, y_batch, output)
                self.update(dW1, db1, dW2, db2, learning_rate)

            # Calculate metrics on training data
            output = self.forward(X)
            loss = -np.mean(np.sum(y * np.log(output + 1e-10), axis=1))

            y_pred = np.argmax(output, axis=1)
            y_true = np.argmax(y, axis=1)
            accuracy = np.mean(y_pred == y_true)

            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Set hyperparameters
input_size = 784
hidden_size = 128
output_size = 10

nn = NeuralNetwork(input_size, hidden_size, output_size)

epochs = 10
batch_size = 64
learning_rate = 0.01

print("Training started...")
nn.train(X_train, y_train, epochs, batch_size, learning_rate)
print("Training complete!")

# Test the network
y_pred = nn.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
test_accuracy = np.mean(y_pred == y_test_labels)

print(f" Final test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")

# Show some predictions
def show_predictions(X, y_true, y_pred, n_samples=8):
    """Display some predictions"""
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    print(f"\n Sample predictions:")
    for i, idx in enumerate(indices):
        true_digit = y_true[idx]
        pred_digit = y_pred[idx]
        status = "yes" if true_digit == pred_digit else "wrong"
        print(f"  Sample {i+1}: True = {true_digit}, Predicted = {pred_digit} {status}")

# Show some predictions
show_predictions(X_test, y_test_labels, y_pred)

