import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
start_time = time.time()


if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA not available, running on CPU")
    dev = torch.device('cpu')
dev = torch.device('cpu')

# Define the neural network class
class DataFittingNN(nn.Module):
    def __init__(self, input_size=3, output_size=3, hidden_size=64):
        super(DataFittingNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# Load data from CSV
csv_file = 'training_data_new.csv'  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Assume the first 3 columns are inputs and the last 3 are outputs
x_data = torch.tensor(data.iloc[:, :3].values, dtype=torch.float32, device=dev)
y_data = torch.tensor(data.iloc[:, -3:].values, dtype=torch.float32, device=dev)

# Split data into training and testing sets
train_ratio = 0.8
train_size = int(train_ratio * len(x_data))
test_size = len(x_data) - train_size

train_indices, test_indices = torch.utils.data.random_split(range(len(x_data)), [train_size, test_size])
x_train = x_data[train_indices]
y_train = y_data[train_indices]
x_test = x_data[test_indices]
y_test = y_data[test_indices]

# Instantiate the model
input_size = 3
output_size = 3
hidden_size = 64
model = DataFittingNN(input_size, output_size, hidden_size)
model = model.to(dev)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop
epochs = 30000
train_losses = []
test_losses = []

for epoch in range(epochs):
    # Training phase
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Record training loss
    train_losses.append(loss.item())

    # Testing phase
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())

    # Print loss every 100 epochs
    # if (epoch + 1) % 100 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

torch.save(model.state_dict(), 'model.pth')

model.load_state_dict(torch.load('model.pth'))

# Predict using the model
x = torch.tensor([5.])
y = torch.linspace(-1., 2., 100)
xi1_full = torch.linspace(0.2, 0.9, 71)
for i in range(len(xi1_full)):
    xi1 = xi1_full[i:i+1]
    print(xi1)
    X, Y, Xi1 = torch.meshgrid(x, y, xi1, indexing='ij')
    test = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Xi1.reshape(-1, 1)))
    T = model(test)
    T_avg = torch.mean(T[:, 2])
    print(f"xi1 = {xi1.item():.6f}, T_avg: {T_avg.item():.6f}")
    # np.savetxt("T_avg.txt", [[xi1.item(), T_avg.item()]], fmt="%.6f", delimiter=" ", newline="\n", header="", footer="", comments="")

    with open("T_avg.txt", "a") as f:
        np.savetxt(f, [[xi1.item(), T_avg.item()]], fmt="%.6f")

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")
