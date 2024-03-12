import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784 # 28x28 images
hidden_size = 100
output_size = 10
epochs = 25
batch_size = 100
learning_rate = 0.005

# MNIST - preparing the dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        o = self.l1(x)
        o = self.relu(o)
        o = self.l2(o)
        return o

model = NeuralNetwork(input_size, hidden_size, output_size)

# Loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_steps = len(train_loader)
for epoch in range(epochs):
     for i, (images, labels) in enumerate(train_loader):
         images = images.reshape(-1, 784).to(device)
         labels = labels.to(device)

         # forward pass
         outputs = model(images)
         loss = criterion(outputs, labels)

         # backward pass
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         if (i+1) % 100 == 0:
             print(f'Epoch:{epoch+1} / {epochs}, step {i+1}/{total_steps}, loss = {loss.item():.4f}')

# Testing loop
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for img, labels in test_loader:
        images = img.reshape(-1, 784).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # value, index 
        _, predictions =  torch.max(outputs, 1)

        # For plotting the images
        # imgs = images[0].reshape(28, 28).cpu().numpy()
        # plt.imshow(imgs, cmap='gray')
        # plt.title(f'Prediction = {predictions[0].item()}')
        # plt.show()

        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')