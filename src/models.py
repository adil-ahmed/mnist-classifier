import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=64, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)    # (B, 1, 28, 28) -> (B, 784)
        logits = self.net(x)   # raw scores (logits), shape: (B, num_classes)
        return logits

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, hidden_fc=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 28→14
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2),                 # 12→6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*6*6, hidden_fc), nn.ReLU(),
            nn.Linear(hidden_fc, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

