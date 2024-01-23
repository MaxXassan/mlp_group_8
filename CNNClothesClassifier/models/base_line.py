import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix, Accuracy, F1Score
from torchmetrics.wrappers import MultioutputWrapper

class BaseLine(nn.Module):
    def __init__(self, 
                 train_loader = None, 
                 test_loader = None, 
                 device = 'cpu', 
                 num_epochs = None, 
                 learning_rate = None,
                 num_classes = 10):
        super(BaseLine, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, num_classes)
        )
        self.multi_wrapper = MultioutputWrapper(Accuracy(task='multiclass', num_classes=num_classes), 2)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.lr = learning_rate

    def forward(self, x):
        return self.layers(x)

    def _evaluate_model(self, epoch):
        pass

    def test_model(self, loader):
        pass

    def train_model(self):
        
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

        train_acc = Accuracy(task='multiclass', num_classes=10)
        test_acc = Accuracy(task='multiclass', num_classes=10)
        acc_data = []
        for epoch in range(self.num_epochs):
            self.train()
            for _, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                out = self(images)
                loss = self.criterion(out, labels)

                loss.backward()
                optimizer.step()

                train_acc.update(out, labels)

            train_acc.compute()
            
            self.eval()
            with torch.inference_mode():
                for _, (images, labels) in enumerate(self.test_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    out = self(images)

                    test_acc.update(out, labels)

            acc_data.append((train_acc.compute().item(), test_acc.compute().item()))
            test_acc.reset()    
            train_acc.reset()

        
            plt.figure(figsize=(10, 6))
            plt.plot(range(epoch + 1), list(zip(*acc_data))[0], label="Training", color="b", marker="o")
            plt.plot(range(epoch + 1), list(zip(*acc_data))[1], label="Validation", color="red", marker="x")
            plt.title('Train and Test Loss Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.show()
            




