import torch.nn as nn
import torch
import matplotlib.pyplot as plt


class ConvNet(nn.Module):
    #constructor of the ConvNet class, using cross entropy as its loss function.
    #the rest is specified by the person creating an instance of this class.
    #model architecture inspired by https://doi.org/10.37398/jsr.2020.640251.
    #can be used both for training the model, but also to create an empty network to load the weights into.
    def __init__(self, 
                 train_loader = None, 
                 test_loader = None, 
                 device = 'cpu', 
                 num_epochs = None, 
                 learning_rate = None):
        super(ConvNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            nn.Conv2d(64, 64, 2),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Flatten(),
            nn.Linear(64*12*12, 64),
            nn.Dropout(p=0.25),
            nn.Linear(64, 10)
        )

        self.train_losses = []
        self.test_losses = []
        self.accuracies = []
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device 
        self.model = None
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.layers(x)

    # evaluate training loss, test loss, and accuracy after each epoch
    def _evaluate_model(self, epoch):
        
        if self.test_loader == None or self.num_epochs == None or self.train_loader == None or self.learning_rate == None: return

        self.eval()

        with torch.no_grad():
            train_loss = sum(self.criterion(self(images.to(self.device)), labels.to(self.device)) for images, labels in self.train_loader)
            test_loss = sum(self.criterion(self(images.to(self.device)), labels.to(self.device)) for images, labels in self.test_loader)

            self.train_losses.append(train_loss / len(self.train_loader))
            self.test_losses.append(test_loss / len(self.test_loader))

            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            self.accuracies.append(accuracy)

        print(f'Epoch [{epoch + 1}/{self.num_epochs}], Accuracy: {accuracy:.2f}%, Train Loss: {self.train_losses[-1]:.4f}, Test Loss: {self.test_losses[-1]:.4f}')

    #function to train the model
    def train_model(self):
        
        if self.test_loader == None or self.num_epochs == None or self.train_loader == None or self.learning_rate == None: return

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        #training loop
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader, 0):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i+1) % 2000 == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Step[{i+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')
                    
            self._evaluate_model(epoch)
            
        #saving the weights
        torch.save(self.state_dict(), './modelweights/model_weights.pth')

    def plots(self):

        if self.num_epochs == None or self.accuracies == None or self.train_losses == None or self.test_losses == None: return
        
        #plotting accuracy, train loss, and test loss
        epochs = range(1, self.num_epochs + 1)

        #accuracy plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
        plt.axhline(y=10, color='r', linestyle='-') #horizontal reference line at 10
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.show()

        #train and test loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_losses, marker='o', linestyle='-', color='r', label='Train Loss')
        plt.plot(epochs, self.test_losses, marker='o', linestyle='-', color='g', label='Test Loss')
        plt.title('Train and Test Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()