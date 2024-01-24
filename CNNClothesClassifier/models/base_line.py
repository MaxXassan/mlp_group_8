import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix, Accuracy, F1Score
from torchmetrics import ConfusionMatrix, Accuracy, F1Score
import torch
from torch.utils.tensorboard import SummaryWriter
import os
writer = SummaryWriter("baseline_runs")

class BaseLine(nn.Module):
    #constructor of the ConvNet class, using cross entropy as its loss function.
    #the rest is specified by the person creating an instance of this class.
    #model architecture inspired by https://doi.org/10.37398/jsr.2020.640251.
    #can be used both for training the model, but also to create an empty network to load the weights into.
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
        self.tm_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.tm_confusionmatrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.tm_f1score = F1Score(task='multiclass', num_classes=num_classes)
        self.train_losses = []
        self.test_losses = []
        self.accuracies = []
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device 
        self.model = None
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

    def forward(self, x):
        return self.layers(x)

    # evaluate training loss, test loss, and accuracy after each epoch
    def _evaluate_model(self, epoch):
        
        if self.test_loader == None or self.num_epochs == None or self.train_loader == None or self.learning_rate == None: return

        self.eval()

        with torch.no_grad():
            train_loss = sum(self.criterion(self(images.to(self.device)), labels.to(self.device)) for images, labels in self.train_loader)
            eval_loss = sum(self.criterion(self(images.to(self.device)), labels.to(self.device)) for images, labels in self.test_loader)

            self.train_losses.append(train_loss / len(self.train_loader))
            self.test_losses.append(eval_loss / len(self.test_loader))

            writer.add_scalars("Losses", {'train_loss': train_loss / len(self.train_loader),
                                         'eval_loss': eval_loss / len(self.test_loader)}, epoch)
            
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
            writer.add_scalar("Accuracy", accuracy, epoch)
            self.accuracies.append(accuracy)

        print(f'Epoch [{epoch + 1}/{self.num_epochs}], Accuracy: {accuracy:.2f}%, Train Loss: {self.train_losses[-1]:.4f}, Evaluation Loss: {self.test_losses[-1]:.4f}')

    def test_model(self, loader):

        self.eval()

        with torch.no_grad():
            for i, (image, label) in enumerate(loader):
                out = self(image)
                y_hat = torch.softmax(out, dim=1).argmax(dim=1)
                self.tm_confusionmatrix.update(y_hat, label)
                self.tm_accuracy.update(y_hat, label)
                self.tm_f1score.update(y_hat, label)

        # Print F1 score and accuracy
            print(f'F1 Score: {self.tm_f1score.compute():.4f}')
            print(f'Accuracy: {self.tm_accuracy.compute():.4f}')

            # Plot the confusion matrix
            fig, ax = self.tm_confusionmatrix.plot()

            # Add title and class names
            ax.set_title("Confusion Matrix - Baseline model")
            class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
            ax.set_xticklabels(class_names, rotation=45,fontsize='small')
            ax.set_yticklabels(class_names, fontsize='small')

            plt.show()




    #function to train the model
    def train_model(self):
        if self.test_loader == None or self.num_epochs == None or self.train_loader == None or self.learning_rate == None: return

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        #training loopc
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
        

        # Flush the TensorBoard writer
        writer.flush()

        # Saving the weights
        current_directory = os.path.dirname(os.path.abspath(__file__))
        torch.save(self.state_dict(), current_directory+'/modelweights/baseline_model_weights.pth')

    def plots(self):

        if self.num_epochs == None or self.accuracies == None or self.train_losses == None or self.test_losses == None: return
        
        #plotting accuracy, train loss, and test loss
        epochs = range(1, self.num_epochs + 1)

        #accuracy plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
        plt.axhline(y=10, color='r', linestyle='-') #horizontal reference line at 10
        plt.title('Accuracy Over Epochs - Baseline model')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.show()

        #train and test loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_losses, marker='o', linestyle='-', color='r', label='Train Loss')
        plt.plot(epochs, self.test_losses, marker='o', linestyle='-', color='g', label='Evaluation Loss')
        plt.title('Train and Evaluation Loss Over Epochs - Baseline model')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()