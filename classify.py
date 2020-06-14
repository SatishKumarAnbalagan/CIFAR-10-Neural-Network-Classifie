"""
classify.py

Neural network classifier

Created by Satish Kumar Anbalagan on 2/5/20.
Copyright © 2020 Satish Kumar Anbalagan. All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from os import path
import matplotlib.pyplot as plt
from PIL import Image   # Loading the test image

epochs = 20

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.CIFAR10(
                    root='./data.cifar10',                 # dataset location
                    train=True,                            # this is training data
                    transform=transform,                   # Converts image to torch.FloatTensor of shape (C x H x W)
                    download=True                          # download dataset
)

train_loader = tdata.DataLoader(dataset=train_data, batch_size=6000, shuffle = True )

test_data = torchvision.datasets.CIFAR10(root='./data.cifar10/', train=False, transform=transform)

test_loader = tdata.DataLoader(dataset=test_data, batch_size=5000, shuffle=True)

criteria = nn.CrossEntropyLoss()

classes = ('aeroplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# define the NN architecture
class Net(nn.Module):
    """
    Class to define the Neural network classifier model architecture
    """
    def __init__(self):
        super(Net, self).__init__()
        # number of hidden nodes in each layer
        hidden_1 = 2048
        hidden_2 = 1024
        # linear layer (3072 -> hidden_1)
        self.fc1 = nn.Linear(32 * 32 * 3, hidden_1)
        # linear layer (hidden_n -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (hidden_n -> 10)
        self.output = nn.Linear(hidden_2, 10)

        # dropout layer (p=0.2) - prevents over fitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 32 * 32 * 3)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)

        # add output layer
        x = self.output(x)

        return x


model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


def train():
    """
    Function to train the model.
    :rtype: training accuracy and training loss

    """
    lossTotal = 0
    trainTotal = 0
    trainCorrect = 0
    for images, labels in train_loader:
        model.train()
        optimizer.zero_grad()
        outputs = model(images)

        # loss
        loss = criteria(outputs, labels)
        loss.backward()
        optimizer.step()
        lossTotal = lossTotal + loss.item()

        # accuracy
        _, predicted = torch.max(outputs.data, 1)
        trainTotal += labels.size(0)
        trainCorrect += predicted.eq(labels.data).sum().item()
        train_accuracy = 100 * trainCorrect / trainTotal
    return train_accuracy, loss.item()


def test():
    """
    Function to test the model.
    :rtype: test accuracy and test loss

    """
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images)
                loss = criteria(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_accuracy = 100 * correct / total
    return test_accuracy, loss.item()


def predict(inputFilePath):
    """
    Function to predict the input image.
    :param: input image path
    :rtype: None

    """
    # Loading the model
    generatedModel = torch.load('./model/model.pt')

    inputImage = Image.open(inputFilePath).convert('RGB')

    inputImage = inputImage.resize((32, 32))

    with torch.no_grad():
        # Image Transformation
        imageTensor = transform(inputImage)
        singleBatchImage = imageTensor.unsqueeze(0)  # shape = [1, 3, 32, 32]

        outputs = generatedModel(singleBatchImage)
        _, predicted = torch.max(outputs.data, 1)
        predictedClass = classes[predicted[0].item()]
        print("Predicted Class : {}".format(predictedClass))


def training():
    """
    Function to train the model.
    :rtype: None

    """
    print("{} epochs for training".format(epochs))
    for i in range(epochs):
        (trainingModel, trainingLoss) = train()
        (testingModel, testingLoss) = test()

        print('Loop {:2s}, train Loss: {:.3f}'.format(str(i), trainingLoss), ", Training Accuracy: {}%".format(trainingModel),
              ", Test Loss: {:.3f}".format(testingLoss), ', Test Accuracy: {}%'.format(testingModel))

    torch.save(model, './model/model.pt')
    print("Saved Neural network classifier model successfully at {}/model/".format(os.getcwd()))


def main():
    """
    main function
    :return: None
    """
    parser = argparse.ArgumentParser(description='Neural network classifier')
    parser.add_argument('function', nargs='*', type=str,
                    help='train/test xxx.png, enter it to train/test the Neural network classifier model accordingly')

    args = parser.parse_args()
    if len(args.function) >= 1:
        if args.function[0] == 'train':
            print('\nModel training')
            training()
        elif args.function[0] == 'test':
            inputFile = args.function[1]
            if path.exists(inputFile):
                print('\nTesting/Predicting')
                predict(inputFile)
            else:
                print('\nInvalid input. Path does not exists')
        else:
            print('\nPlease enter a valid command')
            parser.print_help()
    else:
        print('\nPlease enter a valid command')
        parser.print_help()

    exit()


if __name__ == '__main__':
    main()
