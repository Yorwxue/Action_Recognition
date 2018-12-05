import numpy as np
import torch
from torchvision import models
from tqdm import tqdm

from non_local_block.dot_product import NONLocalBlock2D


class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(56180, 50)
        self.fc2 = torch.nn.Linear(50, 101)

    def forward(self, x):
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 56180)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class NonLocalNetwork(torch.nn.Module):
    def __init__(self, in_channels, num_labels):
        super(NonLocalNetwork, self).__init__()

        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            NONLocalBlock2D(in_channels=32),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            NONLocalBlock2D(in_channels=64),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=128*3*3, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(in_features=256, out_features=num_labels)
        )

    def forward(self, x):
        batch_size = x.size(0)
        output = self.convs(x).view(batch_size, -1)
        output = self.fc(output)
        return output


class CreateModel(object):
    def __init__(self, device, in_channels, num_labels):
        # vgg16 model
        """
        # self.model = models.vgg16(pretrained=True)
        # for parma in self.model.parameters():
        #     parma.requires_grad = False
        # self.model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
        #                                             torch.nn.ReLU(),
        #                                             torch.nn.Dropout(p=0.5),
        #                                             torch.nn.Linear(4096, 4096),
        #                                             torch.nn.ReLU(),
        #                                             torch.nn.Dropout(p=0.5),
        #                                             torch.nn.Linear(4096, 101))
        # self.optim_params = self.model.classifier.parameters()
        # """

        # small model for testing
        """
        # self.model = TestNet()
        # self.optim_params = self.model.parameters()
        # """

        # Non-local Neural Network
        # """
        self.model = NonLocalNetwork(in_channels=in_channels, num_labels=num_labels)
        self.optim_params = self.model.parameters()
        # """

        self.model = self.model.to(device)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.optim_params)

    def train(self, dataloader, device, num_epoch):
        """

        :param dataloader: pytorch pipeline
        :param num_epoch:
        :param use_gpu:
        :return:
        """
        self.model.train()

        for epoch_idx in range(num_epoch):
            batch_loss = 0.0
            batch_correct = 0
            for batch_idx, data in enumerate(tqdm(dataloader), 1):
                x = data["input"]["flow"]
                y = data["label"]

                try:
                    x, y = x.to(device, dtype=torch.float32),  y.to(device)

                    self.optimizer.zero_grad()

                    output = self.model(x)

                    # get the index of the max log-probability
                    pred = output.max(1, keepdim=True)[1]

                    loss = self.loss_func(output, y)

                    loss.backward()
                    self.optimizer.step()

                    batch_loss += loss.item()
                    batch_correct += pred.eq(y.view_as(pred)).sum().item()

                    if batch_idx % 500 == 0:
                        print("Batch %d, Train Loss: %.4f, Train ACC: %.4f" %
                              (batch_idx, batch_loss / (4 * batch_idx), 100 * batch_correct / (4 * batch_idx)))
                except Exception as e:
                    print(e)
                    pass

            epoch_loss = batch_loss / dataloader.__len__()
            epoch_correct = 100 * batch_correct / dataloader.__len__()
            print("Training  Loss: %.4f,  Correct %.4f" % (epoch_loss, epoch_correct))

    def test(self, x, y, use_gpu=True):
        self.model.eval()

        if use_gpu:
            x, y = torch.autograd.Variable(x.cuda()), torch.autograd.Variable(y.cuda())
        else:
            x, y = torch.autograd.Variable(x), torch.autograd.Variable(y)


""""
if __name__ == "__main__":
    model = CreateModel()
    print(model)
"""
