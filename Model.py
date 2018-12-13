import os
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
            torch.nn.Linear(in_features=128*28*28, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(in_features=256, out_features=num_labels)
        )

    def forward(self, x):
        batch_size = x.size(0)
        output = self.convs(x).view(batch_size, -1)
        output = self.fc(output)
        return output


class VGG16(torch.nn.Module):
    def __init__(self, num_labels, pretrained):
        super(VGG16, self).__init__()

        self.module = models.vgg16(pretrained=pretrained)
        for parma in self.module.parameters():
            parma.requires_grad = False
        self.module.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Dropout(p=0.5),
                                                     torch.nn.Linear(4096, 4096),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Dropout(p=0.5),
                                                     torch.nn.Linear(4096, num_labels))

    def forward(self, x):
        output = self.module(x)
        return output


class InceptionV3(torch.nn.Module):
    def __init__(self, num_labels, pretrained):
        """

        :param num_labels: number of classes
        :param pretrained: using pretrained model or not
        :param num_freeze: number of layers to freeze
        """
        super(InceptionV3, self).__init__()

        self.module = models.inception_v3(pretrained=pretrained)

        # freeze several layers
        for parma in self.module.parameters():
            parma.requires_grad = False

        # replace the last layer
        self.module.aux_logits = False
        self.module.fc = torch.nn.Linear(self.module.fc.in_features, num_labels)

    def forward(self, x):
        x = self.module(x)
        return x


class CreateModel(object):
    def __init__(self, device, num_labels, in_channels=3):
        # vgg16 model(model for image)
        """
        self.model = VGG16(num_labels=num_labels, pretrained=True)
        self.optim_params = self.model.module.classifier.parameters()
        # """

        # inception model(model for image)
        # """
        self.model = InceptionV3(num_labels=num_labels, pretrained=True)
        self.optim_params = self.model.module.fc.parameters()
        # """

        # small model for testing
        """
        self.model = TestNet()
        self.optim_params = self.model.parameters()
        # """

        # Non-local Neural Network(model for video)
        """
        self.model = NonLocalNetwork(in_channels=in_channels, num_labels=num_labels)
        self.optim_params = self.model.parameters()
        # """

        self.model = self.model.to(device)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.optim_params)

    def train(self, dataloader, device, num_epoch, display_freq, model_path=None):
        """

        :param dataloader: pytorch pipeline
        :param num_epoch:
        :param use_gpu:
        :return:
        """
        model_dir, model_name = model_path[:model_path.rindex('/')], model_path[model_path.rindex('/')+1:]
        self.model.train()

        for epoch_idx in range(num_epoch):
            print("EPOCH %d" % (epoch_idx+1))
            batch_loss = 0.0
            batch_correct = 0.0
            epoch_loss = 0.0
            epoch_correct = 0
            for batch_idx, data in enumerate(tqdm(dataloader), 1):
                x = data["input"]
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

                    epoch_loss += loss.item()
                    epoch_correct += pred.eq(y.view_as(pred)).sum().item()

                    if batch_idx % display_freq == 0:
                        print("Batch %d, Training Loss: %.4f, Training ACC: %.4f" % (
                            batch_idx,
                            batch_loss / (dataloader.batch_size * display_freq),
                            100 * batch_correct / (dataloader.batch_size * display_freq)))
                        batch_loss = 0.0
                        batch_correct = 0.0
                        if model_path:
                            self.save_model(model_dir=model_dir, model_name=model_name)
                except Exception as e:
                    print(e)
                    pass

            epoch_loss = epoch_loss / (dataloader.__len__() * dataloader.batch_size)
            epoch_acc = 100 * epoch_correct / (dataloader.__len__() * dataloader.batch_size)
            print("Training Loss: %.4f,  Training ACC %.4f%%" % (epoch_loss, epoch_acc))

            if model_path:
                self.save_model(model_dir=model_dir, model_name=model_name)

    def test(self, dataloader, device):
        self.model.eval()

        loss = 0.0
        correct = 0
        for batch_idx, data in enumerate(tqdm(dataloader), 1):
            x = data["input"]
            y = data["label"]

            if batch_idx == 500:
                print("Test Loss: %.4f, Test ACC: %.4f%%" % (
                    loss / (500 * dataloader.batch_size),
                    100 * correct / (500 * dataloader.batch_size)))

            try:
                x, y = x.to(device, dtype=torch.float32), y.to(device)

                output = self.model(x)

                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]

                loss += self.loss_func(output, y).item()
                correct += pred.eq(y.view_as(pred)).sum().item()
            except Exception as e:
                print(e)

        print("Test Loss: %.4f, Test ACC: %.4f%%" % (
            loss / (dataloader.__len__() * dataloader.batch_size),
            100 * correct / (dataloader.__len__() * dataloader.batch_size)))

    def save_model(self, model_dir, model_name=None, entire_model=True):
        """
        function used to save model(state_dict mode just complete and without testing)
        Notice that the load_state_dict() function takes a dictionary object, NOT a path to a saved object.
        This means that you must deserialize the saved state_dict before you pass it to the load_state_dict() function.
        For example, you CANNOT load using model.load_state_dict(PATH).
        :param model:
        :param model_dir:
        :param model_name:
        :param entire_model:
        :return:
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if entire_model:
            # Save Entire Model
            torch.save(self.model, os.path.join(model_dir, model_name))
        else:
            # for inference
            # Notice that the load_state_dict() function takes a dictionary object
            torch.save(self.model.state_dict(), model_dir)

    def load_model(self, TheModelClass, model_path, entire_model=True, *args, **kwargs):
        """
        function used to load model(state_dict mode just complete and without testing)
        Notice that the load_state_dict() function takes a dictionary object, NOT a path to a saved object.
        This means that you must deserialize the saved state_dict before you pass it to the load_state_dict() function.
        For example, you CANNOT load using model.load_state_dict(PATH).

        :param TheModelClass:
        :param model_path:
        :param entire_model:
        :param args:
        :param kwargs:
        :return:
        """
        if entire_model:
            # Load Entire Model
            model = torch.load(model_path)
            self.model = model
        else:
            # for inference
            model = TheModelClass(*args, **kwargs)
            model.load_state_dict(torch.load(model_path))
            self.model = model


""""
if __name__ == "__main__":
    model = CreateModel()
    print(model)
"""
