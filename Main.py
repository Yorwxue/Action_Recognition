import os
import torch
import glob
from torch.utils.data.dataloader import DataLoader

from Dataset import UCF101, kinetics, data_collate
from Model import CreateModel

print("sever start")
model_path = os.path.abspath("./checkpoint/InceptionV3_kinetics_Spatial.pt")
print("model path: %s" % model_path)

print("dataset: kinetics-600")


class Collated_kinetics(kinetics):
    """
    using to skip the broken data
    more detail can be find in the following url:
    https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806/4
    """
    __init__ = kinetics.__init__

    def __getitem__(self, index):
        try:
            return super(Collated_kinetics, self).__getitem__(index)
        except Exception as e:
            pass


kinetics_dataset = Collated_kinetics(sample_num=1)

batch_size = 64
num_workers = 0

use_cuda = True
if use_cuda:
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

device = torch.device("cuda" if use_cuda else "cpu")

print("---------- create model -----------")
model = CreateModel(device, in_channels=3, num_labels=600, pretrain_path=None)  # pretrain_path="./checkpoint/InceptionV3_UCF_Spatial.pt"
#if use_cuda:
#    model.model = torch.nn.DataParallel(model.model, device_ids=[2, 3])

print("---------- train -------------")
dataloader = DataLoader(kinetics_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=data_collate)
model.train(dataloader, device, num_epoch=100, display_freq=500, model_path=model_path)

# test
print("---------- test ------------")
kinetics_dataset.training(False)

dataloader = DataLoader(kinetics_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=data_collate)

model.load_model(CreateModel, model_path=model_path)
model.test(dataloader, device)

pass
