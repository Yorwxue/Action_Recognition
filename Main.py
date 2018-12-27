import os
import torch
import glob
from torch.utils.data.dataloader import DataLoader

from Dataset import UCF101, kinetics, data_collate
from Model import CreateModel

print("sever start")
model_path = os.path.abspath("./checkpoint/InceptionV3_UCF_Spatial.pt")
# model_path = os.path.abspath("./checkpoint/InceptionV3_kinetics_Spatial.pt")
print("model path: %s" % model_path)

print("dataset: ucf-101")
# print("dataset: kinetics-600")


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
            # print("Exception: Collating Fail")
            # print("--------------------------")
            # print(e)
            # label = self.train_list[index][0]
            # print("label: %s" % label)
            # youtube_id = self.train_list[index][1]
            # print("youtube_id: %s" % youtube_id)
            # print("--------------------------\n")
            pass


ucf_dataset = UCF101(sample_num=1)
# kinetics_dataset = Collated_kinetics(sample_num=1)

batch_size = 4
num_workers = 5

use_cuda = True
if use_cuda:
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

device = torch.device("cuda" if use_cuda else "cpu")

print("---------- create model -----------")
model = CreateModel(device, in_channels=3, num_labels=101, pretrain_path=None)  # pretrain_path="./checkpoint/InceptionV3_UCF_Spatial.pt"
#if use_cuda:
#    model.model = torch.nn.DataParallel(model.model, device_ids=[2, 3])

print("---------- train -------------")
dataloader = DataLoader(ucf_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=data_collate)
model.train(dataloader, device, num_epoch=100, display_freq=500, model_path=model_path)

# test
print("---------- test ------------")
ucf_dataset.training(False)

dataloader = DataLoader(ucf_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=data_collate)

model.load_model(CreateModel, model_path=model_path)
model.test(dataloader, device)

pass
