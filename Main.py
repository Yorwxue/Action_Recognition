import os
import torch
from torch.utils.data import DataLoader
import glob

from Dataset import UCF101, kinetics, data_collate
from Model import CreateModel

print("sever start")
model_path = os.path.abspath("./checkpoint/InceptionV3_kinetics_Spatial.pt")
print("model path: %s" % model_path)

print("dataset: kinetics-600")


class Collated_kinetics(kinetics, sample_num=1):
    __init__ = kinetics.__init__

    def __getitem__(self, index):
        try:
            return super(Collated_kinetics, self).__getitem__(index)
        except Exception as e:
            label = self.train_list[index][0]
            youtube_id = self.train_list[index][1]
            video_path = glob.glob("%s_*" % os.path.join(self.video_dir, "train", label, youtube_id))[0]  # glob should return only one result, due to youtube id is unique.
            print(e)
            print("label: %s" % label)
            print("youtube_id: %s" % youtube_id)
            print("video_path: %s" % video_path)


kinetics_dataset = Collated_kinetics(sample_num=1)

use_cuda = True
if use_cuda:
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
device = torch.device("cuda" if use_cuda else "cpu")

print("---------- create model -----------")
model = CreateModel(device, in_channels=3, num_labels=600)

print("---------- train -------------")
dataloader = DataLoader(kinetics_dataset, batch_size=64, shuffle=True, num_workers=30, collate_fn=data_collate)
model.train(dataloader, device, num_epoch=100, display_freq=500, model_path=model_path)

# test
print("---------- test ------------")
kinetics_dataset.training(False)

dataloader = DataLoader(kinetics_dataset, batch_size=64, shuffle=False, num_workers=30, collate_fn=data_collate)

model.load_model(CreateModel, model_path=model_path)
model.test(dataloader, device)

pass
