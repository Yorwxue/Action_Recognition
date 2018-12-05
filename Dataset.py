import os
import patoolib
import numpy as np
import random
import cv2
from skimage import io, transform


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class UCF101(Dataset):
    def __init__(self, training=False, cross_valid="01", download=False):
        super(UCF101, self).__init__()

        self.data_dir = os.path.abspath("dataset")
        self.video_dir = os.path.abspath(os.path.join("dataset", "UCF-101"))
        self.img_rows = 224
        self.img_cols = 224
        self.num_label = 101
        self.sample_num = 10
        self.random_sample_key = 666
        # self.transform = transform

        if download:
            self.download()

        with open(os.path.join(self.data_dir, "ucfTrainTestlist", "trainlist%s.txt" % cross_valid), 'r') as fr:
            self.train_list = fr.readlines()
        if training:
            with open(os.path.join(self.data_dir, "ucfTrainTestlist", "testlist%s.txt" % cross_valid), 'r') as fr:
                self.test_list = fr.readlines()

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        video_path = os.path.join(self.video_dir, self.train_list[index].split(' ')[0])
        label = int(self.train_list[index].split(' ')[-1]) - 1  # start from 0

        # one-hot encoding
        # onehot_label = np.zeros(self.num_label)
        # onehot_label[label] = 1
        # label = {
        #     "class-id": label,
        #     "one-hot": onehot_label
        # }

        sample = {'input': self.get_input_data(video_path), 'label': label}

        return sample

    def get_input_data(self, video_path):
        """
        doing same preprocessing, like optical flow, and transform raw data into pytorch format
        ex:
        For a conv2D in pytorch, input should be in (N, C, H, W) format. N is the number of samples/batch_size. C is the channels. H and W are height and width resp.
        :param video_path:
        :return:
        """
        raw_frames = get_frames(video_path, self.img_rows, self.img_cols)
        optical_frames = create_optical_flow(raw_frames)

        # if self.transform:
        #     sample = self.transform(sample)
        optical_frames["orig"] = [np.transpose(optical_frame, (2, 0, 1)) for optical_frame in optical_frames["orig"]]

        # random sample
        for keys in optical_frames.keys():
            optical_frames[keys] = random_sample(optical_frames[keys], N=self.sample_num, seed=self.random_sample_key)
            optical_frames[keys] = un_roll_timestep(optical_frames[keys]) --------------------------------------------------------------non-implemented

        input_data = optical_frames

        return input_data

    def download(self):
        if not os.path.exists("dataset"):
            os.makedirs("dataset")
        # get video
        os.system("wget http://crcv.ucf.edu/data/UCF101/UCF101.rar")
        os.system("mv UCF101.rar %s" % self.data_dir)
        self.unzip(os.path.abspath("dataset/UCF101.rar"), self.data_dir)

        # get train/test Splits
        os.system("wget http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip")
        os.system("mv UCF101TrainTestSplits-RecognitionTask.zip %s" % self.data_dir)
        self.unzip(os.path.join(self.data_dir, "UCF101TrainTestSplits-RecognitionTask.zip"), self.data_dir)

    def unzip(self, filepath, outdir='.'):
        print(filepath)
        print(outdir)
        patoolib.extract_archive(filepath, outdir=outdir)


def get_frames(video_path, resize_img_rows, resize_img_cols):
    """
    transform video to series of frames
    :param video_path:
    :return: a series of images
    """
    frames = list()

    # Create a VideoCapture object and read from input file
    # If the input is taken from the camera, pass 0 instead of the video file name.
    cap = cv2.VideoCapture(video_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        exit()

    # more detail of propId can be find in https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
    # propId 7: Number of frames in the video file
    # nb_frame = cap.get(propId=7)
    # moving the "frame reader" to the offset of the specific frame
    # cap.set(1, specific_no)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:  # end/pause of this video
            # exit()
            break

        frame = cv2.resize(frame, (resize_img_rows, resize_img_cols))

        frames.append(frame)

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    return frames


def random_sample(frames, N=10, seed=None):
    """
    randomly choose several frames to be the representation of this video
    :param frames:
    :param N: Randomly choose N frames to be the representation of this video
    :return: the representation of this video
    """

    nb_frame = len(frames)

    # randomly choose N frames to be the representation of this video
    N_list = np.arange(nb_frame)
    if seed:
        random.seed(seed)
    random.shuffle(N_list)
    N_list = sorted(N_list[: N])

    sample_video = [frames[no] for no in N_list]

    return sample_video


def create_optical_flow(raw_frames, freq_of_motion=1):
    """
    calculate the optical flow of frames
    Note:
        For a conv2D in pytorch, input should be in (N, C, H, W) format. N is the number of samples/batch_size.
        C is the channels. H and W are height and width resp.
    :param raw_frames: series of frames of one video
    :param img_rows: high of frame
    :param img_cols: width of frame
    :param freq_of_motion: how many frames to calculate optical flow once
    :return:
    """
    frame_set = dict()
    frame_set['orig'] = list()
    frame_set['gray'] = list()
    frame_set['flow'] = list()
    frame_set['hori'] = list()
    frame_set['vert'] = list()

    for idx, frame in enumerate(raw_frames):
        frame_set['orig'].append(frame)
        frame_set['gray'].append(cv2.cvtColor(frame_set['orig'][idx], cv2.COLOR_BGR2GRAY))

        # calculate motion of two frames
        frame_set['flow'].append(cv2.calcOpticalFlowFarneback(frame_set['gray'][idx-freq_of_motion],
                                                              frame_set['gray'][idx],
                                                              None, 0.5, 3, 15, 3, 5, 1.2, 0))
        # horizontal & vertical
        # frame_set['hori'].append(frame_set['flow'][-1][..., 0])
        frame_set['hori'].append(frame_set['flow'][-1][..., 0] -
                                 cv2.mean(frame_set['flow'][-1][..., 0])[0] * np.ones(frame_set['flow'][-1][..., 0].shape))
        # frame_set['vert'].append(frame_set['flow'][-1][..., 1])
        frame_set['vert'].append(frame_set['flow'][-1][..., 1] -
                                 cv2.mean(frame_set['flow'][-1][..., 1])[0] * np.ones(frame_set['flow'][-1][..., 1].shape))

        # change range to 0~255
        frame_set['hori'][-1] = cv2.normalize(frame_set['hori'][-1], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        frame_set['vert'][-1] = cv2.normalize(frame_set['vert'][-1], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    return frame_set


def draw_flow(img, flow, step=16):
    """
    https://github.com/carlosgregoriorodriguez/OpenCV/blob/master/motion/farneback.py
    :param img:
    :param flow:
    :param step: the initial interval is 16
    :return:
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)  # using grid form to select equal interval of 2D image, and reshape to array with 2 rows
    fx, fy = flow[y, x].T  # get the optical flow displacement of the correspond of selected coordinate point
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)  # 将初始點和變化的點堆疊成2*2的數組
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))  # draw a line linking the start point and end point to present the optical flow
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)  # draw a circle to present the start point of all grid point
    return vis


"""
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # video_path = os.path.join(os.path.abspath("dataset"), "UCF-101", "Archery", "v_Archery_g01_c01.avi")
    # raw_frames = get_frames(video_path)
    # optical_frames = create_optical_flow(raw_frames, 256, 256)
    #
    # pass
    #
    # frames = optical_frames
    # for idx, frame_flow in enumerate(frames['flow']):
    #     cv2.imshow('frame', draw_flow(frames['gray'][idx], frame_flow))
    #     cv2.waitKey(100)

    ucf101_dataset = UCF101()
    dataloader = DataLoader(ucf101_dataset, batch_size=2, shuffle=True, num_workers=2)
    for batch_idx, sample_batched in enumerate(dataloader):
        print(batch_idx, sample_batched['input'].size(), sample_batched['label'].size())
"""
