import os
import glob
import patoolib
import numpy as np
import random
import cv2
import warnings
from torch.utils.data.dataloader import default_collate

class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class UCF101(Dataset):
    def __init__(self, training=True, cross_valid="01", sample_num=10, download=False, img_rows=299, img_cols=299):
        """

        :param training:
        :param cross_valid:
        :param sample_num:
        :param download:
        :param img_rows: for VGG, image size should be 224*224, but for inception, it should be 299*299
        :param img_cols: as aforementioned
        """
        super(UCF101, self).__init__()

        self.data_dir = os.path.abspath(os.path.join("dataset", "UCF101"))
        self.video_dir = os.path.abspath(os.path.join("dataset", "UCF101", "UCF-101"))

        self.train = training

        # for VGG, image size should be 224*224, but for inception, it should be 299*299
        self.img_rows = img_rows
        self.img_cols = img_cols

        self.num_label = 101
        self.sample_num = sample_num
        self.random_sample_key = 666
        # self.transform = transform

        if download:
            self.download()

        with open(os.path.join(self.data_dir, "ucfTrainTestlist", "trainlist%s.txt" % cross_valid), 'r') as fr:
            self.train_list = fr.readlines()

        with open(os.path.join(self.data_dir, "ucfTrainTestlist", "testlist%s.txt" % cross_valid), 'r') as fr:
            self.test_list = fr.readlines()

        with open(os.path.join(self.data_dir, "ucfTrainTestlist", "classInd.txt"), 'r') as fr:
            classInd_txt = fr.readlines()
            self.classInd = dict()
            for line in classInd_txt:
                class_id, class_name = line.replace('\n', '').split(' ')
                self.classInd[class_name] = class_id
                self.classInd[class_id] = class_name

    def __len__(self):
        if self.train:
            return len(self.train_list)
        else:
            return len(self.test_list)

    def __getitem__(self, index):
        if self.train:
            video_path = os.path.join(self.video_dir, self.train_list[index].split(' ')[0])
            label = int(self.train_list[index].split(' ')[-1]) - 1  # start from 0

            sample = {'input': self.get_input_data(video_path), 'label': label}
        else:
            video_path = os.path.join(self.video_dir, self.test_list[index].replace('\n', ''))
            class_name = self.test_list[index].split('/')[0]
            label = int(self.classInd[class_name]) - 1  # start from 0

            sample = {'input': self.get_input_data(video_path), 'label': label}

        return sample

    def training(self, training):
        self.train = training

    def get_input_data(self, video_path):
        """
        doing same preprocessing, like optical flow, and transform raw data into pytorch format
        ex:
        For a conv2D in pytorch, input should be in (N, C, H, W) format. N is the number of samples/batch_size. C is the channels. H and W are height and width resp.
        :param video_path:
        :return:
        """
        raw_frames = get_frames(video_path, self.img_rows, self.img_cols)

        # video
        # """
        # Randomly cut a episode from video
        raw_frames = random_cut(raw_frames)

        # optical flow
        optical_frames = create_optical_flow(raw_frames)

        # random sample
        for keys in optical_frames.keys():
            # list to ndarray
            optical_frames[keys] = np.asarray(optical_frames[keys])

        # get the middle frame
        optical_frames["orig"] = optical_frames["orig"][int(optical_frames["orig"].shape[0] / 2)]

        # un-rolled time steps
        optical_frames["flow"] = self.un_rolled_timestep(optical_frames["flow"])

        # channel last-> channel first
        optical_frames["orig"] = np.transpose(optical_frames["orig"], (2, 0, 1))
        optical_frames["flow"] = np.transpose(optical_frames["flow"], (2, 0, 1))
        input_data = {
            "Spatial": optical_frames["orig"],
            "Temporal": optical_frames["flow"]
        }
        # """

        # image
        """
        # random choose one frame from video
        frames = random_sample(raw_frames, N=1, seed=self.random_sample_key)[0]

        # channel last-> channel first
        frames = np.transpose(frames, (2, 0, 1))

        input_data = normalize(frames)
        # """

        return input_data

    def download(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        # get video
        os.system("wget http://crcv.ucf.edu/data/UCF101/UCF101.rar")
        os.system("mv UCF101.rar %s" % self.data_dir)
        self.unzip(os.path.abspath(os.path.join(self.data_dir, "UCF101.rar")), self.data_dir)

        # get train/test Splits
        os.system("wget http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip")
        os.system("mv UCF101TrainTestSplits-RecognitionTask.zip %s" % self.data_dir)
        self.unzip(os.path.join(self.data_dir, "UCF101TrainTestSplits-RecognitionTask.zip"), self.data_dir)

    def unzip(self, filepath, outdir='.'):
        print(filepath)
        print(outdir)
        patoolib.extract_archive(filepath, outdir=outdir)

    def un_rolled_timestep(self, frames):
        """
        spread time steps as channels
        :param frames:
        :return:
        """
        assert frames.ndim in [3, 4]

        if frames.ndim == 3:
            frames = np.reshape(frames, (frames.shape[0], frames.shape[1], frames.shape[2], 1))
        frames = np.transpose(frames, (1, 2, 3, 0))
        un_rolled_set = np.reshape(
            frames,
            (self.img_rows, self.img_cols, frames.shape[-2]*frames.shape[-1])
        )
        return un_rolled_set


class kinetics(Dataset):
    def __init__(self, training=True, sample_num=10, download=False, num_jobs=40, img_rows=299, img_cols=299):
        """
        
        :param training:
        :param cross_valid:
        :param sample_num:
        :param download:
        :param num_jobs: number of threads to download data
        :param img_rows: for VGG, image size should be 224*224, but for inception, it should be 299*299
        :param img_cols: as aforementioned
        """
        super(kinetics, self).__init__()

        self.data_dir = os.path.abspath(os.path.join("dataset", "kinetics"))
        self.video_dir = os.path.abspath(os.path.join("dataset", "kinetics", "dataset"))

        self.train = training

        # for VGG, image size should be 224*224, but for inception, it should be 299*299
        self.img_rows = img_rows
        self.img_cols = img_cols

        self.num_label = 101
        self.sample_num = sample_num
        self.random_sample_key = 666
        # self.transform = transform

        if download:
            self.download(num_jobs=num_jobs)

        # get data list
        # drop the first row which is description of each column.
        # --------------------------------------------------------------------------------------------------------------
        with open(os.path.join(self.data_dir, "kinetics-600_train.csv"), 'r') as fr:
            self.train_list = [line.split(',') for line in (fr.readlines()[1:])]
        with open(os.path.join(self.data_dir, "kinetics-600_test.csv"), 'r') as fr:
            self.test_list = [line.split(',') for line in (fr.readlines()[1:])]
        with open(os.path.join(self.data_dir, "kinetics-600_val.csv"), 'r') as fr:
            self.val_list = [line.split(',') for line in (fr.readlines()[1:])]
        # --------------------------------------------------------------------------------------------------------------
        # Note due to training dataset hasn't download, using testing data as training, and validation as testing
        # with open(os.path.join(self.data_dir, "kinetics-600_test.csv"), 'r') as fr:
        #     self.train_list = [line.split(',') for line in (fr.readlines()[1:])]
        # with open(os.path.join(self.data_dir, "kinetics-600_val.csv"), 'r') as fr:
        #     self.test_list = [line.split(',') for line in (fr.readlines()[1:])]
        # --------------------------------------------------------------------------------------------------------------

        # classes list
        classes_list = sorted(os.listdir(os.path.join(self.video_dir, "train")))
        self.classId = dict()
        for idx, class_id in enumerate(classes_list):
                self.classId[class_id] = idx
        pass

    def __len__(self):
        if self.train:
            return len(self.train_list)
        else:
            return len(self.test_list)

    def __getitem__(self, index):
        if self.train:
            label = self.train_list[index][0]
            youtube_id = self.train_list[index][1]
            video_path = glob.glob("%s_*" % os.path.join(self.video_dir, "train", label, youtube_id))[0]  # glob should return only one result, due to youtube id is unique.
            label = int(self.classId[label])

            try:
                sample = {'input': self.get_input_data(video_path), 'label': label}
            except Exception as e:
                sample = None
        else:
            label = self.test_list[index][0]
            youtube_id = self.test_list[index][1]
            video_path = glob.glob("%s_*" % os.path.join(self.video_dir, "test", label, youtube_id))[0]
            label = int(self.classId[label])

            try:
                sample = {'input': self.get_input_data(video_path), 'label': label}
            except Exception as e:
                sample = None

        return sample

    def training(self, training):
        self.train = training

    def get_input_data(self, video_path):
        """
        doing same preprocessing, like optical flow, and transform raw data into pytorch format
        ex:
        For a conv2D in pytorch, input should be in (N, C, H, W) format. N is the number of samples/batch_size. C is the channels. H and W are height and width resp.
        :param video_path:
        :return:
        """

        raw_frames = get_frames(video_path, self.img_rows, self.img_cols)

        # video
        """
        # Randomly cut a episode from video
        raw_frames = random_cut(raw_frames)

        # optical flow
        optical_frames = create_optical_flow(raw_frames)

        # random sample
        for keys in optical_frames.keys():
            # list to ndarray
            optical_frames[keys] = np.asarray(optical_frames[keys])

        # get the middle frame
        optical_frames["orig"] = optical_frames["orig"][int(optical_frames["orig"].shape[0] / 2)]

        # un-rolled time steps
        optical_frames["flow"] = self.un_rolled_timestep(optical_frames["flow"])

        # channel last-> channel first
        optical_frames["orig"] = np.transpose(optical_frames["orig"], (2, 0, 1))
        optical_frames["flow"] = np.transpose(optical_frames["flow"], (2, 0, 1))
        input_data = {
            "Spatial": optical_frames["orig"],
            "Temporal": optical_frames["flow"]
        }
        # """

        # image
        # """
        # random choose one frame from video
        frames = random_sample(raw_frames, N=self.sample_num, seed=self.random_sample_key)[0]

        # channel last-> channel first
        frames = np.transpose(frames, (2, 0, 1))

        input_data = normalize(frames)
        # """

        return input_data


    def download(self, num_jobs):
        """
        (1) function getting from https://github.com/activitynet/ActivityNet
        (2) youtube-dl and ffmpeg is necessary.
        (3) it may need to run in python2
        """
        import glob
        import json
        import shutil
        import subprocess
        import uuid
        from collections import OrderedDict

        from joblib import delayed
        from joblib import Parallel
        import pandas as pd

        def create_video_folders(dataset, output_dir, tmp_dir):
            """Creates a directory for each label name in the dataset."""
            if 'label-name' not in dataset.columns:
                this_dir = os.path.join(output_dir, 'test')
                if not os.path.exists(this_dir):
                    os.makedirs(this_dir)
                # I should return a dict but ...
                return this_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)

            label_to_dir = {}
            for label_name in dataset['label-name'].unique():
                this_dir = os.path.join(output_dir, label_name)
                if not os.path.exists(this_dir):
                    os.makedirs(this_dir)
                label_to_dir[label_name] = this_dir
            return label_to_dir

        def construct_video_filename(row, label_to_dir, trim_format='%06d'):
            """Given a dataset row, this function constructs the
               output filename for a given video.
            """
            basename = '%s_%s_%s.mp4' % (row['video-id'],
                                         trim_format % row['start-time'],
                                         trim_format % row['end-time'])
            if not isinstance(label_to_dir, dict):
                dirname = label_to_dir
            else:
                dirname = label_to_dir[row['label-name']]
            output_filename = os.path.join(dirname, basename)
            return output_filename

        def download_clip(video_identifier, output_filename,
                          start_time, end_time,
                          tmp_dir='/tmp/kinetics',
                          num_attempts=5,
                          url_base='https://www.youtube.com/watch?v='):
            """Download a video from youtube if exists and is not blocked.
            arguments:
            ---------
            video_identifier: str
                Unique YouTube video identifier (11 characters)
            output_filename: str
                File path where the video will be stored.
            start_time: float
                Indicates the begining time in seconds from where the video
                will be trimmed.
            end_time: float
                Indicates the ending time in seconds of the trimmed video.
            """
            # Defensive argument checking.
            assert isinstance(video_identifier, str), 'video_identifier must be string'
            assert isinstance(output_filename, str), 'output_filename must be string'
            assert len(video_identifier) == 11, 'video_identifier must have length 11'

            status = False
            # Construct command line for getting the direct video link.
            tmp_filename = os.path.join(tmp_dir,
                                        '%s.%%(ext)s' % uuid.uuid4())
            command = ['youtube-dl',
                       '--quiet', '--no-warnings',
                       '-f', 'mp4',
                       '-o', '"%s"' % tmp_filename,
                       '"%s"' % (url_base + video_identifier)]
            command = ' '.join(command)
            attempts = 0
            while True:
                try:
                    output = subprocess.check_output(command, shell=True,
                                                     stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as err:
                    attempts += 1
                    if attempts == num_attempts:
                        return status, err.output
                else:
                    break

            tmp_filename = glob.glob('%s*' % tmp_filename.split('.')[0])[0]
            # Construct command to trim the videos (ffmpeg required).
            command = ['ffmpeg',
                       '-i', '"%s"' % tmp_filename,
                       '-ss', str(start_time),
                       '-t', str(end_time - start_time),
                       '-c:v', 'libx264', '-c:a', 'copy',
                       '-threads', '1',
                       '-loglevel', 'panic',
                       '"%s"' % output_filename]
            command = ' '.join(command)
            try:
                output = subprocess.check_output(command, shell=True,
                                                 stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                return status, err.output

            # Check if the video was successfully saved.
            status = os.path.exists(output_filename)
            os.remove(tmp_filename)
            return status, 'Downloaded'

        def download_clip_wrapper(row, label_to_dir, trim_format, tmp_dir):
            """Wrapper for parallel processing purposes."""
            output_filename = construct_video_filename(row, label_to_dir,
                                                       trim_format)
            clip_id = os.path.basename(output_filename).split('.mp4')[0]
            if os.path.exists(output_filename):
                status = tuple([clip_id, True, 'Exists'])
                return status

            downloaded, log = download_clip(row['video-id'], output_filename,
                                            row['start-time'], row['end-time'],
                                            tmp_dir=tmp_dir)
            status = tuple([clip_id, downloaded, log])
            return status

        def parse_kinetics_annotations(input_csv, ignore_is_cc=False):
            """Returns a parsed DataFrame.
            arguments:
            ---------
            input_csv: str
                Path to CSV file containing the following columns:
                  'YouTube Identifier,Start time,End time,Class label'
            returns:
            -------
            dataset: DataFrame
                Pandas with the following columns:
                    'video-id', 'start-time', 'end-time', 'label-name'
            """
            df = pd.read_csv(input_csv)
            if 'youtube_id' in df.columns:
                columns = OrderedDict([
                    ('youtube_id', 'video-id'),
                    ('time_start', 'start-time'),
                    ('time_end', 'end-time'),
                    ('label', 'label-name')])
                df.rename(columns=columns, inplace=True)
                if ignore_is_cc:
                    df = df.loc[:, df.columns.tolist()[:-1]]
            return df

        def download_process(input_csv, output_dir,
                             trim_format='%06d', num_jobs=4, tmp_dir='/tmp/kinetics',
                             drop_duplicates=False):

            # Reading and parsing Kinetics.
            dataset = parse_kinetics_annotations(input_csv)
            # if os.path.isfile(drop_duplicates):
            #     print('Attempt to remove duplicates')
            #     old_dataset = parse_kinetics_annotations(drop_duplicates,
            #                                              ignore_is_cc=True)
            #     df = pd.concat([dataset, old_dataset], axis=0, ignore_index=True)
            #     df.drop_duplicates(inplace=True, keep=False)
            #     print(dataset.shape, old_dataset.shape)
            #     dataset = df
            #     print(dataset.shape)

            # Creates folders where videos will be saved later.
            label_to_dir = create_video_folders(dataset, output_dir, tmp_dir)

            # Download all clips.
            if num_jobs == 1:
                status_lst = []
                for i, row in dataset.iterrows():
                    status_lst.append(download_clip_wrapper(row, label_to_dir,
                                                            trim_format, tmp_dir))
            else:
                status_lst = Parallel(n_jobs=num_jobs)(delayed(download_clip_wrapper)(
                    row, label_to_dir,
                    trim_format, tmp_dir) for i, row in dataset.iterrows())

            # Clean tmp dir.
            shutil.rmtree(tmp_dir)

            # Save download report.
            with open('download_report.json', 'w') as fobj:
                fobj.write(json.dumps(status_lst))

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(os.path.join(self.video_dir, "train")):
            os.makedirs(os.path.join(self.video_dir, "train"))
        if not os.path.exists(os.path.join(self.video_dir, "test")):  # I'm  merge the test and val directory accidentially...
            os.makedirs(os.path.join(self.video_dir, "test"))
        if not os.path.exists(os.path.join(self.video_dir, "test")):
            os.makedirs(os.path.join(self.video_dir, "test"))

        # get dataset url
        os.system("wget 'https://deepmind.com/documents/193/kinetics_600_train (1).zip'")
        os.system("wget 'https://deepmind.com/documents/194/kinetics_600_val (1).zip'")
        # os.system("wget 'https://deepmind.com/documents/231/kinetics_600_holdout_test.zip'")  # this dataset doesn't be labelled
        os.system("wget 'https://deepmind.com/documents/232/kinetics_600_test (2).zip'")
        os.system("mv kinetics_* %s" % self.data_dir)
        self.unzip(os.path.join(self.data_dir, "kinetics_600_train (1).zip"), self.data_dir)
        self.unzip(os.path.join(self.data_dir, "kinetics_600_val (1).zip"), self.data_dir)
        # self.unzip(os.path.join(self.data_dir, "kinetics_600_holdout_test.zip"), self.data_dir)
        self.unzip(os.path.join(self.data_dir, "kinetics_600_test (2).zip"), self.data_dir)

        # rename
        os.system("mv kinetics_train.csv kinetics-600_train.csv")
        os.system("rm kinetics_train.json")
        os.system("mv kinetics_val.csv kinetics-600_val.csv")
        os.system("rm kinetics_val.json")
        # os.system("mv kinetics_600_holdout_test.csv kinetics-600_holdout_test.csv")
        # os.system("rm kinetics_600_holdout_test.json")
        os.system("mv kinetics_600_test.csv kinetics-600_test.csv")
        os.system("rm kinetics_600_test.json")

        print("download dataset(this may cost about 0.7~0.9TB totally)")

        decision = input("Download validation data will spent lots of time, sure? y/n")
        if decision == 'y' or decision == 'Y':
            download_process(input_csv=os.path.join(self.data_dir, "kinetics-600_val.csv"), output_dir=os.path.join(self.video_dir, "val"), num_jobs=num_jobs)

        decision = input("Download testing data will spent lots of time, sure? y/n")
        if decision == 'y' or decision == 'Y':
            download_process(input_csv=os.path.join(self.data_dir, "kinetics-600_test.csv"), output_dir=os.path.join(self.video_dir, "test"), num_jobs=num_jobs)

        decision = input("Download training data will spent lots of time, sure? y/n")
        if decision == 'y' or decision == 'Y':
            download_process(input_csv=os.path.join(self.data_dir, "kinetics-600_train.csv"), output_dir=os.path.join(self.video_dir, "train"), num_jobs=num_jobs)

    def unzip(self, filepath, outdir='.'):
        print(filepath)
        print(outdir)
        patoolib.extract_archive(filepath, outdir=outdir)

    def un_rolled_timestep(self, frames):
        """
        spread time steps as channels
        :param frames:
        :return:
        """
        assert frames.ndim in [3, 4]

        if frames.ndim == 3:
            frames = np.reshape(frames, (frames.shape[0], frames.shape[1], frames.shape[2], 1))
        frames = np.transpose(frames, (1, 2, 3, 0))
        un_rolled_set = np.reshape(
            frames,
            (self.img_rows, self.img_cols, frames.shape[-2]*frames.shape[-1])
        )
        return un_rolled_set


# def collate_func(data_path):
#     if os.path.exists(data_path):
#         cap = cv2.VideoCapture(data_path)
#         if (cap.isOpened() == False):
#             return None
#         cap.release()
#         return data_path
#     else:
#         return None


def data_collate(batch):
    """
    using to skip the broken data
    more detail can be find in the following url:
    https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806/4
    """
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def normalize(frames):
        normalized_frames = frames/255
        return normalized_frames


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
        # print("Error opening video stream or file ; %s" % video_path)
        cap.release()
        assert False

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


def get_specific_frames(video_path, resize_img_rows, resize_img_cols, specific="random", N=1):
    """
    transform video to series of frames
    :param specific:
        (1) if specific="random", randomly choose frames
        (2) specific=[1, 10, 30] to get the 1th, 10th and 30th frames. if this video ths less than 30 frames, it will
            return only 1th and 10th frames
    :param N: return how many frames. if number of specific frames less than "N", this function will randomly choose
              several frames to meeting the "num" parameter
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
    nb_frame = cap.get(propId=7)

    # check specific length
    if specific == "random":
        specific_list = random.shuffle(np.arange(nb_frame))[:N]
    else:
        # check specific no
        for idx, specific_no in enumerate(specific):
            if specific_no > nb_frame:
                np.delete(specific, idx)

        # meeting the parameter "N"
        if len(specific) < N:
            specific_list = random.shuffle(np.delete(np.arange(nb_frame), specific, None))[:N-len(specific)]
        elif len(specific) > N:
            specific_list = specific[:N]
        else:
            specific_list = specific

    for specific_no in specific_list:
        # moving the "frame reader" to the offset of the specific frame
        cap.set(1, specific_no)

        ret, frame = cap.read()
        frame = cv2.resize(frame, (resize_img_rows, resize_img_cols))
        frames.append(frame)

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    return frames


def random_cut(frames, N=10):
    """
    Randomly cut a episode from video
    :param frames:
    :param N:
    :param seed:
    :return:
    """
    nb_frame = len(frames)

    start_point = np.random.randint(0, nb_frame - N)

    return frames[start_point:(start_point+N)]


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
    # from torch.utils.data import DataLoader
    
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

    # ucf101_dataset = UCF101()
    # dataloader = DataLoader(ucf101_dataset, batch_size=2, shuffle=True, num_workers=2)
    # for batch_idx, sample_batched in enumerate(dataloader):
    #     print(batch_idx, sample_batched['input'].size(), sample_batched['label'].size())

     kinetic_dataset = kinetic()
# """
