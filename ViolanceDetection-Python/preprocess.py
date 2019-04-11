import random, time, ffmpeg
import numpy as np

# Reads train/test filenames from provided splits
# Returns video directions and their labels in a list
def get_data_dir(filename):
    dir_videos, label_videos = [], []
    with open(filename, 'r') as input_file:
        for line in input_file:
            file_name, label = line.split(' ')
            dir_videos.append(file_name)
            label_videos.append(int(label))
    return dir_videos, label_videos


# Shuffles video directions along with labels
def shuffle_list(dir_videos, label_videos, seed=time.time()):
    video_indices = list(range(len(dir_videos)))
    random.seed(seed)
    random.shuffle(video_indices)
    shuffled_video_dirs = [dir_videos[i] for i in video_indices]
    shuffled_labels = [label_videos[i] for i in video_indices]
    return shuffled_video_dirs, shuffled_labels


# Given video directory it reads the video
# extracts the frames, and do preprocessing operation
def read_clip(dirname, model_settings):
    # Method to get frames from video

    num_frames_per_clip = model_settings['frames_per_clip']
    crop_size = model_settings['crop_size']
    np_mean = model_settings['np_mean']
    fps = model_settings['frames_per_clip']
    horizontal_flip = random.random()

    probe = ffmpeg.probe(dirname)
    video_info = probe["streams"][0]
    video_width = int(video_info["width"])
    video_height = int(video_info["height"])

    # Input video
    ff = ffmpeg.input(dirname)
    # Divide into frames
    ff = ffmpeg.filter(ff, 'fps', fps)
    # Crop
    x_pos = max(video_width - video_height, 0) // 2
    y_pos = max(video_height - video_width, 0) // 2
    crop_size1 = min(video_height, video_width)
    ff = ffmpeg.crop(ff, x_pos, y_pos, crop_size1, crop_size1)
    # Subsample
    ff = ffmpeg.filter(ff, 'scale', crop_size, crop_size)
    if horizontal_flip > 0.5:
        ff = ffmpeg.hflip(ff)
    ff = ffmpeg.output(ff, 'pipe:',
                       format='rawvideo',
                       pix_fmt='rgb24')
    out, err = ff.run(capture_stdout=True)
    video = np.frombuffer(out, np.uint8).\
        reshape([-1, crop_size, crop_size, 3])
    video = video - np_mean

    return video



    # # TODO: Check and correct this and following part
    # if (len(tmp_data) == 0):
    #     return np.array([])
    # img_datas = []
    #
    # for j in range(len(tmp_data)):
    #     img = Image.fromarray(tmp_data[j].astype(np.uint8))
    #     if (img.width > img.height):
    #         scale = float(crop_size) / float(img.height)
    #         img = np.array(cv2.resize(np.array(img),
    #                                   (int(img.width * scale + 1),
    #                                    crop_size))).astype(np.float32)
    #     else:
    #         scale = float(crop_size) / float(img.width)
    #         img = np.array(cv2.resize(np.array(img),
    #                                   (crop_size,
    #                                    int(img.height * scale + 1)))).astype(np.float32)
    #     crop_x = int((img.shape[0] - crop_size) / 2)
    #     crop_y = int((img.shape[1] - crop_size) / 2)
    #     img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :] - np_mean[j]
    #
    #
    #     img_datas.append(img)
    # return np.array(img_datas).astype(np.float32)
