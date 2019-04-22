import random, time, ffmpeg
import os
import numpy as np
import tensorflow as tf


# Reads train/test filenames from provided splits
# Returns video directions and their labels in a list
def get_data_dir(filename):
    dir_videos, label_videos = [], []
    with open(filename, 'r') as input_file:
        for line in input_file:
            file_name, label = line.split(' ')
            dir_videos.append(file_name)
            label_videos.append(int(label) - 1)
    return dir_videos, label_videos


# Shuffles video directions along with labels
def shuffle_list(dir_videos, label_videos, seed=time.time()):
    print('Shuffling the dataset...')
    video_indices = list(range(len(dir_videos)))
    random.seed(seed)
    random.shuffle(video_indices)
    shuffled_video_dirs = [dir_videos[i] for i in video_indices]
    shuffled_labels = [label_videos[i] for i in video_indices]
    return shuffled_video_dirs, shuffled_labels


# Given video directory it reads the video
# extracts the frames, and do pre-processing operation
def read_clips_from_video(dirname, model_settings):
    # Input size for the network
    frames_per_batch = model_settings['frames_per_batch']
    video_fps = model_settings['video_fps']
    crop_size = model_settings['crop_size']
    np_mean = model_settings['np_mean']
    trans_max = model_settings['trans_max']

    # Data augmentation randoms
    horizontal_flip = random.random()
    trans_factor = random.randint(-trans_max, trans_max)

    # Video information
    probe = ffmpeg.probe(dirname)
    video_info = probe["streams"][0]
    video_width = video_info["width"]
    video_height = video_info["height"]
    video_duration = float(video_info["duration"])
    num_frame = int(video_info["nb_frames"])

    # Select which portion of the video will be input
    rand_max = int(num_frame - ((num_frame / video_duration) * (frames_per_batch / video_fps)))

    start_frame = random.randint(0, max(rand_max - 1, 0))
    # end_frame = ceil(start_frame + (num_frame / video_duration) * frames_per_batch / video_fps + 1)
    video_start = (video_duration / num_frame) * start_frame
    video_end = min(video_duration, video_start + ((frames_per_batch + 1) / video_fps))

    # Cropping factor
    x_pos = max(video_width - video_height + 2 * trans_factor, 0) // 2
    y_pos = max(video_height - video_width + 2 * trans_factor, 0) // 2
    crop_size1 = min(video_height, video_width)
    # Read specified times of the video
    ff = ffmpeg.input(dirname, ss=video_start, t=video_end - video_start)
    # Trim video -> did not work :(
    # ff = ff.trim(end_frame='50')
    # Divide into frames
    ff = ffmpeg.filter(ff, 'fps', video_fps)
    # Crop
    ff = ffmpeg.crop(ff, x_pos, y_pos, crop_size1, crop_size1)
    # Subsample
    ff = ffmpeg.filter(ff, 'scale', crop_size, crop_size)
    # Horizontal flip with some probability
    if horizontal_flip > 0.5:
        ff = ffmpeg.hflip(ff)
    # Output the video
    ff = ffmpeg.output(ff, 'pipe:',
                       format='rawvideo',
                       pix_fmt='rgb24')
    # Run Process in quiet mode
    out, _ = ffmpeg.run(ff, capture_stdout=True, quiet=True)
    # Extract to numpy array
    video = np.frombuffer(out, np.uint8). \
        reshape([-1, crop_size, crop_size, 3])

    # Copies last frame if # of frames < 16
    # Subtracts the mean and converts type to float32
    num_frames = video.shape[0]
    if num_frames < frames_per_batch:
        last_frame = video[-1]
        num_frame_repeat = frames_per_batch - num_frames
        print('Frames repeated: ', num_frame_repeat)
        last_repeat = np.repeat(last_frame[np.newaxis],
                                num_frame_repeat,
                                axis=0)
        video = np.concatenate((video, last_repeat), axis=0) - np_mean
    else:
        video = video[:frames_per_batch] - np_mean

    return video


def get_frames_data(filename, start_index, num_frames_per_clip=16):
    ret_arr = []
    for parent, dirnames, filenames in os.walk(filename):
        if len(filenames) < num_frames_per_clip:
            return np.array([])
        filenames = sorted(filenames)
        for i in range(start_index - 1, start_index + num_frames_per_clip - 1):
            image_name = str(filename) + '/' + str(filenames[i])
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)
    return ret_arr


def read_clips_from_frames(dirname, model_settings, sess):
    pass
