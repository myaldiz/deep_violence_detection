import random, time, ffmpeg
import os
import numpy as np
import tensorflow as tf
import PIL.Image as Image


# Reads train/test filenames from provided splits
# Returns video directions and their labels in a list
def get_data_dir(filename, model_settings):
    dir_videos, label_videos = [], []
    with open(filename, 'r') as input_file:
        for line in input_file:
            file_name, label = line.split(' ')
            # if will read from frames
            if model_settings['read_from_frames']:
                file_name = '.'.join(file_name.split('.')[:-1])
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
        # print('Frames repeated: ', num_frame_repeat)
        last_repeat = np.repeat(last_frame[np.newaxis],
                                num_frame_repeat,
                                axis=0)
        video = np.concatenate((video, last_repeat), axis=0) - np_mean
    else:
        video = video[:frames_per_batch] - np_mean

    return video


def get_frames_data(filename, model_settings):
    frames_per_batch = model_settings['frames_per_batch']
    ret_arr = []
    for parent, dirnames, filenames in os.walk(filename):
        num_frames = len(filenames)
        start_max = max(0, num_frames - frames_per_batch)
        start_index = random.randint(0, start_max)
        end_index = min(start_index + frames_per_batch, num_frames)
        filenames = sorted(filenames)
        for i in range(start_index, end_index):
            image_name = str(filename) + '/' + str(filenames[i])
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)
    return ret_arr


def process_frames(model_settings):
    with tf.name_scope('Frame_Process'), tf.device('/cpu:0'):
        # Get the variables
        images_placeholder = model_settings['images_placeholder']
        trans_max = model_settings['trans_max']
        crop_size = model_settings['crop_size']
        frames_per_batch = model_settings['frames_per_batch']
        np_mean = tf.convert_to_tensor(model_settings['np_mean'])

        # Shape setting
        clips_shape = tf.shape(images_placeholder)
        video_width = clips_shape[1]
        video_height = clips_shape[2]

        # Calculate remaining frames
        rem_frame = frames_per_batch - clips_shape[0]

        # Translate images by factor
        if model_settings['is_testing']:
            trans_factor = tf.convert_to_tensor([0])
        else:
            trans_factor = tf.random.uniform([1], -trans_max, trans_max, dtype=tf.int32)

        # Crop pos calculation
        crop_size1 = tf.math.minimum(video_height, video_width)
        x_pos = tf.math.maximum(video_width - video_height + 2 * trans_factor, 0) // 2
        x_start, x_end = x_pos[0], x_pos[0] + crop_size1
        y_pos = tf.math.maximum(video_height - video_width + 2 * trans_factor, 0) // 2
        y_start, y_end = y_pos[0], y_pos[0] + crop_size1

        # Crop the images
        clips_cropped = images_placeholder[:, x_start:x_end, y_start:y_end]

        # Interpolate
        clips_interp = tf.image.resize_bicubic(clips_cropped, (crop_size, crop_size))
        clips_interp = tf.clip_by_value(clips_interp, 0, 255)
        last_frame = clips_interp[-1]

        # Tile the remaining frames for final 16 frames
        rem_frames = tf.tile(tf.expand_dims(last_frame, 0), [rem_frame, 1, 1, 1])
        final_clips = tf.concat([clips_interp, rem_frames], 0)
        # Flip by a chance
        final_clips = tf.image.random_flip_left_right(final_clips)
        # Subtract the mean
        final_clips -= np_mean
    return final_clips
