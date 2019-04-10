import random, time, ffmpeg
import numpy as np

def get_data_dir(filename):
    dir_videos, label_videos = [], []
    with open(filename, 'r') as input_file:
        for line in input_file:
            file_name, label = line.split(' ')
            dir_videos.append(file_name)
            label_videos.append(int(label))
    return dir_videos, label_videos


def shuffle_list(dir_videos, label_videos, seed=time.time()):
    video_indices = list(range(len(dir_videos)))
    random.seed(seed)
    random.shuffle(video_indices)
    shuffled_video_dirs = [dir_videos[i] for i in video_indices]
    shuffled_labels = [label_videos[i] for i in video_indices]
    return shuffled_video_dirs, shuffled_labels


def read_clip(dirname, model_settings):
    # Method to get frames from video
    def get_frames_data(file_path):
        ff = ffmpeg.input(file_path).output('pipe:', format='rawvideo', pix_fmt='rgb24')
        out, err = ff.run(capture_stdout=True)
        video = np.frombuffer(out, np.uint8)
        return video

    num_frames_per_clip = model_settings['frames_per_clip']
    crop_size = model_settings['crop_size']
    np_mean = model_settings['np_mean']
    tmp_data = get_frames_data(dirname)

    if(len(tmp_data) == 0):
        return np.array([])
    img_datas=[]
    horizontal_flip = random.random()
    for j in range(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if(img.width>img.height):
            scale = float(crop_size)/float(img.height)
            img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
            scale = float(crop_size)/float(img.width)
            img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size)/2)
        crop_y = int((img.shape[1] - crop_size)/2)
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
        
        #Flip the image 0.5 chance
        if(horizontal_flip > 0.5):
            img = np.fliplr(img)
            
        img_datas.append(img)
    return np.array(img_datas).astype(np.float32)
    