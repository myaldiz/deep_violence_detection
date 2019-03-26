import csv

def get_UCF101_dir(filename):
    dir_images = []
    start_clip = []
    label_images = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            dir_images.append(row[0])
            start_clip.append(int(row[1]))
            label_images.append(row[2])
    return dir_images, start_clip, label_images


def shuffle_list(dir_clips, start_clips, label_clips, seed=time.time()):
    video_indices = list(range(len(dir_clips)))
    random.shuffle(video_indices)
    shuffled_clip_dirs   = [dir_clips[i] for i in video_indices]
    shuffled_starts = [start_clips[i] for i in video_indices]
    shuffled_labels = [label_clips[i] for i in video_indices]
    return shuffled_clip_dirs, shuffled_starts, shuffled_labels


def get_frames_data(filename, start_index, num_frames_per_clip=16):
    ret_arr = []
    for parent, dirnames, filenames in os.walk(filename):
        if(len(filenames) < num_frames_per_clip):
            return np.array([])
        filenames = sorted(filenames)
        for i in range(start_index - 1, start_index + num_frames_per_clip - 1):
            image_name = str(filename) + '/' + str(filenames[i])
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)
    return ret_arr


def read_clip(dirname, start_index, model_settings):
    num_frames_per_clip = model_settings['frames_per_clip']
    crop_size = model_settings['crop_size']
    np_mean = model_settings['np_mean']
    tmp_data = get_frames_data(dirname, start_index)
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
    