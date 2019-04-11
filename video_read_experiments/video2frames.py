import sys
pwd = sys.path[0] + '/'
import os
import shutil
import time
import numpy as np
import matplotlib.image as mpimg
from ffmpy import FFmpeg
import ffmpeg

fps_out = 25
input = pwd+'video.avi'				# Enter filename here
out_dir = pwd+'video_frames'
str = '%s/%%05d.png'%out_dir

try:
	os.mkdir(out_dir)
except(FileExistsError):
	shutil.rmtree(out_dir)
	os.mkdir(out_dir)

ff = FFmpeg(
	inputs={input: None},
	outputs={str: ['-r','%d'%fps_out,'-f','image2']}
)
ff.run()
files = os.listdir(out_dir)
time0 = time.time()
for filename in files:
	img = mpimg.imread(out_dir+'/'+filename)
time1 = time.time()
read_time = time1-time0	

time0 = time.time()
probe = ffmpeg.probe(input)
video_info = probe["streams"][0]
width = video_info["width"]
height = video_info["height"]
duration = video_info["duration"]
ff = ffmpeg.input(input).output('pipe:', format='rawvideo', pix_fmt='rgb24')
out,err = ff.run(capture_stdout=True)
video = np.frombuffer(out,np.uint8).reshape([-1,height,width,3])
frames_in = video.shape[0]
frames_out = int(float(duration)*fps_out)
index = np.linspace(0,frames_in-1, num=frames_out).astype(int)
video = video[index,:,:,:]
time1 = time.time()
conv_time = time1-time0	

print("\n")
print("Time to read from png = %f sec"%read_time)
print("Time to convert on-the-fly = %f sec"%conv_time)