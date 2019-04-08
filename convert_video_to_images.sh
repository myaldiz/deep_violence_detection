# convert the avi video to images
#   Usage (sudo for the remove priviledge):
#       sudo ./convert_video_to_images.sh path/to/video fps
#   Example Usage:
#       sudo ./convert_video_to_images.sh ~/document/videofile/ 5
#   Example Output:
#       ~/document/videofile/walk/video1.avi 
#       #=>
#       ~/document/videofile/walk/video1/00001.jpg
#       ~/document/videofile/walk/video1/00002.jpg
#       ~/document/videofile/walk/video1/00003.jpg
#       ~/document/videofile/walk/video1/00004.jpg
#       ~/document/videofile/walk/video1/00005.jpg
#       ...

#!/bin/bash
time0=`date +%s%3N`
for folder in "$1"
do
    for file in "$folder"
    do
        mkdir -p "${file[@]}_frames"
        ffmpeg -i "$file" -r "$2" -f image2 "${file[@]}_frames"/%05d.png
    done
done
time1=`date +%s%3N`
dur=`expr $time1 - $time0`
echo $dur ms