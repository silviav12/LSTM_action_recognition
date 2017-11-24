# Generate csv files with inception features from video files using model available from:
# https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py

#classify_image from https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py
import classify_image
import cv2
import numpy as np
import subprocess as sp
import glob
import csv
from skimage.feature import hog
import os.path
from skimage import data, color, exposure
import sys

def get_framedims(video):
    """Find video dimensions to extract frames using ffmprobe
    """
    print(video)
    FFMPROBE_BIN = "ffprobe" 
    command = [ FFMPROBE_BIN,
            '-v', 'error',
            '-show_entries', 'stream=width, height',
            '-of', 'default=noprint_wrappers=1', video]
    pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
    dims = pipe.stdout.read(20)
    dstring = str(dims, 'utf-8')
    A = dstring.split('width=')
    B = A[1].split("\n")
    height = int((B[1].split('='))[1])
    width = int(B[0])
    pipe.stdout.flush()
    return height, width

def get_video(video):
    """ Open video using ffmpeg
    """
    FFMPEG_BIN = "ffmpeg" # on Linux ans Mac OS
    command = [ FFMPEG_BIN,
            '-i', video,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
    pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
    return pipe

def read_frames(pipe, height, width):
    """Given video dimensions, read video frames and return as array
    """
    raw_image = pipe.stdout.read(height*width*3)
    # transform the byte read into a numpy array
    image1 =  np.fromstring(raw_image, dtype='uint8')
    if(image1.shape[0]!=height*width*3):
        return False, image1
    image1 = image1.reshape((height,width,3))
    # throw away the data in the pipe's buffer.
    pipe.stdout.flush()
    return True, image1


def get_features(folder):
    """Function to set up and extract features using Inception
    
    Parameters
    ----------
    folder : str
        absolute path in which to get the date and save the features
    """
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    fnames = glob.glob(folder + "/THETIS/VIDEO_RGB/*")
    dir_csv = folder + "/THETIS/THETIS_hog/"
    for i in fnames[a:b]:
        vnames = glob.glob(i + "/*.avi")
        for v in vnames:
            direc_ = v.split('/')[-2]
            v1 = v.split('/')[-1]
            my_file_name = v1.split('.avi')[0]+".csv"
            if not os.path.exists(dir_csv + direc_ + "/"):
                 os.makedirs(dir_csv + direc_ + "/")
            csv_path = dir_csv + direc_ + "/" + my_file_name
            if not(os.path.isfile(csv_path)):
               print(v)
               p = get_video(v)
               h, w = get_framedims(v)
               val = True
               imList = []
               while(val==True):
		    # read frames
                    val, im = read_frames(p, h, w)
                    if(val==True):
                        imList.append(im)

               # Run inference from tensorflow code to get inception features
               one = classify_image.run_inference_on_frame(imList)
               myfile = open(csv_path, "w")
               print(myfile)
               # store as csv with same name as video and frame number 
               # (video name contains label)
               wr = csv.writer(myfile)
               for j in one:    
                    wr.writerow(j)

if __name__ == '__main__':
    get_features()

