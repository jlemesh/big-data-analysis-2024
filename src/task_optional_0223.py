
# pip install opencv-python opencv-contrib-python numpy

import cv2 as cv
import os
import numpy
from timeit import default_timer
from multiprocessing import Process, cpu_count

# https://docs.opencv.org/4.8.0/df/d74/classcv_1_1FastFeatureDetector.html
# https://docs.opencv.org/3.4/df/d0c/tutorial_py_fast.html

output_path = 'data/Pistachio_Image_Dataset/Output/Output/'
input_path = 'data/Pistachio_Image_Dataset/Pistachio_Image_Dataset/All/' # put all images in one directory

# finds FAST keypoints in an image
def find_keypoints(img, threshold = 10, nonmaxSuppression = True, type = cv.FastFeatureDetector_TYPE_9_16):
  fast = cv.FastFeatureDetector_create(threshold, nonmaxSuppression, type)
  kp = fast.detect(img,None)

  return cv.drawKeypoints(img, kp, None, color=(255,0,0)) # returns an image with keypoints drawn on it

# finds FAST keypoints for each image in a list, writes output to a file
def process_files(input, files, output):
  for file in files:
    image = cv.imread(os.path.join(input, file)) # reads a pecified image file
    kp = find_keypoints(image)
    cv.imwrite(os.path.join(output, file), kp) # writes an image to a file

# processes each image in parallel (warning: creates ~2K processes)
def run_all_parallel(files):
  start = default_timer()
  processes = []
  
  for file in files:
    process = Process(target=process_files, args=(input_path, [file], output_path))
    process.start()
    processes.append(process)

  [p.join() for p in processes] # wait for the processes to finish

  finish_time = (default_timer() - start)
  print(f"all parallel: {finish_time}")

# processes images in specified number of processes
def run_parallel(files, cpus):
  chunks = numpy.array_split(numpy.array(files),cpus) # split all images into chunks as many as processes
  processes = []
  start = default_timer()

  for chunk in chunks:
    process = Process(target=process_files, args=(input_path, chunk, output_path))
    process.start()
    processes.append(process)

  [p.join() for p in processes] # wait for the processes to finish

  finish_time = (default_timer() - start)
  print(f"{cpus} parallel: {finish_time}")

# processes images sequentially
def run_sequential(files):
  start = default_timer()
  
  process_files(input_path, files, output_path)

  finish_time = (default_timer() - start)
  print(f"sequential: {finish_time}")

if __name__ == '__main__':
  files = [f for f in os.listdir(input_path)]
  run_sequential(files)
  for cpus in range(2, 13, 2):
    run_parallel(files, cpus)
  run_all_parallel(files)
