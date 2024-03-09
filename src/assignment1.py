# pip install matplotlib opencv-python numpy

import os
import timer_wrapper as tw
from multiprocessing import Process, cpu_count, Pool
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import cv2
import numpy as np

input_dir = './data/data_set_VU_test1/Images'
output_dir = './data/assignment1_output'

# ChatGPT
def convert_to_binary(img):
  # Calculate the histogram of the grayscale image
  hist = cv2.calcHist([img], [0], None, [256], [0, 256])

  # Calculate the cumulative sum of the histogram
  cumulative_hist = np.cumsum(hist)

  # Find the midpoint of the histogram
  midpoint = np.argmax(cumulative_hist >= np.sum(hist) / 2)

  # Threshold the image using the midpoint
  _, binary_img = cv2.threshold(img, midpoint, 255, cv2.THRESH_BINARY)
  
  return binary_img

def black_and_white(input_file, output_file):
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    binary_img = convert_to_binary(img)
    cv2.imwrite(output_file, binary_img)

def blur(input_file, output_file):
  img = cv2.imread(input_file)
  blurred = cv2.GaussianBlur(img,(5,5),0)
  cv2.imwrite(output_file, blurred)

# ChatGPT
def noise(input_file, output_file):
  img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

  binary_img = convert_to_binary(img)

  # Count the number of black pixels
  black_pixel_count = np.sum(binary_img == 0)

  # Calculate the number of noise pixels (10% of black pixels)
  noise_pixel_count = int(0.1 * black_pixel_count)

  # Add noise
  noise_img = add_noise(binary_img, noise_pixel_count)

  cv2.imwrite(output_file, noise_img)

# ChatGPT
def add_noise(binary_img, noise_pixel_count):
    noise_img = binary_img.copy()

    # Generate a mask with 10% of the pixels set to 1 (noise)
    mask = np.zeros_like(binary_img)
    mask[np.random.choice(range(mask.shape[0]), size=noise_pixel_count), 
         np.random.choice(range(mask.shape[1]), size=noise_pixel_count)] = 1

    # Apply the mask to the black pixels in the binary image
    noise_img[mask == 1] = 0

    return noise_img

@tw.timeit
def parallel(func, cpu_count):
  pool = Pool(cpu_count)
  data = [(os.path.join(input_dir, file), os.path.join(output_dir, func.__name__, file)) for file in os.listdir(input_dir)]
  pool.starmap(func, data)
  pool.close()
  pool.join()

@tw.timeit
def multithreaded(func, thread_count):
  print(f'Executing {func.__name__}')
  pool = ThreadPool(thread_count)
  data = [(os.path.join(input_dir, file), os.path.join(output_dir, func.__name__, file)) for file in os.listdir(input_dir)]
  pool.starmap(func, data)
  pool.close()
  pool.join()

def execute(executor, counts, name):
  timings_bw = {}
  timings_b = {}
  timings_n = {}
  for count in counts:
    print(f'Count: {count}')
    r, timings_bw[count] = executor(black_and_white, count)
    r, timings_b[count] = executor(blur, count)
    r, timings_n[count] = executor(noise, count)
  plt.plot(timings_bw.keys(), timings_bw.values())
  plt.plot(timings_b.keys(), timings_b.values())
  plt.plot(timings_n.keys(), timings_n.values())
  plt.title(f'Processing time vs number of {name}')
  plt.ylabel('Processing time (seconds)')
  plt.xlabel(f'Number of {name}')
  plt.show()

if __name__ == '__main__':
  # parallel
  execute(parallel, range(2, 30, 2), 'CPUs')
  
  # threads
  execute(multithreaded, range(2, 400, 20), 'threads')
