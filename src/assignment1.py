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

def convert_to_binary(img):
  """Convert an image to binary with pixel values either 1 or 0.
  Created with help from ChatGPT.

  Parameters
  ----------
  img : list
    An array with image data (2d)

  Returns
  -------
  binary_img : list
    An array with binary image data (2d)
  """
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
  """Convert image to black and white and write it to new file.

  Parameters
  ----------
  input_file : str
    A file path of input image
  output_file : str
    A file path of output image

  Returns
  -------
  None
  """
  img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
  binary_img = convert_to_binary(img)
  cv2.imwrite(output_file, binary_img)

def blur(input_file, output_file):
  """Add blur to the image and write it to new file.

  Parameters
  ----------
  input_file : str
    A file path of input image
  output_file : str
    A file path of output image

  Returns
  -------
  None
  """
  img = cv2.imread(input_file)
  blurred = cv2.GaussianBlur(img,(5,5),0)
  cv2.imwrite(output_file, blurred)

def noise(input_file, output_file):
  """Add noise to the image and write it to new file.
  Created with help from ChatGPT.

  Parameters
  ----------
  input_file : str
    A file path of input image
  output_file : str
    A file path of output image

  Returns
  -------
  None
  """
  img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

  binary_img = convert_to_binary(img)

  # Count the number of black pixels
  black_pixel_count = np.sum(binary_img == 0)

  # Calculate the number of noise pixels (10% of black pixels)
  noise_pixel_count = int(0.1 * black_pixel_count)

  # Add noise
  noise_img = add_noise(binary_img, noise_pixel_count)

  cv2.imwrite(output_file, noise_img)

def add_noise(binary_img, noise_pixel_count):
  """Add noise to the image.
  Created with help from ChatGPT.

  Parameters
  ----------
  binary_img : list
    An image to add noise to (with pixel values either 1 or 0)
  noise_pixel_count : int
    The count of pixels that constitute noise

  Returns
  -------
  noise_img : list
    An array of image data with noise added to it
  """
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
  """Process a task using parallel processing (multiprocessor)

  Parameters
  ----------
  func : func
    A function to execute, either noise(), blur() or black_and_white()
  cpu_count : int
    The count of CPUs

  Returns
  -------
  None
  """
  pool = Pool(cpu_count)
  data = [(os.path.join(input_dir, file), os.path.join(output_dir, func.__name__, file)) for file in os.listdir(input_dir)]
  pool.starmap(func, data)
  pool.close()
  pool.join()

@tw.timeit
def multithreaded(func, thread_count):
  """Process a task using concurrent processing (multithreading)

  Parameters
  ----------
  func : func
    A function to execute, either noise(), blur() or black_and_white()
  thread_count : int
    The count of threads

  Returns
  -------
  None
  """
  print(f'Executing {func.__name__}')
  pool = ThreadPool(thread_count)
  data = [(os.path.join(input_dir, file), os.path.join(output_dir, func.__name__, file)) for file in os.listdir(input_dir)]
  pool.starmap(func, data)
  pool.close()
  pool.join()

def execute(executor, counts, name):
  """Main function to execute the tasks and draw diagrams of results

  Parameters
  ----------
  executor : func
    A function to execute, either parallel() or multithreaded()
  counts : int
    The count of CPUs or threads, depending on th executor
  name : string
    The name of processing units, either 'CPUs' or 'threads'

  Returns
  -------
  None
  """
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
