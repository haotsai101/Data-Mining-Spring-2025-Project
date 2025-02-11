import os
import numpy as np
import argparse
import cv2

def main():
  source = 'data/input.mp4'
  dest_folder = 'data/frames'

  if not os.path.exists(dest_folder):
      os.makedirs(dest_folder)

  cap = cv2.VideoCapture(source)
  path_to_save = os.path.abspath(dest_folder)
  
  current_frame = 1

  if (cap.isOpened() == False):
      print('Cap is not open')

  while(cap.isOpened()):

      ret, frame = cap.read()
      if(ret == True):

          name = 'frame' + str(current_frame) + '.jpg'
          print(f'Creating: {name}')
          cv2.imwrite(os.path.join(path_to_save, name), frame)

          current_frame += 1
      
      else:
          break

  cap.release()
  print('done')

if __name__ == '__main__':
  main()