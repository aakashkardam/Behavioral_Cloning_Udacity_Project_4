import csv
import cv2

lines = []
with open('/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
  render = csv.reader(csvfile)
  for line in render:
    lines.append(line)
    print(line) 
