# -*- coding: utf-8 -*-



"""# Importing Utilitis"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image as im
from PIL import ImageFont, ImageDraw
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

import pickle
import math
import statistics
from scipy.signal import argrelextrema
import matplotlib as mpl

import matplotlib.animation as animation
from matplotlib import pyplot as plt
import time as tm
import glob
from scipy.signal import savgol_filter
import os.path
from os import path


"""# Video Utilities"""

#Convert video to array of frames

def obtain_frame(vid, frame_no,videodata):
  # length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
  # vid.set(1,frame_no)
  # res, frame = vid.read()
  # print(frame.shape)
  return videodata[frame_no]

def crop_image(frame, x_min, x_max, y_min, y_max):
  ROI = frame[y_min:y_max, x_min:x_max]
  return ROI

def vid_to_arr(vid,shape):
  # length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
  # print(vid.get(cv2.CAP_PROP_FRAME_COUNT))
  length = shape[0]
  arr_frames = []
  for i in range(length):
    arr_frames.append(obtain_frame(vid, i))
  return arr_frames

def annotate_pose(img):
  results, image = get_key_points(img)
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  mp_drawing.draw_landmarks(
      image,
      results.pose_landmarks,
      mp_pose.POSE_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
  im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return im_rgb

def annotate_vid(img_arr):
  annotated_dict = {}
  i = 0
  for img in img_arr:
    try:
      annotated_dict[i] = annotate_pose(img)
      i+=1
    except:
      i+=1
      continue
  return annotated_dict

def smoothen(feature_list,window_size=None,alpha=None):
  assert window_size is not None or alpha is not None
  assert not window_size or not alpha
  
  df=pd.DataFrame(feature_list)
  ema = df.ewm(span=window_size,alpha=alpha).mean()[0].tolist()
  return ema
  # print(ema)
  # time_part=[]
  time = [frame/fps for frame in frames]

  # for i in range(len(feature_list)):
  #   time_part.append(time[i])

  p1=plt.plot(time, ema)
  # assert len(time)==len(ema),f'{len(time),len(ema)}'


  maxima_idxs=argrelextrema(np.array(ema), np.greater)[0]
  minima_idxs=argrelextrema(np.array(ema), np.less)[0]
  plt.scatter([time[x] for x in maxima_idxs], 
              [ema[x] for x in maxima_idxs],marker='o',color='g')#Maxima
  plt.scatter([time[x] for x in minima_idxs], [ema[x] for x in minima_idxs],marker='o',color='r')#Maxima 

def smoothen_vis(feature_list,window_size=None,alpha=None):
  assert window_size is not None or alpha is not None
  assert not window_size or not alpha
  fig = plt.figure()

  
  df=pd.DataFrame(feature_list)
  ema = df.ewm(span=window_size,alpha=alpha).mean()[0].tolist()
  # print(ema)
  # time_part=[]
  time = [frame/fps for frame in frames]

  # for i in range(len(feature_list)):
  #   time_part.append(time[i])

  p1=plt.plot(time, ema)
  # assert len(time)==len(ema),f'{len(time),len(ema)}'


  maxima_idxs=argrelextrema(np.array(ema), np.greater)[0]
  minima_idxs=argrelextrema(np.array(ema), np.less)[0]
  plt.scatter([time[x] for x in maxima_idxs], 
              [ema[x] for x in maxima_idxs],marker='o',color='g')#Maxima
  plt.scatter([time[x] for x in minima_idxs], [ema[x] for x in minima_idxs],marker='o',color='r')#Maxima

  return fig

"""# TShirt and Bounding Box Utils"""

#TO FIND TSHIRT FROM BOUNDING BOXES

def get_key_points(img):
  with mp_pose.Pose(static_image_mode=True,
      model_complexity=1,
      enable_segmentation=False,
      min_detection_confidence=0.2) as pose:
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    annotated_image = img.copy()
  return results, annotated_image

def get_torso_points(keypoints,image_height, image_width):
  torso={}
  torso['right_hip']=(keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x*image_width,keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y*image_height)
  torso['left_hip']=(keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x*image_width,keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y*image_height)
  torso['right_shoulder']=(keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*image_width,keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*image_height)
  torso['left_shoulder']=(keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x*image_width,keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y*image_height)
  return torso

def get_box(torso):
  points={}#xmin, xmax, ymin, ymax
  points['x_min']=int(np.min([torso['right_hip'][0],torso['left_hip'][0],torso['right_shoulder'][0],torso['left_shoulder'][0]]))
  points['x_max']=int(np.max([torso['right_hip'][0],torso['left_hip'][0],torso['right_shoulder'][0],torso['left_shoulder'][0]]))
  points['y_min']=int(np.min([torso['right_hip'][1],torso['left_hip'][1],torso['right_shoulder'][1],torso['left_shoulder'][1]]))
  points['y_max']=int(np.max([torso['right_hip'][1],torso['left_hip'][1],torso['right_shoulder'][1],torso['left_shoulder'][1]]))
  return points

def get_tshirt_points(img):
  keypoints, _ = get_key_points(img)
  image_height, image_width, _ = img.shape
  torso=get_torso_points(keypoints,image_height, image_width)
  points=get_box(torso)
  return points

def crop_image(frame, x_min, x_max, y_min, y_max):
  ROI = frame[ y_min:y_max+1,x_min:x_max+1]
  return ROI

def apply_mediapipe(image):
  points=get_tshirt_points(image)
  tshirt=crop_image(image,points['x_min'],points['x_max'],points['y_min'],points['y_max'])
  return tshirt

def get_dict(arr_id, arr_img):
  dict1={}
  # for (i, j) in zip(arr_id, arr_img):
  #   t = apply_mediapipe(j)
  for i in range(len(arr_id)):
    t = apply_mediapipe(arr_img[i])
    if(dict1.get(i)==None):
      dict1[i]=[]
    dict1[i].append(t)
  return dict1

def save_dict(dict1):
  filehandler = open(b"dict.obj","wb")
  pickle.dump(dict1, filehandler)

"""# Video Capture"""



"""Find Direction"""

def get_nose(image):
  image_height, image_width, _ = image.shape
  results, annotated_image = get_key_points(image)
  print(results)
  nose=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y*image_height)
  return nose

def find_direction(curr, prev):
  if curr[0]<prev[0]:
    return True
  else:
    return False

"""# KFA """

#TO FIND KFA for a particular frame

def get_KFA_pts(image):
  image_height, image_width, _ = image.shape
  results, annotated_image = get_key_points(image)
  kfa_pts={}
  kfa_pts['right_hip']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y*image_height)
  kfa_pts['right_knee']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y*image_height)
  kfa_pts['right_ankle']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y*image_height)
  kfa_pts['left_hip']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y*image_height)
  kfa_pts['left_knee']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y*image_height)
  kfa_pts['left_ankle']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y*image_height)
  return kfa_pts
 
def getAngle(a, b, c):
  ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
  return ang + 360 if ang < 0 else ang

def find_KFA(image):
  dict2 = get_KFA_pts(image)
  A_right = getAngle(dict2.get('right_hip'), dict2.get('right_knee'), dict2.get('right_ankle'))
  A_left = getAngle(dict2.get('left_hip'), dict2.get('left_knee'), dict2.get('left_ankle'))
  # if(A_right>180):
  #   A_right = 180 - A_right
  # if(A_left>180):
  #   A_left = 180 - A_left
  return A_left, A_right

def KFA_arr(img_arr):
  a = []
  frame = []
  direction = []
  i = 0


  prev = get_nose(img_arr[10])
  for img in img_arr:
    try:
      KFA = find_KFA(img)
      curr = get_nose(img)
      if find_direction(curr, prev):
        a.append(KFA)
        direction.append("left")
      else:
        a.append((360 - KFA[0], 360 - KFA[1]))
        direction.append("right")
      frame.append(i)
      i+=1
      prev = curr
    except:
      i+=1
      continue
  return a, frame, direction

def KFA_min(angles, frame):
  left = smoothen([i[0] for i in angles],10)
  right = smoothen([i[1] for i in angles],10)
  min1 = []
  min2 = []
  for i in range(int(0.2*len(left)),len(left)-1):
    if (left[i]<left[i-1] and left[i]<left[i+1]):
      min1.append(left[i])
    if (right[i]<right[i-1] and right[i]<right[i+1]):
      min2.append(right[i])
  mea1 = statistics.mean(min1)
  mea2 = statistics.mean(min2)
  return mea1, mea2



"""# Hip ROM"""

#Find hip flexion

def hip_pts(image):
  image_height, image_width, _ = image.shape
  results, annotated_image = get_key_points(image)
  hip_pts={}
  hip_pts['right_hip']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y*image_height)
  hip_pts['right_knee']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y*image_height)
  hip_pts['left_hip']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y*image_height)
  hip_pts['left_knee']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y*image_height)
  hip_pts['groin'] = ((hip_pts['right_hip'][0]+hip_pts['left_hip'][0])/2, (hip_pts['right_hip'][1]+hip_pts['left_hip'][1])/2)
  return hip_pts

def hipROM(image):
  pts = hip_pts(image)
  A = getAngle(pts['right_knee'], pts['groin'], pts['left_knee'])
  if (A>180):
    A = 360-A
  return A

def hipROM_arr(img_arr):
  a = []
  frame = []
  i=0
  for img in img_arr:
    try:
      a.append(hipROM(img))
      frame.append(i)
      i+=1
    except:
      i+=1
      continue
  return a, frame

def hipROM_max(angles, frames,fps):
  time = [frame/fps for frame in frames]
  a = smoothen(angles, 10)
  max = []
  arr = []
  for i in range(int(0.2*len(a)),len(a)-2):
    if(a[i]>a[i-1] and a[i]>a[i-2] and a[i]>a[i+1] and a[i]>a[i+2]):
      max.append(a[i])
      arr.append(i)
  mean_max = statistics.mean(max)
  return mean_max, arr

"""Original Plot"""


"""# Ankle ROM"""

# Ankle ROM

def get_ankle_pts(image):
  image_height, image_width, _ = image.shape
  results, annotated_image = get_key_points(image)
  ankle_pts={}
  ankle_pts['right_foot_index']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y*image_height)
  ankle_pts['right_knee']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y*image_height)
  ankle_pts['right_ankle']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y*image_height)
  ankle_pts['left_foot_index']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y*image_height)
  ankle_pts['left_knee']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y*image_height)
  ankle_pts['left_ankle']=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x*image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y*image_height)
  return ankle_pts

def find_ankle_ROM(image):
  dict2 = get_ankle_pts(image)
  A_right = getAngle(dict2.get('right_foot_index'), dict2.get('right_ankle'), dict2.get('right_knee'))
  A_left = getAngle(dict2.get('left_foot_index'), dict2.get('left_ankle'), dict2.get('left_knee'))
  if A_left>180:
    A_left = 360-A_left
  if A_right>180:
    A_right = 360-A_right
  return A_left, A_right

def ankle_ROM_arr(img_arr):
  ankle_ROM = []
  frame = []
  i=0
  for img in img_arr:
    try:
      ankle_ROM.append(find_ankle_ROM(img))
      frame.append(i)
      i+=1
    except:
      i+=1
      continue
  return ankle_ROM, frame

def ankle_ROM_min_max(a, frame):
  max1 = []
  max2 = []
  min1 = []
  min2 = []
  for i in range(int(0.2*len(a)),len(a)-2):
    if(a[i][0]>a[i-1][0] and a[i][0]>a[i-2][0] and a[i][0]>a[i+1][0] and a[i][0]>a[i+2][0]):
      max1.append(a[i][0])
    if(a[i][1]>a[i-1][1] and a[i][1]>a[i-2][1] and a[i][1]>a[i+1][1] and a[i][1]>a[i+2][1]):
      max2.append(a[i][1])
    if(a[i][0]<a[i-1][0] and a[i][0]<a[i-2][0] and a[i][0]<a[i+1][0] and a[i][0]<a[i+2][0]):
      min1.append(a[i][0])
    if(a[i][1]<a[i-1][1] and a[i][1]<a[i-2][1] and a[i][1]<a[i+1][1] and a[i][1]<a[i+2][1]):
      min2.append(a[i][1])
  mean_max = statistics.mean(max1), statistics.mean(max2)
  mean_min = statistics.mean(min1), statistics.mean(min2)
  return mean_max, mean_min

"""Original Plot"""


def remove_initial(arr_x,frames):
  index=[]
  for i in range(len(arr_x)):
    if(arr_x[i]<int(0.2*len(frames))):
      index.append(i)
  arr_x=np.delete(arr_x,index)
  return arr_x

"""# Combined Plot"""

def comb_plot(frames,time,hipROM_a,KFA_a,ankle_ROM_a,imgs,imgf,i):
  
  fig=plt.figure()
  
  # Placing the plots in the plane
  plot1 = plt.subplot2grid((4, 8), (0, 0),colspan=2)
  plot2 = plt.subplot2grid((4, 8), (1, 0),colspan=2)
  plot3 = plt.subplot2grid((4, 8), (2, 0),colspan=2)
  plot5 = plt.subplot2grid((4, 8), (1, 2),colspan=2)
  plot6 = plt.subplot2grid((4, 8), (2, 2),colspan=2)
  plot4 = plt.subplot2grid((4, 8), (0, 4), rowspan=4, colspan=5)
  # plot7= plt.subplot2grid((4, 8), (2, 4), rowspan=2, colspan=4)

  plt.xlim,plt.ylim=(0,10),(0,200)
  plot1.xlim,plot1.ylim=(0,10),(0,200)
  plot2.xlim,plot2.ylim=(0,10),(0,200)
  plot3.xlim,plot3.ylim=(0,10),(0,200)

  #HIP subplot
  df=pd.DataFrame(hipROM_a)
  ema = df.ewm(span=10,).mean()[0].tolist()
  
  # time = [frame/fps for frame in frames]
  plot1.xlim,plot1.ylim=(0,10),(0,200)
  plot1.plot(time, ema)
  # ema2=ema[int(0.5*len(ema)):]
  maxima_idxs=argrelextrema(np.array(ema), np.greater)[0]
  maxima_idxs=remove_initial(maxima_idxs,frames)
  minima_idxs=argrelextrema(np.array(ema), np.less)[0]
  minima_idxs=remove_initial(minima_idxs,frames)
  plot1.scatter([time[x] for x in maxima_idxs],[ema[x] for x in maxima_idxs],marker='.',color='g')#Maxima
  plot1.scatter([time[x] for x in minima_idxs],[ema[x] for x in minima_idxs],marker='.',color='r')#Minima    
  # plot2.plot(smoothen_vis(hipROM_a,10))
  # plot1.set_title('Hip ROM')
  plot1.set_title('Hip ROM:'+str(round(ema[-1],2)),fontsize = 7)

  # xlim =(0, 10), ylim =(0, 200))

  #KFA subplot
  df=pd.DataFrame(KFA_a)
  ema=df.ewm(span=10,).mean()
  ema_l = df.ewm(span=10,).mean()[0].tolist()
  ema_r=df.ewm(span=10,).mean()[1].tolist()
  # time = [frame/fps for frame in frames]
  plot2.xlim,plot2.ylim=(0,10),(0,200)
  plot2.plot(time, ema_l)
  plot5.plot(time, ema_r)
  # ema_l2=ema_l[int(0.5*len(ema_l)):]
  # ema_r2=ema_r[int(0.5*len(ema_r)):]
  #Left leg
  maxima_idxs=argrelextrema(np.array(ema_l), np.greater)[0]
  maxima_idxs=remove_initial(maxima_idxs,frames)
  minima_idxs=argrelextrema(np.array(ema_l), np.less)[0]
  minima_idxs=remove_initial(minima_idxs,frames)
  plot2.scatter([time[x] for x in maxima_idxs],[ema_l[x] for x in maxima_idxs],marker='.',color='g')#Maxima
  plot2.scatter([time[x] for x in minima_idxs],[ema_l[x] for x in minima_idxs],marker='.',color='r')#Minima    
  #Right leg
  maxima_idxs=argrelextrema(np.array(ema_r), np.greater)[0]
  maxima_idxs=remove_initial(maxima_idxs,frames)
  minima_idxs=argrelextrema(np.array(ema_r), np.less)[0]
  minima_idxs=remove_initial(minima_idxs,frames)
  plot5.scatter([time[x] for x in maxima_idxs],[ema_r[x] for x in maxima_idxs],marker='.',color='g')#Maxima
  plot5.scatter([time[x] for x in minima_idxs],[ema_r[x] for x in minima_idxs],marker='.',color='r')#Minima    

  # plot2.set_title('KFA')
  plot2.set_title('KFA L:'+str(round(ema_l[-1],2)),fontsize = 7)
  plot5.set_title('KFA R:'+str(round(ema_r[-1],2)),fontsize = 7)

  #Ankle subplot
  df=pd.DataFrame(ankle_ROM_a)
  ema=df.ewm(span=10,).mean()
  ema_l = df.ewm(span=10,).mean()[0].tolist()
  ema_r=df.ewm(span=10,).mean()[1].tolist()
  # time = [frame/fps for frame in frames]
  plot3.xlim,plot3.ylim=(0,10),(0,200)
  plot3.plot(time, ema_l)
  plot6.plot(time, ema_r)
  # plot3.set_title('Ankle ROM L:'+str(ema[0][-1])+'Ankle ROM R:'+str(ema[1][-1]),fontsize = 12)
  # ema_l2=ema_l[int(0.5*len(ema_l)):]
  # ema_r2=ema_r[int(0.5*len(ema_r)):]
  #Left leg
  maxima_idxs=argrelextrema(np.array(ema_l), np.greater)[0]
  maxima_idxs=remove_initial(maxima_idxs,frames)
  minima_idxs=argrelextrema(np.array(ema_l), np.less)[0]
  minima_idxs=remove_initial(minima_idxs,frames)
  plot3.scatter([time[x] for x in maxima_idxs],[ema_l[x] for x in maxima_idxs],marker='.',color='g')#Maxima
  plot3.scatter([time[x] for x in minima_idxs],[ema_l[x] for x in minima_idxs],marker='.',color='r')#Minima    
  #Right leg
  maxima_idxs=argrelextrema(np.array(ema_r), np.greater)[0]
  maxima_idxs=remove_initial(maxima_idxs,frames)
  minima_idxs=argrelextrema(np.array(ema_r), np.less)[0]
  minima_idxs=remove_initial(minima_idxs,frames)
  plot6.scatter([time[x] for x in maxima_idxs],[ema_r[x] for x in maxima_idxs],marker='.',color='g')#Maxima
  plot6.scatter([time[x] for x in minima_idxs],[ema_r[x] for x in minima_idxs],marker='.',color='r')#Minima     

  plot3.set_title('Ankle L:'+str(round(ema_l[-1],2)),fontsize = 7)
  plot6.set_title('Ankle R:'+str(round(ema_r[-1],2)),fontsize = 7)

  plot4.axis('off')
  plot4.imshow(imgs)

  # plot7.axis('off')
  # plot7.imshow(imgf)


  plt.tight_layout()
  # plt.show()          #If want to visualize simultaneously uncomment this
  plt.savefig('/root/'+str(i)+'.png')







  # imgi=annotate_pose(obtain_frame(vidcap,frames[i]))    # need to chang to annotated pose
  # comb_fig.append(comb_plot(time=timei,hipROM_a=hipi,KFA_a=kfai,ankle_ROM_a=anklei,img=imgi,i=i+1))

# data = im.fromarray(dict1[250])
# comb_fig[12].canvas.draw()
# image = np.frombuffer(comb_fig[12].canvas.tostring_rgb(), dtype='uint8')
# data=im.fromarray(image)
# data

# img = cv2.imread('/content/'+str(1)+'.png')
# img_c=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_array.append(img_c)

# img = cv2.imread('/content/200.png')
# height, width, layers = img.shape
# img=cv2.putText(img,'Made by BITS Pilani',(int(2.9*width/4),int(height/19)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
# img=cv2.putText(img,'Transient State',(int(1.1*width/4),int(1.3*height/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
# img=cv2.putText(img,'HipROM: '+str(round(hipROM_print[0],2)),(int(width/20),int(4.9*height/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
# img=cv2.putText(img,'HipROM: '+str(round(hipROM_print[0],2)),(int(width/20),int(5.3*height/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
# img=cv2.putText(img,'HipROM: '+str(round(hipROM_print[0],2)),(int(width/20),int(5.7*height/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
# # img=cv2.putText(img,'Transient State',(int(1.1*width/4),int(height/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
# cv2_imshow(img)

# def write_cred(img):
#   height, width, layers = img.shape
#   img=cv2.putText(img,'Made by BITS Pilani',(int(2.9*width/4),int(height/19)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
#   return img

def write_state(img,flag):
  height, width, layers = img.shape
  if(flag==1):
    img=cv2.putText(img,'Transient State',(int(1.1*width/4),int(1.3*height/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
  else:
    img=cv2.putText(img,'Active State',(int(1.1*width/4),int(1.3*height/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
  return img

def write_features(img,threshold,hipROM_print,KFA_print,ankle_ROM_print):
    height, width, layers = img.shape
    if(hipROM_print[0]<threshold['hip']):
       img=cv2.putText(img,'Problem in Hip, Hip ROM: '+str(round(hipROM_print[0],2)),(int(width/20),int(4.9*height/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0,255), 1, cv2.LINE_AA)
    else:
       img=cv2.putText(img,'Hip is fine , Hip ROM: '+str(round(hipROM_print[0],2)),(int(width/20),int(4.9*height/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

    if(KFA_print[0]<threshold['knee'] or KFA_print[1]<threshold['knee']):
       img=cv2.putText(img,'Problem in Knee, '+' KFA L: '+str(round(KFA_print[0],2))+' R:'+str(round(KFA_print[1],2)),(int(width/20),int(5.3*height/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0,255), 1, cv2.LINE_AA)
    else:
       img=cv2.putText(img,'Knee is fine , '+' KFA L: '+str(round(KFA_print[0],2))+' R:'+str(round(KFA_print[1],2)),(int(width/20),int(5.3*height/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

    if(ankle_ROM_print[0]<threshold['ankle'] or ankle_ROM_print[1]<threshold['ankle']):
       img=cv2.putText(img,'Problem in Ankle, '+' Ankle ROM L: '+str(round(ankle_ROM_print[0],2))+' R:'+str(round(ankle_ROM_print[1],2)),(int(width/20),int(5.7*height/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 0,255), 1, cv2.LINE_AA)
    else:
       img=cv2.putText(img,'Ankle is fine , '+' Ankle ROM L: '+str(round(ankle_ROM_print[0],2))+' R:'+str(round(ankle_ROM_print[1],2)),(int(width/20),int(5.7*height/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)



    # img=cv2.putText(img,'Hip ROM: '+str(round(hipROM_print[0],2))+' KFA L: '+str(round(KFA_print[0],2))+' R:'+str(round(KFA_print[1],2))+' Ankle ROM L: '+str(round(ankle_ROM_print[0],2))+' R:'+str(round(ankle_ROM_print[1],2)),(int(width/4),int(height/5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
    # cv2_imshow(img)
    return img

def display_features(threshold,frames,time,hipROM_print,kfa,ankle):
  img_array = []
  for i in range(1,len(time)):
    # img_file=glob.glob("/content/"+str(i)+".png")
    if(path.exists('/root/'+str(i)+'.png')):
      img = cv2.imread('/root/'+str(i)+'.png')
    else:
      break

    # img_c=cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)
    height, width, layers = img.shape
    size = (width,height)
    # img=write_cred(img)
    if(i>0.6*len(time)):
      img=write_features(img,threshold,hipROM_print,kfa,ankle)
    
    if(i<0.2*len(frames)):
      img=write_state(img,1)
    else:
      img=write_state(img,0)
    
    img_array.append(img)
  return img_array

def final():

  import skvideo.io  
  video = "/var/www/html/videos/video.mp4"
  videodata = skvideo.io.vread(video)  
  vidcap = cv2.VideoCapture(video)
  # arr = vid_to_arr(vidcap,videodata.shape)
  arr = videodata
  fps = vidcap.get(cv2.CAP_PROP_FPS)
  print(fps)
  # vidcap2=cv2.VideoCapture('siddharth_4_front.avi')
  
  dict1 = annotate_vid(arr)
  print("Video Processing Done")
  print("KFA Started")
  KFA_a, frames, direction = KFA_arr(arr)

  KFA_print=KFA_min(KFA_a, frames)
  KFA_print

  """Original KFA -Orange:Left Leg and Blue:Right Leg"""

  #Original Plot -Orange:Left Leg and Blue:Right Leg
  time = [frame/fps for frame in frames]
  plt.plot(time, KFA_a)
  print("KFA Done")
  print("Hip Analysis Started")
  hipROM_a, frames = hipROM_arr(arr)
  # plt.plot(frames, hipROM_a)

  hipROM_print=hipROM_max(hipROM_a, frames,fps)
  hipROM_print

  """Original Plot(Using Time)"""

  time = [frame/fps for frame in frames]

  plt.plot(time, hipROM_a)

  """Smoothened Plot"""

  smooth_hip = smoothen(hipROM_a,10)
  plt.plot(time, hipROM_a)

  print("Hip Analysis over")
  print("Ankle Analysis Started")
  ankle_ROM_a, frames = ankle_ROM_arr(arr)
  plt.plot(frames, ankle_ROM_a)

  print(ankle_ROM_min_max(ankle_ROM_a, frames))

  x,ankle_ROM_print=[],[]
  x=ankle_ROM_min_max(ankle_ROM_a, frames)
  ankle_ROM_print.append(abs(x[0][0]-x[1][0]))
  ankle_ROM_print.append(abs(x[0][1]-x[1][1]))

  # ankle_ROM_print[1]=x[1][1]-x[1][0]
  ankle_ROM_print

  """Original Plot(Using Time)"""

  time = [frame/fps for frame in frames]
  plt.plot(time, [i[1] for i in ankle_ROM_a])
  print("Ankle Analysis Done")
  print("Video Analysis staert")
  imgs = cv2.cvtColor(annotate_pose(obtain_frame(vidcap,frames[0],videodata)), cv2.COLOR_BGR2RGB)
  # cv2_imshow(imgs)

  timei,hipi,kfai,anklei,comb_fig=[],[],[],[],[]

  for i in range(len(time)):
    timei.append(time[i])
    hipi.append(hipROM_a[i])
    kfai.append(KFA_a[i])
    anklei.append(ankle_ROM_a[i])
    # imgfront = cv2.cvtColor(annotate_pose(obtain_frame(vidcap,frames[i])), cv2.COLOR_BGR2RGB)
    imgside = cv2.cvtColor(annotate_pose(obtain_frame(vidcap,frames[i],videodata)), cv2.COLOR_BGR2RGB)
    comb_plot(frames,time=timei,hipROM_a=hipi,KFA_a=kfai,ankle_ROM_a=anklei,imgs=imgside,imgf=None,i=i+1)

  threshold={'hip':25.6,'ankle':10.54,'knee':142}

  img = cv2.imread('/root/1.png')
  height, width, layers = img.shape
  size = (width,height)
  print(size)

  import pickle
  f = open("/root/arr.pkl", "wb")
  img_array = []
  for i in range(1,len(time)):
    # img_file=glob.glob("/content/"+str(i)+".png")
    if(path.exists('/root/'+str(i)+'.png')):
      img = cv2.imread('/root/'+str(i)+'.png')
    img_array.append(img)
  pickle.dump(img_array,f)

  # !cp /content/arr.pkl /content/drive/MyDrive/KT

  out = cv2.VideoWriter('/root/fin.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
  img_array=display_features(threshold,frames,time,hipROM_print,KFA_print,ankle_ROM_print)
  for i in range(len(img_array)):
      # image = np.frombuffer(comb_fig[i].canvas.tostring_rgb(), dtype='uint8')
    out.write(img_array[i])
  out.release()
  print("Video made successfully")

