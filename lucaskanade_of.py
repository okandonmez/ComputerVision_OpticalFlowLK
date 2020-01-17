#region IMPORT AREA
import os
import moviepy.video.io.VideoFileClip as mpy
import cv2
import numpy as np
from scipy import signal
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

#endregion

#region GETTING THE ROI OF IMAGE FOR MASK AND IMAGE
def mask_to_area_for_img(img, mask):

    top_left_coord = (304, 210)       # initial point for top left
    bottom_right_coord = (top_left_coord[0] + 40, top_left_coord[1] + 30)

    crop_img = img[top_left_coord[1]:bottom_right_coord[1], top_left_coord[0]:bottom_right_coord[0]]
    #print("shape of cropped image: ", crop_img.shape)

    #cv2.imshow("crop_image", crop_img)
    #cv2.waitKey()

    return crop_img

def mask_to_area_for_mask(mask):
    top_left_coord = (304, 210)       # initial point for top left
    bottom_right_coord = (top_left_coord[0] + 40, top_left_coord[1] + 30)

    crop_img = mask[top_left_coord[1]:bottom_right_coord[1], top_left_coord[0]:bottom_right_coord[0]]

    return crop_img

#endregion

#region OPTICAL FLOW CALCULATIONS

def calc_OF_LK(old_frame, current_frame, window_size = 5):

    #normalizing
    old_frame     = old_frame     / 255.
    current_frame = current_frame / 255.

    #for window size
    neighX = np.array([[-1., 1.], [-1., 1.]])
    neighY = np.array([[-1., -1.], [1., 1.]])
    neighT = np.array([[1., 1.], [1., 1.]])
    w = int(window_size / 2)

    funcX = signal.convolve2d(old_frame, neighX, boundary='symm', mode='same')
    funcY = signal.convolve2d(old_frame, neighY, boundary='symm', mode='same')
    funcT = signal.convolve2d(current_frame, neighT, boundary='symm', mode='same') + signal.convolve2d(old_frame, -neighT, boundary='symm', mode='same')


    vecX = np.zeros(old_frame.shape)
    vecY = np.zeros(old_frame.shape)

    for i in range(w, old_frame.shape[0] - w):
        for j in range(w, old_frame.shape[1] - w):

            valX = funcX[i - w:i + w + 1, j - w:j + w + 1].flatten()
            valY = funcY[i - w:i + w + 1, j - w:j + w + 1].flatten()
            valT = funcT[i - w:i + w + 1, j - w:j + w + 1].flatten()

            b = np.reshape(valT, (valT.shape[0], 1))  # get b here
            A = np.vstack((valX, valY)).T  # get A here

            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= 1e-2:
                nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here
                vecX[i, j] = nu[0]
                vecY[i, j] = nu[1]

    return (vecX, vecY)

#endregion

#region READING THE FRAMES

walker      = mpy.VideoFileClip("inputs/walker.avi")
walker_hand = mpy.VideoFileClip("inputs/walker_hand.avi")

frame_count = walker_hand.reader.nframes
video_fps   = walker_hand.fps

frames      = []
frames_hand = []

for i in range(frame_count):

    walker_frame      = walker.get_frame(i * 1.0 / video_fps )
    walker_hand_frame = walker_hand.get_frame(i  * 1.0 / video_fps )

    walker_frame = cv2.cvtColor(walker_frame, cv2.COLOR_BGR2GRAY)
    walker_hand_frame = cv2.cvtColor(walker_hand_frame, cv2.COLOR_BGR2GRAY)

    frames.append(walker_frame)
    frames_hand.append((walker_hand_frame))
#endregion

#region FINDING THE ONLY HAND COORDINATES
def give_feature_points(mask):
    c1 = 0
    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            value = mask.item(i, j)
            if value > 127:
                c1 += 1

    old_points =  np.zeros((c1,1,2), dtype=np.float32)
    counter = 0
    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            value = mask.item(i, j)
            if value > 127:
                old_points[counter][0] = [j, i]
                counter += 1
                #old_points = np.array([[i, j]], dtype=np.float32)
    return old_points

#endregion

#region SET FOR FIRST FRAME
mask = mask_to_area_for_mask(frames_hand[1])
current_hand_points = give_feature_points(mask)

old_frame = frames[0]
old_mask = frames_hand[0]
old_hand_area = mask_to_area_for_img(old_frame, old_mask)

frames_for_video = []
#endregion

#region MAIN DRIVER BLOCK
for i in range(1, len(frames)):
    print(str(i) + ". frame has been processing ...")
    if i == len(frames)-1:
        break

    if i%2 == 0:
        continue

    old_frame      = frames[i-1]
    old_mask       = frames_hand[i-1]

    current_frame = frames[i]       # big frame
    current_mask  = frames_hand[i]  # big frame segmented

    current_hand_area = mask_to_area_for_img(current_frame, current_mask)   # only hand area
    current_mask_hand = mask_to_area_for_mask(current_mask)

    #cv2.imshow("current_hand",  current_hand_area)
    #cv2.imshow("old_hand_area", current_mask_hand)
    #cv2.waitKey()

    current_hand_points = give_feature_points(current_mask_hand)

    result_vec1 = calc_OF_LK(old_hand_area, current_hand_area)

    old_hand_area = current_hand_area.copy()

    sumX = 0
    sumY = 0
    for i in range(current_hand_points.shape[0]):
        x = int(current_hand_points[i][0][0])
        y = int(current_hand_points[i][0][1])

        sumX += result_vec1[0][y][x]
        sumY += result_vec1[1][y][x]

    avX = sumX / current_hand_points.shape[0]
    avY = sumY / current_hand_points.shape[0]

    #print(avX)
    #print(avY)

    x = int(current_hand_points[int(current_hand_points.shape[0] / 2)][0][0])
    y = int(current_hand_points[int(current_hand_points.shape[0] / 2)][0][1])

    vecX = result_vec1[0][y][x] * 5
    vecY = result_vec1[1][y][x] * 5

    #print("-----------------------------")

    current_hand_area = cv2.arrowedLine(current_hand_area, (x,y),
                                        (int((avX * 10) + x), int((avY * 10) + y)), (0, 0, 0), 1)

    image_for_temp = current_hand_area.copy()
    image_for_temp = cv2.cvtColor(image_for_temp, cv2.COLOR_GRAY2BGR)
    frames_for_video.append(image_for_temp)

    #cv2.imshow("asdasd", current_hand_area)
    #cv2.waitKey()
#endregion

#region VIDEO RECORDING

clip = ImageSequenceClip(frames_for_video, fps=5) # for slow mo

try:
    clip.write_videofile('outputs/part1.mp4', codec="mpeg4")
except:
    os.mkdir('outputs')
    clip.write_videofile('outputs/part1.mp4', codec="mpeg4")

#endregion

