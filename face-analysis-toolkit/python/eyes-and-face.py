import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as img_keras
from collections import deque

Q = deque(maxlen=10)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# loading models
emotion_model_path = 'MediaPipe\\FuncIntegration\\_trained.hdf5'
model = load_model(emotion_model_path, compile=False)
# parameters for loading data and images
emotions = ("Angry", "Disgusted", "Feared", "Happy", "Sad", "Surprise", "Neutral")
writer = None
out_video_path = 'video.avi'
frame_counter =0
CEF_COUNTER =0
TOTAL_BLINKS =0
CLOSED_EYES_FRAME =3
FONTS =cv.FONT_HERSHEY_COMPLEX
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]
map_face_mesh = mp.solutions.face_mesh
camera = cv.VideoCapture(0)

def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # return a list of tuples for each landmark
    return mesh_coord

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# EAR
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right Eye Horizontal
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # Right Eye Vertical
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # Draw line on right eye
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # Left Eye Horizontal
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # Left Eye Vertical
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    # Right Eye Distance
    rhDistance = euclaideanDistance(rh_right, rh_left)# horizontal
    rvDistance = euclaideanDistance(rv_top, rv_bottom)# vertical

    # Left Eye Distance
    lhDistance = euclaideanDistance(lh_right, lh_left)# horizontal
    lvDistance = euclaideanDistance(lv_top, lv_bottom)# vertical

    # Ratios
    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    # Average Ratio
    ratio = (reRatio+leRatio)/2
    return ratio 

def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # Convert a color image to a scaled image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Get the dimensions of the image
    dim = gray.shape

    # Create a mask from grayscale dims
    mask = np.zeros(dim, dtype=np.uint8)

    # Draw the eye shape on the white mask The 
    # cv2.fillPoly() function can be used to fill any shape and also draw polygons. 
    # Approximate the curve. The cv2.fillPoly() function can fill multiple graphics at one time.
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # show mask 
    # cv.imshow('mask', mask) 
    # Draw the eye image on the mask, where the white shape 
    # cv2.bitwise_and(iamge,image,mask=mask) 1 RGB image selection mask selected area 2 Find the intersection of two images
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # Change black to gray except for eyes
    # cv.imshow('eyes draw', eyes)
    eyes[mask==0]=155
    
    # Get min and max x and y values ​​for right and left eye
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # Cut out the eyes from the mask
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]
    # return cropped eyes
    return cropped_right, cropped_left

def positionEstimator(cropped_eye):
     # Measure the height and width of the eyes
     h, w =cropped_eye.shape
    
     # remove noise from image
     gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)
     median_blur = cv.medianBlur(gaussain_blur, 3)

     # apply threshold to transform binary_image
     ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

     # create eye fixation part
     piece = int(w/3)

     # Divide the eyes into three parts
     right_piece = thresshed_eye[0:h, 0:piece]
     center_piece = thresshed_eye[0:h, piece: piece+piece]
     left_piece = thresshed_eye[0:h, piece +piece:w]
    
     # Call the pixel counter function
     eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

     return eye_position, color

# create pixel counter function
def pixelCounter(first_piece, second_piece, third_piece):
     # count black pixels for each part
     right_part = np.sum(first_piece==0)
     center_part = np.sum(second_piece==0)
     left_part = np.sum(third_piece==0)
     # Create a list of these values
     eye_parts = [right_part, center_part, left_part]

     # Get the index of the Max value in the list
     max_index = eye_parts. index(max(eye_parts))
     pos_eye = ''
     if max_index==0:
         pos_eye="RIGHT"
         color=[utils. BLACK, utils. GREEN]
     elif max_index==1:
         pos_eye = 'CENTER'
         color = [utils. YELLOW, utils. PINK]
     elif max_index ==2:
         pos_eye = 'LEFT'
         color = [utils.GRAY, utils.YELLOW]
     else:
         pos_eye="Closed"
         color = [utils.GRAY, utils.YELLOW]
     return pos_eye, color
 
with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
    # start time here
    start_time = time. time()
    # Start the video loop.
    while True:
        frame_counter +=1 # frame counter
        ret, frame = camera.read() # 从相机获取帧
        if not ret: 
            break # if no more frames,then break
        #  Adjust the frame
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        frame.flags.writeable = False
        #face_mesh processing result: a collection of detected faces, where each face is represented as a list of 468 face coordinates,
        results = face_mesh.process(rgb_frame) #Each coordinate is composed of x, y and z, and x and y are normalized by image width and height respectively.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=map_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                h, w, c = frame.shape
                cx_min=  w
                cy_min = h
                cx_max= cy_max= 0
                for id, lm in enumerate(face_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx<cx_min:
                        cx_min=cx
                    if cy<cy_min:
                        cy_min=cy
                    if cx>cx_max:
                        cx_max=cx
                    if cy>cy_max:
                        cy_max=cy
                        
                # crop detected face
                frame_copy = frame.copy()
                detected_face = frame_copy[int(cy_min):int(cy_max), int(cx_min):int(cx_max)]
                detected_face = cv.cvtColor(
                    detected_face, cv.COLOR_BGR2GRAY)  # transform to gray scale
                detected_face = cv.resize(detected_face, (64, 64))

                img_pixels = img_keras.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)

                # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
                img_pixels /= 255

                # store probabilities of 7 expressions
                emotion = model.predict(img_pixels)[0]
                Q.append(emotion)
                
                # perform prediction averaging over the current history of previous predictions
                res = np.array(Q).mean(axis=0)
                i = np.argmax(res)
                label = emotions[i]

                # write emotion text above rectangle
                cv.putText(frame, label, (cx_min, cy_min),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv.rectangle(frame, (cx_min, cy_min), (cx_max, cy_max), (0, 255, 0), 2)
                #cv2.circle(image, ((cx_min+cx_max)//2, (cy_min+cy_max)//2), 100, (0, 255, 0), 2)

            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
            utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)

            if ratio >3.5: # blink aspect ratio
                CEF_COUNTER +=1
                # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )

            else:
                if CEF_COUNTER>CLOSED_EYES_FRAME:
                    TOTAL_BLINKS +=1
                    CEF_COUNTER =0
            # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
            utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
            
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

            # Blink detector counter complete
            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            #eye extraction function
            crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
            # cv.imshow('right', crop_right)
            # cv.imshow('left', crop_left)
            #eye position estimator
            eye_position, color = positionEstimator(crop_right)
            utils.colorBackgroundText(frame, f'R: {eye_position}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
            eye_position_left, color = positionEstimator(crop_left)
            utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)
        # calculate FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time

        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
        # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
        cv.imshow('frame', frame)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
    cv.destroyAllWindows()
    camera.release()