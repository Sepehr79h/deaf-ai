# For more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import cv2
import numpy as np
import torch


net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
with open("PoseLSTM.pt", 'rb') as f:
    model = torch.load(f)
    model.eval()

# mean_std = np.load(f"dataset_mean_and_std.npy")
mean = 140.1
std = 114.5

# create numpy array of wlasl vids that is of shape (num_frames, 38)
# # def generate_pose_data(width=368, height=368, max_duration=10):
#     body_parts = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#                     "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#                     "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
#                     "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

#     all_points = []
#     try:
#         cap = cv2.VideoCapture(0)
#         for i in range(0, 100):
#             _, frame = cap.read()

#             frame_width = frame.shape[1]
#             frame_height = frame.shape[0]

#             net.setInput(
#                 cv2.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
#             out = net.forward()

#             points = []
#             for j in range(len(body_parts)):
#                 # Slice heatmap of corresponding body's part.
#                 heat_map = out[0, j, :, :]
#                 _, conf, _, point = cv2.minMaxLoc(heat_map)
#                 x = (frame_width * point[0]) / out.shape[3]
#                 y = (frame_height * point[1]) / out.shape[2]
#                 # Add a point if it's confidence is higher than threshold.
#                 points.append(float(x))
#                 points.append(float(y))
#                 cv2.imshow('frame',frame)

#             all_points.append(points)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         all_points = np.array(all_points)
        
#     except:
#         pass
#     return all_points

def openPose(frame, width=368, height=368):
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    print(f"{frame_width}, {frame_height}")
    body_parts = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

    net.setInput(
        cv2.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()

    points = []
    for j in range(len(body_parts)):
        # Slice heatmap of corresponding body's part.
        heat_map = out[0, j, :, :]
        _, conf, _, point = cv2.minMaxLoc(heat_map)
        x = (frame_width * point[0]) / out.shape[3]
        y = (frame_height * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append(float(x))
        points.append(float(y))
        cv2.imshow('frame',frame)
    return np.array(points)


def inference(inputs):
    inputs = inputs.float()
    inputs = (inputs - mean) / std
    inputs = inputs.unsqueeze(0)
    print(inputs)
    outputs = model(inputs)
    return outputs[-1]

# Playing video from file:
cap = cv2.VideoCapture('sign_clip.avi')
# Capturing video from webcam:
cap = cv2.VideoCapture(0)

num_frames = 100
currentFrame = 0
inputs = []
for i in range(100):
    inputs.append(np.zeros((38)))

while(True):
    # Capture frame-by-frame
    label = "Starting Up"
    ret, frame = cap.read()

    # Handles the mirroring of the current frame
    frame = cv2.flip(frame,1)
    # pose_data = openPose(frame)
    pose_data = openPose(cv2.resize(frame, (426, 240), interpolation = cv2.INTER_LINEAR))
    
    # print(pose_data)
    inputs.append(pose_data)
    inputs_size = len(inputs)
    for i in range(inputs_size, num_frames, 1):
        inputs.append(np.zeros((38)))
    # print(len(inputs))
    if len(inputs) > num_frames:
        inputs.pop(0)
        pred = inference(torch.tensor(inputs))
        print(pred)
        pred = 1 if pred > 0.5 else 0
        label = "Signing" if pred == 1 else "Not Signing"


    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frame = cv2.putText(frame, label, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0,0,255), 2)

    # Display the resulting frame
    # cv2.imshow('frame',gray)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()