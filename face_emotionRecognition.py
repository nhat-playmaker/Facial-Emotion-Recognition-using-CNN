# Importing the libraries
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import *
import numpy as np

# Load model
model = load_model('new_dataset_model.h5')

# Loading DNN model
modelFile = 'res10_300x300_ssd_iter_140000.caffemodel'
configFile = 'deploy.prototxt.txt'
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
conf_threshold = 0.5

# Emotion classifier array
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Count frame
frame_cnt = 0

label = ''

# Using webcam to have imaged for detecting face
# VideoCapture(i),
# i = 0: Using embedded laptop/PC webcam
# i = 1: Using external webcam
capture = cv2.VideoCapture(0)

# Loop for each frame
while True:
    _, frame = capture.read()

    frame_cnt += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    labels = []

    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104., 177., 123.))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:

            # Face bounding box
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, w, h) = box.astype('int')
            cv2.rectangle(frame, (x, y), (w, h), (255, 255, 0), 2)

            if frame_cnt == 10:
                # Reset frame counter
                frame_cnt = 0

                gray_roi = gray[y:h, x:w]

                if gray_roi.shape[0] == 0 or gray_roi.shape[1] == 0:
                    continue
                else:
                    gray_roi = cv2.resize(gray_roi, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([gray_roi]) != 0:
                    roi = gray_roi.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # make a prediction on the ROI, then lookup the class
                    predict = model.predict(roi)[0]
                    print(model.predict(roi))
                    label = class_labels[predict.argmax()]
                    print(label)

            label_position = (x, y - 20)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow('Emotion :3', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
