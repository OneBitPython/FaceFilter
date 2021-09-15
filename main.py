import dlib
import cv2
import numpy as np
from math import hypot

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


while True:
    ret, frame = cap.read()
    girl = frame.copy()
    boy=frame.copy()

    black = np.zeros_like(frame)
    mask = np.zeros_like(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        landmarks = predictor(gray, face)

        myPoints=[]
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y


            myPoints.append([x, y])
            # cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

        myPoints = np.array(myPoints)
        cv2.fillPoly(girl, [myPoints[48:68]], (0, 0, 255)) # lip
        cv2.fillPoly(girl, [myPoints[22:26]], (0, 0, 0)) # right eyebrow
        cv2.fillPoly(girl, [myPoints[17:21]], (0, 0, 0))  # left eyebrow
        cv2.fillPoly(girl, [myPoints[43:47]], (73, 46, 23))  # right eye
        cv2.fillPoly(girl, [myPoints[36:41]], (73, 46, 23))  # left eye

        cv2.fillPoly(boy, [myPoints[1:16]], (0, 0, 0)) # mask

        edge_face_left = myPoints[2]
        left_nose = myPoints[31]

        distanceX = hypot(left_nose[0] - edge_face_left[0])
        cv2.circle(girl, (int(edge_face_left[0]) + int(distanceX)//2, int(edge_face_left[1])), 12, (131, 93, 222), -1)


        edge_faceright = myPoints[14]
        right_nose = myPoints[35]

        distanceX = hypot(right_nose[0] - edge_faceright[0])
        cv2.circle(girl, (int(right_nose[0]) + int(distanceX)//2, int(edge_faceright[1])), 12, (131, 93, 222), -1)

        # newPoints = []
        # for n in range(31, 32):
        #     x = landmarks.part(n).x
        #     y = landmarks.part(n).y
        #
        #
        #     newPoints.append((x-40, y))
        #
        #
        # newPoints = newPoints[0]
        # cv2.circle(girl, newPoints, 10, (193, 182, 255), -1) # mask

        cv2.fillPoly(black, [myPoints[48:68]], (255, 255, 255))
        cv2.fillPoly(black, [myPoints[22:26]], (255, 255, 255)) # right eyebrow
        cv2.fillPoly(black, [myPoints[17:21]], (255, 255, 255))  # left eyebrow
        cv2.fillPoly(black, [myPoints[43:47]], (255, 255, 255))  # right eye
        cv2.fillPoly(black, [myPoints[36:41]], (255, 255, 255))  # left eye
        cv2.fillPoly(black, [myPoints[27:35]], (255, 255, 255))

        # cv2.fillPoly(black, [myPoints[0:18]], (255, 255, 255))
        cv2.fillPoly(mask, [myPoints[0:18]], (255, 255, 255))
        frame = cv2.bitwise_and(frame, mask)

    cv2.imshow('girl', girl)
    cv2.imshow('frame', frame)
    cv2.imshow('boy', boy)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

