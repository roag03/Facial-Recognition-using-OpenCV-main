import cv2
import numpy as np
import os

dataset = cv2.CascadeClassifier(r"/Users/anantupadhyay/Downloads/face_recogination/data.xml")
vid = cv2.VideoCapture(0)

face_1 = np.load(r"/Users/anantupadhyay/Downloads/face_recogination/faces/user_1.npy").reshape(141, 50*50)
face_2 = np.load(r"/Users/anantupadhyay/Downloads/face_recogination/faces/user_2.npy").reshape(162, 50*50)
#face_3 = np.load(r"/Users/anantupadhyay/Downloads/face_recogination/faces/user_3.npy").reshape(50, 50*50)
#face_4 = np.load("").reshape(49, 50*50)

faces_data = np.r_[face_1, face_2]
label_1 = np.zeros(243)
label_2 = np.ones(243)
#label_3 = np.twos(50)
#label_4 = np.ones(49)

labels = np.r_[label_1, label_2]
name_list = {0:"Anant", 1:"Aditya"}
print(labels)

def distance(x1,x2):
    return np.sqrt(sum(x2 - x1)**2)

def knn(x, train, k=5):
    n = train.shape[0]
    d = []
    for i in range(n):
        d.append(distance(x, train[i]))
    d = np.array(d)
    indexes = np.argsort(d)
    sortedLabels = labels[indexes][:k]
    count = np.unique(sortedLabels, return_counts=True)
    return count[0][np.argmax(count[1])]

while True:
    ret, frame = vid.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray, 1.2)
        for x,y,w,h in faces:
            print(x,y,w,h)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)
            face = gray[y:y+h, x:x+w]
            # print(face)
            face = cv2.resize(face, (50, 50))
            name = name_list[int(knn(face.reshape(2500), faces_data))]
            cv2.putText(frame, name, (x,y), cv2.FONT_HERSHEY_PLAIN, 5, (255, 180,0))
        cv2.imshow("result", frame)
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
vid.release()