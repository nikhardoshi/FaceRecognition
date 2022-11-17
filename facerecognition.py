import cv2 as cv
import numpy as np

# features=np.load(r'features.npy',allow_pickle=True)
# labels=np.load(r'labels.npy',allow_pickle=True)

people =['Andrew Garfield', 'Stephen Amell', 'Dylan Obrien','Pierce Brosnan']
face_recog=cv.face.LBPHFaceRecognizer_create()
face_recog.read(r'trainedfaces.yml')

img=cv.imread(r"C:\Users\nikso\Downloads\dylanobrien.jpg")
resized_img=cv.resize(img,(450,450))
cv.imshow('Rss',resized_img)
gray=cv.cvtColor(resized_img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

haarcsc=cv.CascadeClassifier('haar_face.xml')
face_rec=haarcsc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6)

for (x,y,w,h) in face_rec:
    face_roi=gray[y:y+h,x:x+w]

    label,confidence=face_recog.predict(face_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    cv.putText(resized_img,str(people[label]),(x-10,y-10),cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),2)
    cv.rectangle(resized_img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('Detected',resized_img)

cv.waitKey(0)


