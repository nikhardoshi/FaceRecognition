import os
import cv2 as cv
import numpy as np
import caer

people =['Andrew Garfield', 'Stephen Amell', 'Dylan Obrien','Pierce Brosnan']

# just for checking
# p=[]
# for i in os.listdir(r'C:\Users\nikso\OneDrive\Desktop\OpenCv\FaceRecognition'):
#     p.append(i)
# print(p)
dir=r'C:\Users\nikso\OneDrive\Desktop\OpenCv\FaceRecognition'
features=[]
facelabels=[]
haarcsc=cv.CascadeClassifier('haar_face.xml')
def trains():
    for person in people:
        path=os.path.join(dir,person)
        label=people.index(person)
          
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
            print(f'Path i: {img_path}')
            resized_img=cv.resize(img_array,(450,450))
            gray=cv.cvtColor(resized_img,cv.COLOR_BGR2GRAY)
            # cv.imshow('Rss',resized_img)

            face_rec=haarcsc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6)
            for (x,y,w,h) in face_rec:
                face_roi=gray[y:y+h,x:x+w]
                features.append(face_roi)
                facelabels.append(label)
            

trains()               
print('Length of features',len(features))
print('Length of labels',len(facelabels))

print('Training Done-----')
face_recog=cv.face.LBPHFaceRecognizer_create()
features1=np.array(features,dtype='object')
labels=np.array(facelabels)
#train the face recognizer on feature list and facelabels list
face_recog.train(features1,labels)

face_recog.save('trainedfaces.yml')
np.save('features1.npy',features)
np.save('labels.npy',labels)
cv.waitKey(0)