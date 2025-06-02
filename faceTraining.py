import cv2
import os
import numpy as np
from PIL import Image

path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    facesamples = []
    ids = []

    dirs = os.listdir(path)
    
    for dir in dirs:
        fullPath = os.path.join(path, dir) 
        images = [os.path.join(fullPath, f) for f in os.listdir(fullPath)]

        for image in images:
            PILimg = Image.open(image)

            imgNumpy = np.array(PILimg, 'uint8')

            faces = detector.detectMultiScale(imgNumpy)
            for (x, y, w, h) in faces:
                facesamples.append(imgNumpy[y:y+h, x:x+w])
                ids.append(int(dir))
    return facesamples, ids

faces, ids = getImagesAndLabels(path)

recognizer.train(faces, np.array(ids))

recognizer.save('trainer/trainer.yml')

print("\n[INFO] Training complete. Model saved as 'trainer/trainer.yml'.")
print(f"\n[INFO] {len(np.unique(ids))} faces trained. Exiting program.")