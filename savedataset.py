import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set width
cam.set(4, 480)  # Set height

faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faceid = input('\n Enter user id and press <return> ==>  ')

count = 0
pathDataset = 'dataset/' + str(faceid) + '/'
if not os.path.exists(pathDataset):
    os.makedirs(pathDataset)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        count += 1
        # Tampilkan deteksi wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite(pathDataset + str(count) + '.jpg', gray[y:y+h, x:x+w])
        print(f"[INFO] Gambar ke-{count} disimpan.")

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27 or count >= 30:
        break

print("\n[INFO] Exiting program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
