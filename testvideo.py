# Buat file test.py
import cv2

cap = cv2.VideoCapture("sc/tester_30.mp4")
if cap.isOpened():
    print("Video berhasil dibuka!")
    ret, frame = cap.read()
    if ret:
        print("Frame berhasil dibaca!")
    else:
        print("Tidak bisa baca frame")
else:
    print("Video tidak bisa dibuka")
cap.release()