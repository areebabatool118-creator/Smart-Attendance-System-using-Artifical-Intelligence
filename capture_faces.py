import cv2
import os

name = input("Enter Student Name: ")

path = f"dataset/{name}"

os.makedirs(path, exist_ok=True)

cam = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cam.read()

    cv2.imshow("Capture Faces", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    file_name = f"{path}/{count}.jpg"

    cv2.imwrite(file_name, gray)

    count += 1

    if count == 50:
        break

    if cv2.waitKey(1) == 13:
        break

cam.release()
cv2.destroyAllWindows()

print("Face Captured Successfully")