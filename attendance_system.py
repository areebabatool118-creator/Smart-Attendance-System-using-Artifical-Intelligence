import cv2
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import os

# Load trained model
with open("trainer/model.pkl", "rb") as f:
    data = pickle.load(f)

known_faces = data["faces"]
known_names = data["names"]

attendance_list = []

os.makedirs("attendance", exist_ok=True)

cam = cv2.VideoCapture(0)

# Set deadline time (HH:MM:SS format)
deadline = "04:00:00"

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Current time check
    now = datetime.now().strftime("%H:%M:%S")

    # Agar time pass ho gaya
    if now > deadline:
        cv2.putText(
            frame,
            "Attendance Can't Be Marked Further",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        cv2.imshow("Smart Attendance System", frame)
        if cv2.waitKey(1) == 13:  # Enter key to quit
            break
        continue

    # Normal attendance marking
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

            if name not in attendance_list:
                attendance_list.append(name)
                now_full = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                df = pd.DataFrame([[name, now_full]], columns=["Name", "Time"])
                df.to_csv(
                    "attendance/attendance.csv",
                    mode="a",
                    header=False,
                    index=False
                )

                # Show message when attendance marked
                cv2.putText(
                    frame,
                    "Attendance Marked",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) == 13:  # Enter key to quit
        break

cam.release()
cv2.destroyAllWindows()
