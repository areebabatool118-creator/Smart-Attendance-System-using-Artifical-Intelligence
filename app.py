import cv2
import face_recognition
import pickle
import csv
import os
from datetime import datetime
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load trained model
with open("trainer/model.pkl", "rb") as f:
    data = pickle.load(f)

known_face_encodings = data["faces"]
known_face_names = data["names"]

# Open webcam
video_capture = cv2.VideoCapture(0)

# Attendance file
attendance_file = "attendance/attendance.csv"

os.makedirs("attendance", exist_ok=True)

# Create file if not exists
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

marked_names = []

def mark_attendance(name):

    if name not in marked_names:

        now = datetime.now()
        time_string = now.strftime("%H:%M:%S")

        with open(attendance_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, time_string])

        marked_names.append(name)

        print(f"Attendance Marked for {name}")

def generate_frames():

    while True:

        success, frame = video_capture.read()

        if not success:
            break

        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        # Detect faces
        face_locations = face_recognition.face_locations(
            rgb_small_frame
        )

        face_encodings = face_recognition.face_encodings(
            rgb_small_frame,
            face_locations
        )

        for face_encoding, face_location in zip(
            face_encodings,
            face_locations
        ):

            matches = face_recognition.compare_faces(
                known_face_encodings,
                face_encoding,
                tolerance=0.6
            )

            name = "Unknown"

            face_distances = face_recognition.face_distance(
                known_face_encodings,
                face_encoding
            )

            if len(face_distances) > 0:

                best_match_index = face_distances.argmin()

                if matches[best_match_index]:

                    name = known_face_names[best_match_index]

                    # Mark attendance
                    mark_attendance(name)

            # Scale back face locations
            top, right, bottom, left = face_location

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw rectangle
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            # Name
            cv2.putText(
                frame,
                name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            # Attendance Marked Message
            cv2.putText(
                frame,
                "Attendance Marked",
                (left, bottom + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Convert frame
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/video')

def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    app.run(debug=True)