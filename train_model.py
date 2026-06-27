import face_recognition
import os
import pickle

known_faces = []
known_names = []

dataset_path = "dataset"

for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        print(f"Processing: {image_path}")

        image = face_recognition.load_image_file(image_path)

        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:

            known_faces.append(encodings[0])
            known_names.append(person_name)

            print(f"Face Added: {person_name}")

        else:
            print(f"No face found in {image_name}")

data = {
    "faces": known_faces,
    "names": known_names
}

os.makedirs("trainer", exist_ok=True)

with open("trainer/model.pkl", "wb") as f:
    pickle.dump(data, f)

print("Training Complete")