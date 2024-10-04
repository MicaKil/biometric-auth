import face_recognition
import cv2
import numpy as np
import os

# Function to load and encode sample pictures
def learn_faces(picture_files, input_database):
    user_face_encodings = [] 
    user_names = []
    for picture in picture_files:
        if picture.split(".")[0] + ".npy" not in encoded_files:
        
            picture_image = face_recognition.load_image_file(input_database +"/"+ picture)
            picture_face_encoding = face_recognition.face_encodings(picture_image)[0]
            np.save("Face_encodings/" + picture.split(".")[0], picture_face_encoding)

            user_face_encodings.append(picture_face_encoding)
            user_names.append(picture.split(".")[0])
        
        
            print(f"Picture {picture} loaded and encoded.")

    print("Sample pictures loaded and encoded.")
    return user_face_encodings, user_names

def display(face_locations, face_names):
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Folders names 
input_database = "Input_images"
encoded_faces_database = "Face_encodings"

#List .jpg files in the input_database
picture_files = os.listdir(input_database)
map(lambda x : x.endswith('.jpg'), picture_files)

#List .npy files in the encoded_faces_database 
encoded_files = os.listdir(encoded_faces_database)
map(lambda x : x.endswith('.npy'), encoded_files)

# Load a sample picture and learn how to recognize it.

user_face_encodings = []
user_names = []

if picture_files != []:
    print("Loading sample pictures...")
    user_face_encodings, user_names = learn_faces(picture_files,input_database)

# Load the known face encodings and names
for encoded in encoded_files:
    user_face_encodings.append(np.load(encoded_faces_database +"/"+ encoded))
    user_names.append(encoded.split(".")[0])

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
authorized = False

# Ask for the mode of the program
print("Press 'r' to recognize faces.")
print("Press 'a' to authorize user.")
print("Press 'l' to load .jpg faces from a folder.")
mode = ''
while mode != 'r' and mode != 'a':
    mode = input("Mode: ")
    if mode == 'l':
        path = input("Path to the folder that contains the pictures: ")
        picture_files = os.listdir(path)
        map(lambda x : x.endswith('.jpg'), picture_files)
        input_database = path
        user_face_encodings, user_names = learn_faces(picture_files,input_database)
        print("Faces loaded.")

while True:

    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        print(f"Found {len(face_locations)} face(s) in the frame.")
        if face_locations:
            print(rgb_frame.shape)
            print(face_locations)
            # Choose the biggest face
            if mode == 'a':
                face_locations = [max(face_locations, key = lambda x: (x[2]-x[0])*(x[1]-x[3]))]
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            print(f"Encoded {len(face_encodings)} face(s).")

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            i = 0
            name = "Unknown"
            while name == "Unknown" and i < len(user_face_encodings):
                user_face_encoding = user_face_encodings[i]

                # User recognition with a tolerance of 0.6
                matches = face_recognition.compare_faces([user_face_encoding], face_encoding, tolerance = 0.6)
                
                name = "Unknown"
                if True in matches:
                    name = user_names[i]
                    if mode == 'r':
                        print(f"User recognized: {name}")
                    elif mode == 'a':
                        print(f"User authorized: {name}")
                        authorized = True

                i+=1
            face_names.append(name)
        
        display(face_locations, face_names)
                    
    process_this_frame = not process_this_frame

    # Hit 'q' on the keyboard to quit!
    if (cv2.waitKey(1) & 0xFF == ord('q')) or authorized:
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
print("Released webcam and destroyed all windows.")