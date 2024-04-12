import tkinter as tk
import cv2
import numpy as np
import dlib
from imutils import face_utils
from PIL import ImageTk, Image

class DrowsinessDetectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.cap = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.blink_count = 0
        self.blink_start_time = None
        self.eye_closure_duration = 0
        self.active, self.drowsy, self.sleep = 0, 0, 0
        self.status = ""
        self.color = (0, 0, 0)

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.btn_quit = tk.Button(window, text="Quit", command=self.window.quit)
        self.btn_quit.pack(anchor=tk.SE, padx=10, pady=10)

        self.update()

    def eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def update(self):
        _, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray)
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = self.predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]

            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < 0.25:
                if self.blink_start_time is None:
                    self.blink_start_time = current_time
                self.eye_closure_duration = current_time - self.blink_start_time

                if self.eye_closure_duration > 1.0:  # Checking if eye has been closed for over 1 second
                    self.sleep += 1
                    self.status = "SLEEPING !!!"
                    self.color = (255, 0, 0)
                elif self.eye_closure_duration > 0.2:  # Checking if eye closure is long enough to consider drowsy
                    self.drowsy += 1
                    self.status = "Drowsy !"
                    self.color = (0, 0, 255)
            else:
                if self.blink_start_time is not None:
                    self.blink_count += 1
                self.blink_start_time = None
                self.eye_closure_duration = 0
                self.drowsy = 0
                self.sleep = 0
                self.active += 1
                self.status = "Active :)"
                self.color = (0, 255, 0)

            cv2.putText(frame, self.status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.color, 3)

            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        self.photo = self.convert_img_to_photo(frame)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(10, self.update)

    def convert_img_to_photo(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        photo = ImageTk.PhotoImage(image=img)
        return photo

if __name__ == "__main__":
    window = tk.Tk()
    app = DrowsinessDetectorApp(window, "Drowsiness Detector")
    window.mainloop()
