import numpy as np
from sys import exit
from sys import stderr
from os import path
import cv2
import dlib

cascPath = path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def compress_image(image, k):
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    compressed_image = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
    return compressed_image

def vid_capture():
    try :
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Error: Could not open camera.", file=stderr)
        while True:
                ret, frame = video_capture.read()
                if not ret:
                    print("Error: Failed to capture frame from the camera.", file=stderr)
                    return
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(
                    gray_frame,
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                for (x, y, w, h) in faces:
                    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                    landmarks = predictor(gray_frame, dlib_rect)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    frame[y:y + h, x:x + w] = cv2.GaussianBlur(frame[y:y + h, x:x + w], (23, 23), 30)

                    for i in range(68):
                        x_lm = landmarks.part(i).x
                        y_lm = landmarks.part(i).y
                        cv2.circle(frame, (x_lm, y_lm), 1, (0, 0, 255), -1)

                cv2.imshow('Video', cv2.flip(frame, 1))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
         print(f"Error: an error occurred: {e}", file=stderr)
    finally:
        if 'video_capture' in locals() and video_capture.isOpened():
            video_capture.release()
        cv2.destroyAllWindows()

def main():
    vid_capture()

if __name__ == "__main__":
    main()
