import cv2
import numpy as np
import time
import os
import winsound
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model

# Initialize Flask App
app = Flask(__name__)

# Load Trained Model
model = load_model("model/mask_detector.h5")

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Alert Settings
last_alert_time = 0
alert_interval = 5  # seconds between alerts

if not os.path.exists("alerts"):
    os.makedirs("alerts")

# Video Frame Generator
def generate_frames():
    global last_alert_time

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        status = "No Face"

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))
            face = face.astype("float32") / 255.0
            face = np.reshape(face, (1, 100, 100, 3))

            prediction = model.predict(face, verbose=0)
            label = np.argmax(prediction)

            current_time = time.time()

            # MASK DETECTED
            if label == 0:
                status = "Mask Detected"
                color = (8, 105, 48) # dark green "#086930" 
                
            # NO MASK DETECTED
            else:
                status = "No Mask"
                color = (0, 0, 240)  # Soft red 

                # Prevent alert spam
                if current_time - last_alert_time > alert_interval:

                    # Beep Alert (Windows)
                    winsound.Beep(2000, 300)
                    winsound.Beep(1500, 300)
                    winsound.Beep(1000, 300)

                    # Save Screenshot
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"alerts/alert_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)

                    print(f"🚨 ALERT! No Mask Detected. Screenshot saved: {filename}")

                    last_alert_time = current_time

            # Draw Bounding Box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, status, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

        # Top Status Banner
        if status == "No Mask":
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 240), -1)
            cv2.putText(frame, "WARNING: MASK NOT DETECTED!",
                        (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)

        elif status == "Mask Detected":
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (8, 105, 48), -1)
            cv2.putText(frame, "Mask Detected",
                        (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)

        # Encode Frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Run App
if __name__ == "__main__":
    app.run(debug=True)

    


'''import cv2
import numpy as np
import time
import os
import winsound
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model

# Flask App Initialization
app = Flask(__name__)

# Load Model
model = load_model("model/mask_detector.h5")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Webcam Setup
camera = cv2.VideoCapture(0)

# Alert Control
last_alert_time = 0
alert_interval = 5  # seconds

# Frame Generator
def generate_frames():
    global last_alert_time

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        status = "Detecting..."
        color = (200, 200, 200)

        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))
            face = face.astype("float32") / 255.0
            face = np.reshape(face, (1, 100, 100, 3))

            prediction = model.predict(face, verbose=0)
            label = np.argmax(prediction)

            current_time = time.time()

            # MASK DETECTED
            if label == 0:
                status = "Mask Detected"
                color = (46, 204, 113)  # soft green

            # NO MASK
            else:
                status = "No Mask Detected"
                color = (231, 76, 60)  # soft red

                if current_time - last_alert_time > alert_interval:

                    print("🚨 ALERT: No Mask Detected")

                    # Sound Alert (Windows only)
                    try:
                        winsound.Beep(2000, 300)
                        winsound.Beep(1500, 300)
                        winsound.Beep(1000, 300)
                    except:
                        pass

                    # Save Screenshot
                    if not os.path.exists("alerts"):
                        os.makedirs("alerts")

                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"alerts/alert_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)

                    print(f"📸 Screenshot saved: {filename}")

                    last_alert_time = current_time

            # Draw Face Box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            cv2.putText(frame, status, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        # Top Status Banner (Aesthetic)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60),
                      (15, 23, 42), -1)

        cv2.putText(frame, "Real-Time Face Mask Monitoring",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (200, 200, 200), 2)

        # Encode Frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Run App
if __name__ == "__main__":
    app.run(debug=True)
'''




# import cv2
# import numpy as np
# from flask import Flask, render_template, Response
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load trained model
# model = load_model("model/mask_detector.h5")

# # Load Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# def generate_frames():
#     camera = cv2.VideoCapture(0)

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#             for (x, y, w, h) in faces:
#                 face = frame[y:y+h, x:x+w]
#                 face = cv2.resize(face, (100, 100))
#                 face = face / 255.0
#                 face = np.reshape(face, (1, 100, 100, 3))

#                 prediction = model.predict(face)
#                 label = np.argmax(prediction)

#                 if label == 0:
#                     text = "Mask Detected"
#                     color = (0, 255, 0)
#                 else:
#                     text = "No Mask Alert!"
#                     color = (0, 0, 255)

#                 cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#                 cv2.putText(frame, text, (x, y-10),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.9, color, 2)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     camera.release()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video')
# def video():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True)
