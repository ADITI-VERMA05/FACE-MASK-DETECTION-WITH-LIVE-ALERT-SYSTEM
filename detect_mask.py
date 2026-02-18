import cv2
import numpy as np
import os
import time
import smtplib
import winsound
from email.mime.text import MIMEText
from tensorflow.keras.models import load_model


# Load Model
model = load_model("model/mask_detector.h5")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Email Alert Function
def send_email_alert():
    sender_email = "aditiverma1605@gmail.com"
    receiver_email = "itz.la.mani04@gmail.com"
    password = "1234567890"  # ⚠ Use App Password (not real password)

    msg = MIMEText("ALERT: No Mask Detected!")
    msg["Subject"] = "Face Mask Violation Alert"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("📧 Email alert sent!")
    except Exception as e:
        print("Email failed:", e)

# Start Webcam
cap = cv2.VideoCapture(0)

last_alert_time = 0
alert_interval = 5  # seconds

print("System Started... Press 'q' to exit")

# Real-Time Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    no_mask_detected = False  # flag

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face = face.astype("float32") / 255.0
        face = np.reshape(face, (1, 100, 100, 3))

        prediction = model.predict(face, verbose=0)
        label = np.argmax(prediction)

        if label == 0:
            text = "Mask Detected"
            color = (0, 255, 0)
        else:
            text = "NO MASK - ALERT!"
            color = (0, 0, 255)
            no_mask_detected = True

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

    # Global Alert System
    current_time = time.time()

    if no_mask_detected:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
        cv2.putText(frame, "WARNING: MASK NOT DETECTED!",
                    (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        if current_time - last_alert_time > alert_interval:

            # Sound Alert
            winsound.Beep(2000, 300)
            winsound.Beep(1500, 300)
            winsound.Beep(1000, 300)

            # Save Screenshot
            if not os.path.exists("alerts"):
                os.makedirs("alerts")

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"alerts/alert_{timestamp}.jpg"
            cv2.imwrite(filename, frame)

            print(f"🚨 ALERT! No Mask Detected. Screenshot saved: {filename}")

            # Optional Email
            # send_email_alert()

            last_alert_time = current_time

    cv2.imshow("Face Mask Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()