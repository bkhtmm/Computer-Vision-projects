import cv2
from deepface import DeepFace

def main():
    """
    Main function to run real-time emotion detection using webcam.
    """


    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting camera. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            try:
                detections = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
            except Exception as e:
                print(f"DeepFace error: {e}")
                cv2.putText(frame, "No face detected", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Real-Time Emotion Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            if isinstance(detections, list):
                if len(detections) > 0:
                    detection = detections[0]
                else:
                    detection = None
            else:
                detection = detections

            if detection is not None and 'region' in detection:
                region = detection['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                dominant_emotion = detection.get('dominant_emotion', 'N/A')

                cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            else:
                cv2.putText(frame, "No face detected", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Real-Time Emotion Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

if __name__ == '__main__':
    main()
