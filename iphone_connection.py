import cv2

def main():
    droidcam_url = "http://192.168.8.99:9372/video"

    cap = cv2.VideoCapture(droidcam_url)

    if not cap.isOpened():
        print(f"Error: Could not open video stream from {droidcam_url}")
        return

    print(f"Successfully connected to {droidcam_url}")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame. Exiting...")
            break
        cv2.imshow("DroidCam iPhone Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
