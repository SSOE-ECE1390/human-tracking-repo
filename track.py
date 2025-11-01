import cv2
import yaml
from ultralytics import YOLO


if __name__ == '__main__':
    with open('parameters.yml','r') as file:
        parameters = yaml.safe_load(file)['try']
        device = parameters['device']
        imgsz = parameters['imgsz']
    # Load the YOLO11 model
    model = YOLO("yolo11l.pt")
    # Open the video file
    video_path = "soccer_noaudio.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.predict(frame,imgsz = imgsz, conf=0.3, device=device)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()