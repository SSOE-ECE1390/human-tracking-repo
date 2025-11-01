import os
import cv2
import numpy as np

if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    cap = cv2.VideoCapture('match.mp4')   # keep simple; relpath not needed
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {os.path.abspath('match.mp4')}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    minute = np.linspace(4,80,2000)

    time = minute*60
    targets = (time*fps).astype(np.int32)
    for mini, target in zip(minute,targets):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            raise IOError(f"Cannot read frame {target}")
        else:
            cv2.imwrite(f'figures/frame{mini}.png', frame)
            print(f'saved{target}')
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)           # wait until a key is pressed
    cv2.destroyAllWindows()  # close the window
    cap.release()
