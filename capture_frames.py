import cv2
import os

def video_to_frames(path1, path2):
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame index
    vidcap = cv2.VideoCapture(1)
    vidcap2 = cv2.VideoCapture(0)
    count = 0
    while vidcap.isOpened() and vidcap2.isOpened():
        success, image = vidcap.read()
        success2, image2 = vidcap2.read()
        image = cv2.flip(image, -1)
        image2 = cv2.flip(image2, -1)
        if success and success2:
            cv2.imwrite(os.path.join(path1, '%d.png') % count, image)
            cv2.imwrite(os.path.join(path2, '%d.png') % count, image2)
            count += 1
                    
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()

video_to_frames('../StereoCam/left_pngs', '../StereoCam/right_pngs')
