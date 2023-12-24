import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import get_parking_spots_bboxes,empty_or_not

path="C:\\Users\\chait\\parking-space-detector-and-counter\\data\\parking_1920_1080_loop.mp4"
mask="mask_1920_1080.png"

cap=cv2.VideoCapture(path)
mask=cv2.imread(mask,0)

connected_components= cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)

spots=get_parking_spots_bboxes(connected_components)

ret=True
step=30 
spots_status=[None for j in spots]
frame_nmr=0

diffs=[None for j in spots]
previous_frame=None
def calc_diff(im1,im2):
    return np.abs(np.mean(im1)-np.mean(im2))

while ret:
    ret,frame=cap.read()
    if frame_nmr%step==0 and previous_frame is not None: 

        for spot_index,spot  in enumerate(spots):
            x1,y1,w,h=spot
            spot_crop=frame[y1:y1+h,x1:x1+w,:] 
            diffs[spot_index]=calc_diff(spot_crop,previous_frame)

    if frame_nmr%step==0: 

        for spot_index,spot  in enumerate(spots):
            x1,y1,w,h=spot
            spot_crop=frame[y1:y1+h,x1:x1+w,:]  
            #print(spot_crop)

            spot_status=empty_or_not(spot_crop)
            spots_status[spot_index]=spot_status 
    if frame_nmr%step==0:  
        previous_frame=frame.copy()
    spots_status_numeric = [0 if status is None else status for status in spots_status]

    for spot_index,spot  in enumerate(spots):   
        spot_status=spots_status[spot_index] 
        x1,y1,w,h=spots[spot_index]
        if spot_status:
            frame=cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2)
        else:
            frame=cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,0,255),2)
        
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status_numeric)), str(len(spots_status))), (100, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.imshow("frame",frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr+=1

cap.release()
cv2.destroyAllWindows()