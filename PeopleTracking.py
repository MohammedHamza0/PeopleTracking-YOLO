import torch
import numpy as np
import cv2
from tracker import *

cap = cv2.VideoCapture("cctv.mp4")
area = np.array([[490, 559], [451, 501], [703, 420], [746, 458]])
targetLable = ["person"]

tracker = Tracker()

model = torch.hub.load("ultralytics/yolov5", 'yolov5l')

count = set()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't read the frame.")
        break
    
    else:
        points = []
        frame = cv2.resize(frame, (1100, 700))
        result = model(frame)
        cv2.polylines(frame, [area], True, [0, 0, 255], 2)
        # cv2.putText(frame, "ROI", (451, 501),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, [150, 150, 255], 2)
        for index, row in result.pandas().xyxy[0].iterrows():
            xmin = int(row["xmin"])
            ymin = int(row["ymin"])
            xmax = int(row["xmax"])
            ymax = int(row["ymax"])
            cx = int((xmax+xmin)/2)
            cy = int((ymax+ymin)/2)
            label = row["name"]
            if label in targetLable:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [0, 255, 255], 2)  
                
                # Add shadow for the text
                cv2.putText(frame, "Person", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0, 0, 0], 3)  # Shadow (black)
                cv2.putText(frame, "Person", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, [0, 255, 0], 2)  # Main text (green)
                
                cv2.circle(frame, (cx, cy), 5, [150, 150, 255], -1)  

                points.append([xmin, ymin, xmax, ymax])
                
        persons_id = tracker.update(points)
        for personID in persons_id:
            x , y , w , h, ID = personID
            check = cv2.pointPolygonTest(area, (w, h), False)
            if check >= 0:
                count.add(ID)
                
        counts = len(count)

        # Add shadow for the counter text
        cv2.putText(frame, f"Number of pedestrians {counts}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 3)  # Shadow (black)
        cv2.putText(frame, f"Number of pedestrians {counts}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2)  # Main text (red)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            break
        
cap.release()
cv2.destroyAllWindows()

    