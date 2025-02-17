from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import imutils
import math

# HOG descriptor and person detector initialization
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# video capturing
cap = cv2.VideoCapture('test2.mp4')

# frame number
#frameNR = 0

# tracking variables
center_points_path = []
tracking_objects = {}
track_id = 0

while (cap.isOpened()):
    # capturing frame from video
    ret, frame = cap.read()

    if ret == 0:
        break

    #frameNR += 1

    center_points_cur_frame = []

    # resizing for faster working
    frame = cv2.resize(frame, (640, 480))

    # people detector, returning bounding boxes and confidence
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    # creating array with rectangle vertexes
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    # non max suppression function
    picked_boxes = non_max_suppression(boxes, probs=None, overlapThresh=0.6)

    # deleting redundant weights
    k = -1
    for i, box in enumerate(boxes):
        k = k + 1
        if box not in picked_boxes:
            weights = np.delete(weights, k)
            k = k - 1

    # drawing rectangles
    for i, (xA, yA, xB, yB) in enumerate(picked_boxes):
        # display the detected boxes in the colour picture
        if weights[i] < 0.3:
            continue

        elif weights[i] < 0.7 and weights[i] > 0.3:
            cx = int((xA + xB) / 2)
            cy = int((yA + yB) / 2)
            center_points_path.append((cx, cy))
            center_points_cur_frame.append((cx, cy))
            #cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
            cv2.rectangle(frame, (xA, yA), (xB, yB), (50, 127, 255), 2)
        elif weights[i] > 0.7:
            cx = int((xA + xB) / 2)
            cy = int((yA + yB) / 2)
            center_points_path.append((cx, cy))
            center_points_cur_frame.append((cx, cy))
            #cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # centroid tracking algorithm
    tracking_objects_copy = tracking_objects.copy()
    for person_id, pt2 in tracking_objects_copy.items():
        person_tracked = False
        for pt in center_points_cur_frame:
            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
            if distance < 20:
                tracking_objects[person_id] = pt
                person_tracked = True
                if pt in center_points_cur_frame:
                    center_points_cur_frame.remove(pt)
                continue
        # removing lost track
        if not person_tracked:
            tracking_objects.pop(person_id)
    # ID for new persons
    for pt in center_points_cur_frame:
        tracking_objects[track_id] = pt
        track_id += 1
    # drawing ID's
    for person_id, pt in tracking_objects.items():
        cv2.putText(frame, str(person_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    # removing too many center points path
    if len(center_points_path) > 200:
        for i in range(len(center_points_path) - 200):
            center_points_path.pop(0)

    # drawing center points path
    for ptp in center_points_path:
        cv2.circle(frame, ptp, 2, (0, 0, 255), -1)

    # framer number
    #cv2.putText(frame, str(frameNR), (0, 0 + 25), 0, 1, (255, 0, 0), 2)

    # displaying output
    cv2.imshow('Output', frame)
    # closing video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# realising capture and closing window
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
