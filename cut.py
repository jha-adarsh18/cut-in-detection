import cv2
from ultralytics import YOLO
import numpy as np
import statistics
import os
import time
import sys

detector = YOLO(r"C:\Users\jhaad\Downloads\best.pt")

def get_color(value):
    inverted_value = 1 - value
    blue = 0
    red = int(255 * inverted_value)
    green = int(255 * value)
    return (blue, green, red)

def normalize(val, min_val=0.5, max_val=2):
    return (val - min_val) / (max_val - min_val)

def calculate_distance(height, cls):
    real_height = {0: 1000, 1: 1600, 2: 4000}
    sensor_height = 3.02
    focal_length = 2.12
    frame_height = 1080
    distance = (focal_length * real_height[int(cls)] * frame_height) / (height * sensor_height)
    return distance / 1000

try:
    img_dir = sys.argv[1]
except IndexError:
    print("Please provide path to img_dir")
    print("\nUsage: python script.py <img_dir>")
    print("Example: python script.py ./images")
    sys.exit(1)

if img_dir[-1] == '/':
    img_dir = img_dir[:-1]

img_files = os.listdir(img_dir)

frame_w = 1920
frame_h = 1080
center_w = frame_w / 2 - 80
center_h = frame_h / 2
left_p1 = (int(center_w - frame_w * 0.33), frame_h)
left_p2 = (int(center_w - frame_w * 0.02), int(frame_h * 0.5))
right_p1 = (int(center_w + frame_w * 0.33), frame_h)
right_p2 = (int(center_w + frame_w * 0.02), int(frame_h * 0.5))
slope_left = abs((left_p2[1] - left_p1[1]) / (left_p2[0] - left_p1[0]))
slope_right = abs((right_p2[1] - right_p1[1]) / (right_p2[0] - right_p1[0]))

dist = dict()
vel_ang = dict()
alerts = dict()

cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

output_file = 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter(output_file, fourcc, 15.0, (frame_w, frame_h))

for img_file in img_files:
    start_time = time.time()
    frame = cv2.imread(f'{img_dir}/{img_file}')
    resized_img = cv2.resize(frame, (640, 384))
    try:
        results = detector.track(resized_img, conf=0.70, persist=True, iou=0.90)
    except np.linalg.LinAlgError:
        continue

    cv2.line(frame, left_p1, left_p2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(frame, right_p1, right_p2, (0, 0, 255), 2, cv2.LINE_AA)

    for res in results:
        bbox = res.boxes.xywhn
        bbox[:, 0] *= frame_w
        bbox[:, 2] *= frame_w
        bbox[:, 1] *= frame_h
        bbox[:, 3] *= frame_h

        ids = res.boxes.id
        classes = res.boxes.cls

        if ids is None:
            dist = dict()
            vel_ang = dict()
            continue

        temp_ids = list(dist.keys())

        for key in temp_ids:
            if key not in ids:
                dist.pop(key)
                if key in alerts:
                    del alerts[key]

        for j in range(bbox.shape[0]):
            x, y, w, h = bbox[j, :]

            if int(ids[j]) not in dist.keys():
                dist[int(ids[j])] = [0, 0]
                dist[int(ids[j])][1] = float(calculate_distance(h, classes[j])) - 1.5
                continue
            else:
                dist[int(ids[j])][0] = dist[int(ids[j])][1]
                dist[int(ids[j])][1] = float(calculate_distance(h, classes[j])) - 1.5

                v = abs(dist[int(ids[j])][1] - dist[int(ids[j])][0]) * 15

                if x + w / 2 < center_w:
                    m = (y - (h / 2) - left_p1[1]) / (x + (w / 2) - left_p1[0])
                    angle = float(((np.arctan(abs(m)) - (np.arctan(slope_left))) * 180 / np.pi))
                else:
                    m = (y - (h / 2) - right_p1[1]) / (x - (w / 2) - right_p1[0])
                    angle = float(((np.arctan(abs(m)) - (np.arctan(slope_right))) * 180 / np.pi))

                if int(ids[j]) not in vel_ang.keys():
                    vel_ang[int(ids[j])] = list()
                    vel_ang[int(ids[j])].append([v])
                    vel_ang[int(ids[j])].append([angle])

                else:
                    if len(vel_ang[int(ids[j])][0]) == 5:
                        vel = statistics.median(vel_ang[int(ids[j])][0])
                        angle_stdev = statistics.stdev(vel_ang[int(ids[j])][1])
                        ttc = dist[int(ids[j])][1] / vel

                        angle_diff = vel_ang[int(ids[j])][1][-1] - vel_ang[int(ids[j])][1][-2]

                        vel_ang[int(ids[j])][0].pop(0)
                        vel_ang[int(ids[j])][1].pop(0)

                        if ttc < 3 and angle_diff < 0 and angle_stdev > 0.5 and angle < 5:
                            if ttc < 0.8 and angle_stdev > 1.5:
                                alerts[int(ids[j])] = [int(x), int(y), int(w), int(h), ttc, 30, 1]
                            elif int(ids[j]) in alerts.keys() and alerts[int(ids[j])][6] == 1:
                                alerts[int(ids[j])] = [int(x), int(y), int(w), int(h), ttc, 30, 1]
                            elif dist[int(ids[j])][1] < 5:
                                alerts[int(ids[j])] = [int(x), int(y), int(w), int(h), ttc, 30, 0]

                        elif int(ids[j]) in alerts.keys():
                            alerts[int(ids[j])][0:5] = [int(x), int(y), int(w), int(h), ttc]

                    vel_ang[int(ids[j])][0].append(v)
                    vel_ang[int(ids[j])][1].append(angle)

    for key, value in alerts.items():
        font = cv2.FONT_HERSHEY_SIMPLEX
        if value[5] > 0:
            text = 'Alert'
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            textX = int(value[0] - (textsize[0] / 2))
            textY = int(value[1] + (textsize[1] / 2))

            if value[6] == 1:
                cv2.putText(frame, f'Alert', (textX, textY), font, 1, (0, 200, 255), 2, cv2.LINE_AA)

            pts = np.array([[value[0], value[1] + (value[3] / 2)], [value[0] - 25, value[1] + (value[3] / 2) + 20],
                            [value[0] + 25, value[1] + (value[3] / 2) + 20]], np.int32)
            cv2.fillPoly(frame, [pts], get_color(normalize(value[4])))
            value[5] -= 1
        else:
            del alerts[key]
            break

    video_out.write(frame)

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

video_out.release()
cv2.destroyAllWindows()
