import yaml
import numpy as np
import cv2

# -----------------------------------------------------------------------------------------------------------------
#       File Paths
# -----------------------------------------------------------------------------------------------------------------
video_path = r"C:\Users\sonalika bisht\Desktop\project\carPark.mp4"
yaml_path = r"C:\Users\sonalika bisht\Desktop\project\auto_generated_spots.yml"

config = {
    'text_overlay': True,
    'parking_overlay': True,
    'parking_id_overlay': True,
    'parking_detection': True,
    'min_area_motion_contour': 60,
    'park_sec_to_wait': 80
}

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Failed to open video.")
    exit()

video_info = {
    'fps': cap.get(cv2.CAP_PROP_FPS),
    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
    'num_of_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
}
print("✅ Video Opened")
print("Total Frames:", video_info['num_of_frames'])

# Safe frame start
start_frame = 1000
if video_info['num_of_frames'] > start_frame:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
else:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# -----------------------------------------------------------------------------------------------------------------
#       Rescale Function
# -----------------------------------------------------------------------------------------------------------------
def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

# -----------------------------------------------------------------------------------------------------------------
#       Load Parking Spot Coordinates
# -----------------------------------------------------------------------------------------------------------------
with open(yaml_path, 'r') as stream:
    parking_data = yaml.load(stream, Loader=yaml.FullLoader)

parking_contours = []
parking_bounding_rects = []
parking_mask = []

for park in parking_data:
    points = np.array(park['points'])
    rect = cv2.boundingRect(points)
    points_shifted = points.copy()
    points_shifted[:, 0] -= rect[0]
    points_shifted[:, 1] -= rect[1]
    parking_contours.append(points)
    parking_bounding_rects.append(rect)
    mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], -1, 255, -1, cv2.LINE_8)
    mask = mask == 255
    parking_mask.append(mask)

parking_status = [False] * len(parking_data)
parking_buffer = [None] * len(parking_data)

# -----------------------------------------------------------------------------------------------------------------
#       Video Loop
# -----------------------------------------------------------------------------------------------------------------
video_cur_frame = 0
video_cur_pos = 0
errorcolor = []

while cap.isOpened():
    spot = 0
    occupied = 0
    video_cur_pos += 1
    video_cur_frame += 1

    ret, frame = cap.read()
    if not ret:
        print("❌ Capture Error")
        break

    frame_blur = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
    frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    frame_out = frame.copy()

    # Parking Detection
    if config['parking_detection']:
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            rect = parking_bounding_rects[ind]
            roi_gray = frame_gray[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            points[:, 0] -= rect[0]
            points[:, 1] -= rect[1]
            status = np.std(roi_gray) < 30 and np.mean(roi_gray) > 50
            if status != parking_status[ind] and parking_buffer[ind] is None:
                parking_buffer[ind] = video_cur_pos
            elif status != parking_status[ind] and parking_buffer[ind] is not None:
                if video_cur_pos - parking_buffer[ind] > config['park_sec_to_wait']:
                    parking_status[ind] = status
                    parking_buffer[ind] = None
            elif status == parking_status[ind] and parking_buffer[ind] is not None:
                parking_buffer[ind] = None

    # Draw Boxes
    if config['parking_overlay']:
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            if parking_status[ind]:
                color = (0, 255, 0)  # Green: Free
                spot += 1
            else:
                color = (0, 0, 255)  # Red: Occupied
                occupied += 1
            moments = cv2.moments(points)
            centroid = (int(moments['m10'] / moments['m00']) - 3, int(moments['m01'] / moments['m00']) + 3)
            cv2.drawContours(frame_out, [points], -1, color, 2)
            if config['parking_id_overlay']:
                cv2.putText(frame_out, str(park['id']), (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Overlay Info
    if config['text_overlay']:
        cv2.putText(frame_out, f"Frames: {video_cur_frame}/{video_info['num_of_frames']}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)
        cv2.putText(frame_out, f"Free: {spot} Occupied: {occupied}", (5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

    # Show Output
    frame_resized = rescale_frame(frame_out, percent=100)
    cv2.imshow('🚗 Spot Detection System', frame_resized)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        cv2.imwrite(f'frame{video_cur_frame}.jpg', frame_out)

cap.release()
cv2.destroyAllWindows()
