import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize Pycaw for volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Calibration
min_hand_dist = 30
max_hand_dist = 220

# Webcam
cap = cv2.VideoCapture(0)
# Set camera resolution (try 1280x720 or 1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


vol_bar = 400
vol_per = 0
p_time = 0
mute_state = False
last_action_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    lm_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if lm_list:
        # Thumb tip (4) & Index tip (8) for volume control
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw the connection
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        # Volume
        length = math.hypot(x2 - x1, y2 - y1)
        vol_per = np.interp(length, [min_hand_dist, max_hand_dist], [0, 100])
        vol_bar = np.interp(length, [min_hand_dist, max_hand_dist], [400, 150])
        volume.SetMasterVolumeLevelScalar(vol_per / 100, None)

        # Circle changes color when very close
        if length < min_hand_dist + 10:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        # Detect Mute/Unmute using Pinky (20 tip vs 17 MCP)
        pinky_tip, pinky_mcp = lm_list[20][2], lm_list[17][2]
        if pinky_tip < pinky_mcp and not mute_state:
            volume.SetMute(1, None)
            mute_state = True
            print("Muted")
        elif pinky_tip >= pinky_mcp and mute_state:
            volume.SetMute(0, None)
            mute_state = False
            print("Unmuted")

        # Action cooldown to avoid multiple triggers
        current_time = time.time()

        # Play using Middle finger (12 tip vs 9 MCP)
        middle_tip, middle_mcp = lm_list[12][2], lm_list[9][2]
        if middle_tip < middle_mcp and current_time - last_action_time > 1:
            pyautogui.press("playpause")
            last_action_time = current_time
            print("Play triggered")

        # Pause using Ring finger (16 tip vs 13 MCP)
        ring_tip, ring_mcp = lm_list[16][2], lm_list[13][2]
        if ring_tip < ring_mcp and current_time - last_action_time > 1:
            pyautogui.press("playpause")
            last_action_time = current_time
            print("Pause triggered")

    # Volume Bar
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f"{int(vol_per)} %", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # FPS Counter
    c_time = time.time()
    fps = 1 / (c_time - p_time) if c_time != p_time else 0
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

    cv2.imshow("Hand Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
