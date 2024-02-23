import cv2
import mediapipe as mp
import numpy as np
import time
import screeninfo
from google.protobuf.json_format import MessageToDict


def draw_landmarks(landmarks, img):
    for i in landmarks:
        drawing.draw_landmarks(img, i, mp_hands.HAND_CONNECTIONS)

live = cv2.VideoCapture(0)

screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils


hands = mp_hands.Hands(
    model_complexity = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7,
    max_num_hands = 2
)

cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
cv2.moveWindow("Right", screen_width//2, 0)
cv2.setWindowProperty("Right", cv2.WND_PROP_TOPMOST, 1)

cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
# cv2.moveWindow("Right", screen_width//2, 0)

prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

while True:
    ret, img = live.read(0)
    if not ret:
        print("ERROR/ FINISHED. Exiting...")
        break

    flipped = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)



    processed_img = hands.process(img_rgb)

    mh_lms = processed_img.multi_hand_landmarks
    img = cv2.flip(img, 1)

    blank1 = np.zeros_like(img)
    blank2 = np.zeros_like(img)

    if mh_lms:
        # draw_landmarks(mh_lms, img)
        hand_classes = processed_img.multi_handedness
        print(len(mh_lms))
        if len(hand_classes)==2:
            draw_landmarks([mh_lms[0]], blank1)
            draw_landmarks([mh_lms[1]], blank2)
        else:
            if MessageToDict(hand_classes[0])["classification"][0]["label"]=="Right":
                draw_landmarks([mh_lms[0]], blank1)
            else:
                draw_landmarks([mh_lms[0]], blank2)
        # print(processed_img.multi_handedness)
        print("=="*10)

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)

    cv2.putText(blank1, f"{fps}", (7, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(blank2, f"{fps}", (7, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)


    screen_height, screen_width, _ = img.shape

    blank1 = cv2.resize(blank1, (screen_width//2, screen_height))
    blank2 = cv2.resize(blank2, (screen_width//2, screen_height))

    resized_height, resized_width, _ = blank1.shape
    aspect_ratio = resized_width / resized_height

    window_width = int(screen_height * aspect_ratio)

    cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Left", window_width, screen_height)
    cv2.resizeWindow("Right", window_width, screen_height)

    cv2.imshow("Left", blank1)
    cv2.imshow("Right", blank2)


    if cv2.waitKey(10)==27:
        break
cv2.destroyAllWindows()
