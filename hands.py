import cv2
import mediapipe as mp
import numpy as np
import time
import screeninfo
import hand_tracking as htm


def wrap_text(text, font, max_width):
    lines = []
    line = ''
    for word in text.split():
        if cv2.getTextSize(line + ' ' + word, font, fontScale=2.0, thickness=1)[0][0] <= max_width:
            line += ' ' + word
        else:
            lines.append(line.lstrip())
            line = word
    lines.append(line)
    return lines

def write_text_on_image(image, text, coordinates):
    image_height, image_width, _ = image.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 2
    font_color = (100, 255, 0)

    x, y = coordinates

    max_text_width = image_width - x 

    wrapped_lines = wrap_text(text, font, max_text_width)

    for line in wrapped_lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        cv2.putText(image, line, (x, y), font, font_scale, font_color, font_thickness)
        y += h + 10 

    return image


live = cv2.VideoCapture(0)

screen = screeninfo.get_monitors()[0]
screen_dims = (1000, 2000, 1)
screen_width, screen_height = 2000, 2000

tracker = htm.HandTracker()

cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
cv2.moveWindow("Right", screen_width//2, 0)
cv2.setWindowProperty("Right", cv2.WND_PROP_TOPMOST, 1)


cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
# cv2.moveWindow("Right", screen_width//2, 0)

prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0
image_canvas = None
xp =0
yp=0
curr_col = np.array([255,255,255])
while True:
    ret, img = live.read(0)
    if not ret:
        print("ERROR/ FINISHED. Exiting...")
        break


    left = np.zeros_like(img)
    imm = cv2.imread("paint2.png")
    left[:200, :] = imm
    left[210:, :] = curr_col
    # right = np.zeros_like(img)
    right = cv2.flip(img, 1)
    if image_canvas is None:
        image_canvas = np.zeros_like(right)

    lms, img, left, right = tracker.track_hands(img, left, right, draw=True)
    # print(lms)
    right_mode = "None"
    left_mode = "None"
    if "right" in lms:
        # lms = lms["right"]
        fingers = tracker.find_mode()
        if sum(fingers)==1 and fingers[1]:
            right_mode = "Drawing mode"
        elif sum(fingers)==2 and fingers[1]&fingers[2]:
            right_mode = "Eraser"
        else:
            right_mode = "Idle mode"

    if "left" in lms:
        fingers = tracker.find_mode()
        if sum(fingers)==1 and fingers[1]:
            left_mode = "Selection mode"
        elif sum(fingers)==2 and fingers[1]&fingers[2]:
            left_mode = "Eraser"
        else:
            left_mode = "Idle mode"

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)

    # cv2.putText(left, f"{fps}", (7, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    # cv2.putText(right, f"{fps} {right_mode}", (7, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)


    screen_height, screen_width, _ = screen_dims

    left = cv2.resize(left, (screen_width//2, screen_height))

    right = cv2.resize(right, (screen_width//2, screen_height))
    image_canvas = cv2.resize(image_canvas, (screen_width//2, screen_height))

    # x, y = 0,0
    if len(lms)>0 and lms["right"]:
        x, y = lms["right"][8][1:]
        x = int(x * (screen_width//2))
        y = int(y* (screen_height))
        if right_mode=="Drawing mode":
            cv2.circle(right, (x, y), 13, (255, 0, 255), cv2.FILLED)
            print(type(curr_col))
            cv2.line(right, (xp,yp),(x,y),color = curr_col.tolist(), thickness=6)
            cv2.line(image_canvas, (xp,yp),(x,y),color= curr_col.tolist(), thickness=6)
        elif right_mode=="Eraser":
            cv2.circle(right, (x, y), 27, (0, 0, 0), cv2.FILLED)
            cv2.line(right, (xp,yp),(x,y),color= (0, 0, 0), thickness = 80)
            cv2.line(image_canvas, (xp,yp),(x,y),color= (0, 0, 0), thickness = 80)
        xp , yp = x, y

    if len(lms)>0 and lms["left"]:
        x, y = lms["left"][8][1:]
        x = int(x * (screen_width//2))
        y = int(y* (screen_height))
        if left_mode=="Selection mode":
            cv2.circle(left, (x, y), 13, (255, 0, 255), 8)
            try:
                print("COLOR : ", left[y][x], "CURR : ", x, y)
                curr_col = left[y][x]
            except:
                pass
    else:
        # cv2.putText(left, "Only use index finger of left hand to select color", (7, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        left = write_text_on_image(left, "Only use index finger of left hand to select color", (200, 600))

    img_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
    _, imginv= cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)
    right = cv2.bitwise_and(right, imginv)
    right =cv2.bitwise_or(right, image_canvas)

    resized_height, resized_width, _ = left.shape
    aspect_ratio = resized_width / resized_height

    window_width = int(screen_height * aspect_ratio)

    cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Left", window_width, screen_height)
    cv2.resizeWindow("Right", window_width, screen_height)

    cv2.imshow("Left", left)
    cv2.imshow("Right", right)


    if cv2.waitKey(10)==27:
        break
cv2.destroyAllWindows()
