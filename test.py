import cv2

live = cv2.VideoCapture(1)

while True:
    ret, img = live.read()

    if not ret:
        break
    cv2.imshow("LIVE", img)

    if cv2.waitKey(10)==27:
        break