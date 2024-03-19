import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict


class HandTracker:
    def __init__(self, min_detection_confidence = 0.7, min_tracking_confidence = 0.7, max_num_hands = 2):
        self.mp_hands = mp.solutions.hands
        self.drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            model_complexity = 1,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence,
            max_num_hands = max_num_hands
        )
        self.lms = None
        self.processed_img = None
        self.finger_tip_id = [4, 8, 12, 16, 20]


    def track_hands(self, img, left, right, draw = False):
        flipped = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        self.processed_img = self.hands.process(img_rgb)
        mh_lms = self.processed_img.multi_hand_landmarks
        self.lms = dict()

        if mh_lms:
            hand_classes = self.processed_img.multi_handedness
            if len(hand_classes)==2:
                self.lms["right"] = self.finger_positions(mh_lms[0])
                self.lms["left"] = self.finger_positions(mh_lms[1])
                self.draw_landmarks([mh_lms[0]], right)
                self.draw_landmarks([mh_lms[1]], left)
            else:
                if MessageToDict(hand_classes[0])["classification"][0]["label"]=="Right":
                    self.draw_landmarks([mh_lms[0]], right)
                    self.lms["right"] = self.finger_positions(mh_lms[0])
                    self.lms["left"] = None
                else:
                    # self.draw_landmarks([mh_lms[0]], left)
                    self.lms["left"] = self.finger_positions(mh_lms[0])
                    self.lms["right"] = None
        return self.lms, img, left, right

    def draw_landmarks(self, landmarks, img):
        for i in landmarks:
            self.drawing.draw_landmarks(img, i, self.mp_hands.HAND_CONNECTIONS)
        return img

    def finger_positions(self, x):
        ret = []
        if x:
            myhand = x
            for ind, lm in enumerate(myhand.landmark):
                ret.append([ind, lm.x, lm.y])
        return ret


    def find_mode(self):
        fingers=[0,0,0,0,0]
        if self.lms and self.lms["right"]:
            # print(self.lms["right"])
            if (self.lms["right"][self.finger_tip_id[0]][1] < self.lms["right"][8][1]):
                fingers[0] = 1
            for i in range(1,5):
                if (self.lms["right"][self.finger_tip_id[i]][2] < self.lms["right"][self.finger_tip_id[i] - 2][2]):
                    fingers[i] = 1

        elif self.lms and self.lms["left"]:
            for i in range(1,5):
                if (self.lms["left"][self.finger_tip_id[i]][2] < self.lms["left"][self.finger_tip_id[i] - 2][2]):
                    fingers[i] = 1
        # print(fingers)
        return fingers
