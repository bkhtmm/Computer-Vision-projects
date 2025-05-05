import cv2
import mediapipe as mp
import pyautogui
import math
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

screen_width, screen_height = pyautogui.size()

class HandMouseController:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        self.hands = mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.prev_time = 0
        self.scroll_sensitivity = 20  
        
        self.thumb_tip_id = 4
        self.index_tip_id = 8
        self.middle_tip_id = 12
        self.middle_mcp_id = 9  

    def process_frame(self, frame):
        """ Обрабатывает один кадр, возвращает при необходимости команды для мыши. """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return frame

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            landmark_list = hand_landmarks.landmark

            index_tip = landmark_list[self.index_tip_id]
            thumb_tip = landmark_list[self.thumb_tip_id]
            middle_tip = landmark_list[self.middle_tip_id]
            middle_mcp = landmark_list[self.middle_mcp_id]

            frame_height, frame_width, _ = frame.shape

            index_x = int(index_tip.x * frame_width)
            index_y = int(index_tip.y * frame_height)

            screen_x = int(index_tip.x * screen_width)
            screen_y = int(index_tip.y * screen_height)

            pyautogui.moveTo(screen_x, screen_y)

            pinch_distance = self.distance_2d(thumb_tip, index_tip, frame_width, frame_height)
            if pinch_distance < 30:
                pyautogui.click()
                time.sleep(0.2)  

            if (middle_tip.y < middle_mcp.y) and (pinch_distance > 50):
                dist_mid_index = self.distance_2d(middle_tip, index_tip, frame_width, frame_height)
                if dist_mid_index > 50:  # пороговое значение
                    pyautogui.rightClick()
                    time.sleep(0.2)

            center_frame_y = frame_height // 2
            if index_y < center_frame_y - 50:
                pyautogui.scroll(self.scroll_sensitivity)
            elif index_y > center_frame_y + 50:
                pyautogui.scroll(-self.scroll_sensitivity)

        return frame

    @staticmethod
    def distance_2d(a, b, width, height):
        """ 
        Вычисление 2D-расстояния между точками a и b.
        a, b - объекты landmarks с x,y в диапазоне [0..1].
        width, height - реальные размеры кадра.
        """
        return math.hypot((a.x - b.x)*width, (a.y - b.y)*height)


def main():
    cap = cv2.VideoCapture(0)

    controller = HandMouseController()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        processed_frame = controller.process_frame(frame)

        cv2.imshow('Hand Mouse Control', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
