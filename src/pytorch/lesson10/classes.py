from abc import ABC, abstractmethod
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import GestureRecognitionModel


class FrameHandler(ABC):
    @abstractmethod
    def process_frame(self, frame):
        return frame


# Открывает видео поток, обрабатывает кадры вызывая указанный FrameHandler
class VideoStreamer:
    def run(self, handler: FrameHandler):

        plt.ion()  # Pyplot interactive mode ON
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            self.run_video_stream(cap, handler)
        else:
            print('Cannot open video capture! Aborting.')
        cap.release()
        plt.close()

    def run_video_stream(self, cap, frame_handler: FrameHandler):
        window_is_open = True

        def on_close(event):
            print('Window close event encountered', event)
            nonlocal window_is_open
            window_is_open = False

        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('close_event', on_close)

        while cap.isOpened() and window_is_open:
            ret, frame = cap.read()

            if not ret:
                break

            frame = frame_handler.process_frame(frame)

            # Конвертируем изображение из BGR в RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Отображаем изображение с помощью matplotlib
            plt.imshow(frame)
            plt.draw()
            plt.pause(.01)
            fig.canvas.flush_events()


# Обрабатывает кадры видео: при наличии лица человеаа извлекает изображение правой руки (слева от лица на кадре)
# и распознает жест. Выводит жест в консоль.
# Жесты распознаются моделью, которая обучается отдельно (исходник в отдельном ноутбуке в Колаб).
class GestureHandler(FrameHandler):
    def __init__(self):
        use_cuda = False
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=500, varThreshold=24)

        model = GestureRecognitionModel()
        model.load_state_dict(torch.load('data/gestures_weights.pth'))
        self.gestureModel = model.to(self.device)

        self.prev_gesture = None

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        maybe_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        frame_w, frame_h = frame.shape[1], frame.shape[0]
        # Отберем только те лица, которые размером хотя бы в четверть кадра
        faces = [face for face in maybe_faces if face[3] / frame_h > 0.25]
        face = faces[0] if faces else None

        if face is not None:
            # Получим маску для выделения переднего плана
            fg_mask = self.backSub.apply(frame)

            # Возьмем часть картинки слева от лица (где-то там должна быть правая рука, учитывая зеркальность)
            hand = frame[0:frame_h, 0:face[0]]  # frame[y:y+h, x:x+w]
            hand_mask = fg_mask[0:frame_h, 0:face[0]]

            # Подготовим картинку с рукой
            # оставим передний план и добавим блюр
            hand = cv2.bitwise_and(hand, hand, mask=hand_mask)
            # Наложим подготовленную руку поверх исходного изображения для мониторинга
            frame[0:frame_h, 0:face[0]] = hand
            hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
            hand = cv2.blur(hand, (5, 5))
            gesture = self.try_detect_gesture(hand)
            if gesture:
                if self.prev_gesture is None or \
                        gesture.class_id != self.prev_gesture.class_id or \
                        gesture.probability > self.prev_gesture.probability:
                    self.prev_gesture = gesture
                    print(f"Распознан жест: {gesture.class_id} {gesture.class_name} {gesture.probability}")

            # Рисуем прямоугольники вокруг обнаруженных лиц. Первое лицо обводим жирнее,
            # так как мы считаем именно его главным.
            first_face = True
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3 if first_face else 1)
                first_face = False
            pass
        return frame

    def try_detect_gesture(self, image):
        image = cv2.resize(image, (128, 128))
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device).float()
        pred = self.gestureModel(tensor[None, ...])
        probabilities = F.softmax(pred, dim=1).squeeze(0).tolist()

        gesture = int(pred.argmax().item())
        prob = probabilities[gesture]

        class_names = {
            0: "Palm",
            1: "L",
            2: "Fist",
            3: "Fist_moved",
            4: "Thumb",
            5: "Index",
            6: "OK",
            7: "Palm_moved",
            8: "C",
            9: "Down"
        }
        return Gesture(class_id=gesture, class_name=class_names.get(gesture), probability=prob)


class Gesture:
    def __init__(self, class_id, class_name, probability):
        self.class_id = class_id
        self.class_name = class_name
        self.probability = probability