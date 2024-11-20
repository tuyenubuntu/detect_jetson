import threading
import cv2
import jetson.inference
import jetson.utils
import os
import time
from detection import DetectionAttribute


class Stream(threading.Thread):
    def __init__(self, fps=30):
        threading.Thread.__init__(self)
        self.fps = fps

        # Đường dẫn tới model TensorRT
        model_path = os.path.join('model', 'ssd_mobilenet_v2.engine')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Tải mô hình sử dụng Jetson Inference
        self.detection_net = jetson.inference.detectNet(argv=[
            '--model=' + model_path,
            '--labels=model/labels.txt',
            '--input-blob=input_0',
            '--output-cvg=scores',
            '--output-bbox=boxes'
        ], threshold=0.5)

        # Cấu hình camera
        self.camera = jetson.utils.gstCamera(640, 480, '/dev/video0')  # Thay đổi thiết bị nếu cần
        self.frame = None
        self.lock = threading.Lock()
        self.detected_objects = []

    def run(self):
        """Luồng xử lý video"""
        while True:
            # Lấy frame từ camera
            frame_rgba = self.camera.CaptureRGBA(zeroCopy=1)
            frame_bgr = cv2.cvtColor(jetson.utils.cudaToNumpy(frame_rgba), cv2.COLOR_RGBA2BGR)

            # Xử lý phát hiện đối tượng
            self.process(frame_bgr)

            # Vẽ bounding boxes và hiển thị thông tin
            self.draw_bounding_boxes(frame_bgr)

            # Cập nhật frame đã xử lý
            with self.lock:
                self.frame = frame_bgr

            # Điều chỉnh FPS
            time.sleep(1 / self.fps)

    def process(self, frame):
        """Phát hiện đối tượng và cập nhật danh sách"""
        cuda_img = jetson.utils.cudaFromNumpy(frame)
        detections = self.detection_net.Detect(cuda_img)
        self.detected_objects = []

        for detection in detections:
            obj = DetectionAttribute(
                _class=detection.ClassID,
                center=(detection.Center[0], detection.Center[1]),
                width=detection.Width,
                height=detection.Height,
                confidence=detection.Confidence
            )
            self.detected_objects.append(obj)

    def draw_bounding_boxes(self, frame):
        """Vẽ bounding boxes và thông tin detection lên frame"""
        for obj in self.detected_objects:
            # Vẽ bounding box
            start_point = (int(obj.Center[0] - obj.Width / 2), int(obj.Center[1] - obj.Height / 2))
            end_point = (int(obj.Center[0] + obj.Width / 2), int(obj.Center[1] + obj.Height / 2))
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

            # Thêm thông tin status
            label = f"Class: {obj.ClassID}, Conf: {obj.Confidence:.2f}"
            cv2.putText(frame, label, (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def get_frame(self):
        """Trả về frame đã được xử lý"""
        with self.lock:
            if self.frame is not None:
                ret, jpeg = cv2.imencode('.jpg', self.frame)
                if ret:
                    return (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
