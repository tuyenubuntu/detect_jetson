class DetectionAttribute:
    def __init__(self, _class, center, width, height, confidence):
        """Khởi tạo đối tượng phát hiện"""
        self.ClassID = _class
        self.Center = center
        self.Width = width
        self.Height = height
        self.Confidence = confidence
