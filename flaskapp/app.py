from flask import Flask, render_template, Response
from stream import Stream

app = Flask(__name__)

# Khởi tạo luồng video xử lý AI
stream = Stream()

@app.route('/')
def index():
    """Trang chủ với video stream"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Endpoint cung cấp video stream"""
    return Response(stream.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Chạy Flask
    stream.start()  # Khởi chạy luồng video
    app.run(host='0.0.0.0', port=5000, debug=False)
