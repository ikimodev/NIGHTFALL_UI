import os
import cv2
import threading
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from uv_drone_prototype import process_frame

app = Flask(__name__)
CORS(app) # Allow React to fetch data

# Ensure uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for sharing data between video thread and web endpoints
output_frame = None
latest_messages = []
lock = threading.Lock()

# Video source controls
video_source = 0 # Default to webcam
source_changed = False

def capture_video():
    global output_frame, latest_messages, video_source, source_changed
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")

    while True:
        if source_changed:
            cap.release()
            print(f"Switching video source to: {video_source}")
            cap = cv2.VideoCapture(video_source)
            source_changed = False
            
        ret, frame = cap.read()
        if not ret:
            # If it's a video file and it reached the end, loop it
            if isinstance(video_source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                continue
            
        # Process the frame through the prototype pipeline
        proc_frame, mask, messages = process_frame(frame)
        
        with lock:
            output_frame = proc_frame.copy()
            if messages:
                latest_messages = messages

def generate_video_stream():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def get_logs():
    with lock:
        return jsonify(latest_messages)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_source, source_changed
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file:
            filename = secure_filename(file.filename)
            if not filename:
                # Fallback if secure_filename removes all characters (e.g., non-ASCII names)
                filename = "uploaded_video.mp4"
                
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Switch the video source to the uploaded file
            with lock:
                video_source = filepath
                source_changed = True
                
            return jsonify({"message": "Video uploaded successfully", "filename": filename}), 200
    except Exception as e:
        print("Upload Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/use_webcam', methods=['POST'])
def use_webcam():
    global video_source, source_changed
    with lock:
        video_source = 0
        source_changed = True
    return jsonify({"message": "Switched to webcam"}), 200

if __name__ == '__main__':
    # Start the video capture in a separate thread so it runs continuously
    t = threading.Thread(target=capture_video, daemon=True)
    t.start()
    
    # Start the Flask web server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
