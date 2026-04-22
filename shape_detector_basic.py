import cv2
import numpy as np
import time
import json

# ==========================================
# Phase 1: Basic Shape Detection (Normal Light)
# Focuses ONLY on finding geometry, no UV yet
# ==========================================

SHAPE_TO_COMMAND = {
    "TRIANGLE": "MEDICAL_ASSISTANCE",
    "CIRCLE": "FRIENDLY_POSITION",
    "SQUARE": "EXTRACTION_REQUEST",
    "UNKNOWN": "UNVERIFIED_SIGNAL"
}

def detect_shape(contour):
    """
    Classical geometry reasoning to classify shapes.
    """
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return "UNKNOWN"
        
    # Approximate polygon
    epsilon = 0.04 * peri
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    
    # Calculate circularity for circle detection
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * (area / (peri * peri))
    
    if vertices == 3:
        return "TRIANGLE"
    elif vertices == 4:
        # compute aspect ratio to distinguish square from rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.85 <= aspect_ratio <= 1.15:
            return "SQUARE"
        else:
            return "RECTANGLE"
    elif circularity > 0.75: # Roughly circular tolerance
        return "CIRCLE"
    else:
        return "UNKNOWN"

def process_frame(frame, simulated_gps=[24.7136, 46.6753]):
    """
    Core Perception Pipeline for Normal Lighting:
    Grayscale -> Blur -> Edge Detection -> Classification
    """
    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Blur to remove normal camera noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Canny Edge Detection (Finds lines/shapes in normal light instead of UV thresholding)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 4. Dilate edges slightly to close gaps in lines drawn on paper
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # 5. Extract Contours
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    messages = []
    output_frame = frame.copy()
    
    for cnt in contours:
        # Ignore small noise blobs (increased size for normal lighting)
        if cv2.contourArea(cnt) < 1500:
            continue
            
        shape = detect_shape(cnt)
        
        if shape != "UNKNOWN":
            command = SHAPE_TO_COMMAND.get(shape, "UNKNOWN")
            
            packet = {
                "timestamp": round(time.time(), 2),
                "symbol": shape,
                "meaning": command,
                "confidence": 0.85,
                "GPS": simulated_gps
            }
            messages.append(packet)
            
            # Draw on frame
            cv2.drawContours(output_frame, [approx_shape(cnt)], -1, (0, 255, 0), 3)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output_frame, f"{shape}", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            
    return output_frame, edges_dilated, messages

def approx_shape(contour):
    """Helper purely for drawing a cleaner bounding box/polygon on the screen"""
    peri = cv2.arcLength(contour, True)
    epsilon = 0.04 * peri
    return cv2.approxPolyDP(contour, epsilon, True)

def run_test_camera():
    print("====================================")
    print("NIGHTFALL: BASIC SHAPE DETECTOR NODE")
    print("====================================")
    print("-> Press 'q' to Quit.")
    print("-> Draw a thick triangle, square, or circle on paper and show it to the camera.\n")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        processed_frame, edges, messages = process_frame(frame)
        
        for msg in messages:
            print(json.dumps(msg))
            print("-" * 30)
            
        cv2.imshow("1. Raw Feed", frame)
        cv2.imshow("2. Edge Detection (What the AI sees)", edges)
        cv2.imshow("3. Output", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test_camera()
