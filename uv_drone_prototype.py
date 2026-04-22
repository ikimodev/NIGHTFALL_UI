# NIGHTFALL: UV-Fluorescent Distress Marker
# Phase 1: Prototype Perception Pipeline
# ==========================================

import cv2
import numpy as np
import time
import json

# 1. Semantic Translation Table (Ontology)
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
    # Calculate perimeter
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
    Core Perception Pipeline:
    UV Image -> Threshold -> Contours -> Classification -> JSON Output
    """
    # Step 1: Grayscale (Simulating UV band processing)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Thresholding (Extract bright fluorescent pixels)
    # Adjust threshold value (e.g., 200) based on actual lighting/UV intensity
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Step 3: Morphological operations to clean up noise (Simulating spray gaps)
    # This connects broken spray lines
    kernel = np.ones((5,5), np.uint8)
    clean_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Step 4: Extract Contours
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    messages = []
    output_frame = frame.copy()
    
    for cnt in contours:
        # Ignore small noise blobs
        if cv2.contourArea(cnt) < 500:
            continue
            
        # Detect shape
        shape = detect_shape(cnt)
        
        if shape != "UNKNOWN":
            # Map to command
            command = SHAPE_TO_COMMAND.get(shape, "UNKNOWN")
            
            # Generate structured output
            packet = {
                "timestamp": round(time.time(), 2),
                "symbol": shape,
                "meaning": command,
                "confidence": 0.85, # Simulated heuristic confidence
                "GPS": simulated_gps
            }
            messages.append(packet)
            
            # Draw on frame for visualization
            cv2.drawContours(output_frame, [cnt], -1, (0, 255, 0), 3)
            # Find centroid to put text
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output_frame, f"{shape}: {command}", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
    return output_frame, clean_mask, messages

def run_test_camera():
    """Run real-time via webcam"""
    print("====================================")
    print("NIGHTFALL DRONE PERCEPTION NODE LIVE")
    print("====================================")
    print("-> Press 'q' to Quit.")
    print("-> Hold up a very bright white shape (or use phone flashlight) to test thresholding.\n")
    
    cap = cv2.VideoCapture(0) # 0 is usually the default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        # Process the frame
        processed_frame, mask, messages = process_frame(frame)
        
        # Output JSON packets to console if detected
        for msg in messages:
            print(json.dumps(msg))
            print("-" * 30)
            
        # Display results in 3 windows to show the pipeline to the judges
        cv2.imshow("1. Raw UV Feed Simulation", frame)
        cv2.imshow("2. Fluorescence Mask (Threshold & Morph)", mask)
        cv2.imshow("3. AI Perception Output", processed_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test_camera()
