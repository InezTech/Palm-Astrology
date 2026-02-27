import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import math

class CVPipeline:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
    def process_frame(self, frame):
        """Processes a frame to detect hands, calculate shape, and draw AR overlay."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_stats = None
        overlay_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                
                # Calculate hand shape stats
                hand_stats = self._calculate_hand_shape(hand_landmarks.landmark, w, h)
                
                # Draw AR scan overlay
                self._draw_ar_overlay(overlay_frame, hand_landmarks.landmark, w, h)
                
                # Extract and highlight palm lines
                self._extract_palm_lines(overlay_frame, hand_landmarks.landmark, w, h)
                
        return overlay_frame, hand_stats, results.multi_hand_landmarks

    def _calculate_hand_shape(self, landmarks, w, h):
        """Calculates deterministic ML hand ratios for elemental hand shape."""
        # Helper to get pixel coords
        def pt(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
            
        # Palm width (index base to pinky base)
        palm_w = np.linalg.norm(pt(5) - pt(17))
        # Palm length (middle finger base to wrist)
        palm_l = np.linalg.norm(pt(9) - pt(0))
        # Middle finger length
        finger_l = np.linalg.norm(pt(12) - pt(9))
        
        is_square_palm = abs(palm_w - palm_l) / max(palm_w, palm_l) < 0.15
        is_long_fingers = finger_l > palm_l * 0.8
        
        if is_square_palm and not is_long_fingers:
            element = "Earth"
        elif is_square_palm and is_long_fingers:
            element = "Air"
        elif not is_square_palm and not is_long_fingers:
            element = "Fire"
        else:
            element = "Water"
            
        return {
            "palm_width": round(palm_w, 2),
            "palm_length": round(palm_l, 2),
            "finger_length": round(finger_l, 2),
            "element": element,
            "is_square_palm": is_square_palm,
            "is_long_fingers": is_long_fingers
        }

    def _draw_ar_overlay(self, frame, landmarks, w, h):
        """Draws a sci-fi bounding box and scanning line."""
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        pad = 20
        x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
        x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)
        
        # Draw high-end corner brackets
        length = 30
        thick = 2
        col = (0, 255, 255) # Yellow AR color
        
        # Corners
        cv2.line(frame, (x_min, y_min), (x_min + length, y_min), col, thick)
        cv2.line(frame, (x_min, y_min), (x_min, y_min + length), col, thick)
        cv2.line(frame, (x_max, y_min), (x_max - length, y_min), col, thick)
        cv2.line(frame, (x_max, y_min), (x_max, y_min + length), col, thick)
        cv2.line(frame, (x_min, y_max), (x_min + length, y_max), col, thick)
        cv2.line(frame, (x_min, y_max), (x_min, y_max - length), col, thick)
        cv2.line(frame, (x_max, y_max), (x_max - length, y_max), col, thick)
        cv2.line(frame, (x_max, y_max), (x_max, y_max - length), col, thick)
        
        # Draw a scanning line moving down
        import time
        t = time.time()
        scan_y = int(y_min + (y_max - y_min) * ((t * 1.5) % 1.0))
        cv2.line(frame, (x_min, scan_y), (x_max, scan_y), (0, 255, 0), 2)
        
    def _extract_palm_lines(self, frame, landmarks, w, h):
        """Uses Canny Edge detection specifically on the palm region."""
        # Create a mask for the palm area (roughly landmarks 0, 1, 5, 9, 13, 17)
        palm_pts = np.array([
            [landmarks[0].x * w, landmarks[0].y * h],
            [landmarks[1].x * w, landmarks[1].y * h],
            [landmarks[5].x * w, landmarks[5].y * h],
            [landmarks[9].x * w, landmarks[9].y * h],
            [landmarks[13].x * w, landmarks[13].y * h],
            [landmarks[17].x * w, landmarks[17].y * h]
        ], np.int32)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, palm_pts, 255)
        
        # Apply Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray_blurred, 30, 100)
        
        # Mask the edges to only the palm
        palm_edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Highlight edges in cyan on the original frame
        frame[palm_edges > 0] = [255, 255, 0]  # Cyan in BGR
