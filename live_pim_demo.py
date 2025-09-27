#!/usr/bin/env python3
"""
Live Demo of Ensemble PIM Detection Model
Uses webcam to demonstrate real-time PIM movement classification with ensemble voting
"""

import cv2
import mediapipe as mp
import torch
import numpy as np
import time
from collections import deque
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pim_detection_system import load_trained_model

class LivePIMDemo:
    def __init__(self, model_path="models/pim_model_joint_bone.pth", sequence_length=30):
        self.sequence_length = sequence_length
        self.pose_sequences = deque(maxlen=sequence_length)

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Load the joint bone model
        self.model, self.movements, self.model_type = load_trained_model(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Set up movement names
        self.movement_names = {i: movement for i, movement in enumerate(self.movements)}

        print(f"✅ Model loaded: {self.model_type}")
        print(f"📊 Movement classes: {self.movement_names}")

    def extract_pose_landmarks(self, frame):
        """Extract pose landmarks from a frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            # Extract joint positions in [33, 3] format as expected by the model
            landmarks = results.pose_landmarks.landmark
            joint_positions = np.zeros((33, 3), dtype=np.float32)

            for i, landmark in enumerate(landmarks):
                joint_positions[i] = [landmark.x, landmark.y, landmark.z]

            return joint_positions, results.pose_landmarks

        return None, None

    def draw_pose(self, frame, landmarks):
        """Draw pose landmarks on frame"""
        self.mp_draw.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    def predict_movement(self, features):
        """Predict PIM movement from features"""
        if len(self.pose_sequences) < self.sequence_length:
            return "Collecting data...", 0.0

        # Convert sequence to tensor in [batch, seq, 33, 3] format
        sequence = np.array(list(self.pose_sequences))  # Shape: [seq_length, 33, 3]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # Shape: [1, seq_length, 33, 3]

        # Make prediction
        with torch.no_grad():
            if self.model_type == 'joint_bone':
                # Joint bone model returns (logits, confidence)
                outputs, _ = self.model(sequence_tensor)
            else:
                # Regular model
                outputs = self.model(sequence_tensor)

            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        movement_name = self.movement_names[int(predicted_class.item())]
        confidence_score = confidence.item()

        return movement_name, confidence_score

    def run_live_demo(self):
        """Run the live webcam demo"""
        print("🎬 Starting Live PIM Detection Demo")
        print("📹 Opening webcam... Press 'q' to quit")
        print("=" * 50)

        cap = cv2.VideoCapture(0)  # Use default webcam

        print(f"🔍 Webcam status: {'Open' if cap.isOpened() else 'Closed'}")
        if not cap.isOpened():
            print("❌ Could not open webcam - this may be expected in a headless environment")
            print("💡 Demo Description:")
            print("   - Uses MediaPipe for real-time pose detection")
            print("   - Extracts 33-point skeleton landmarks")
            print("   - Feeds 30-frame sequences to Joint-Bone Ensemble LSTM model")
            print("   - Classifies into 9 PIM movement types:")
            for i, movement in self.movement_names.items():
                print(f"     {i}: {movement}")
            print("   - Shows live predictions with confidence scores")
            print("   - Displays pose skeleton overlay on video feed")
            print("   - Real-time confidence visualization with color coding")
            print("   - FPS counter for performance monitoring")
            print("   - Uses advanced ensemble of joint and bone features")
            return

        # Set webcam properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame from webcam")
                    break

                frame_count += 1

                # Extract pose features
                features, landmarks = self.extract_pose_landmarks(frame)

                if features is not None:
                    # Add to sequence buffer
                    self.pose_sequences.append(features)

                    # Make prediction
                    movement, confidence = self.predict_movement(features)

                    # Draw pose on frame
                    if landmarks:
                        self.draw_pose(frame, landmarks)

                    # Add prediction text
                    status = f"Frames: {len(self.pose_sequences)}/{self.sequence_length}"
                    cv2.putText(frame, status, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if len(self.pose_sequences) >= self.sequence_length:
                        pred_text = f"Prediction: {movement} ({confidence:.2f})"
                        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
                        cv2.putText(frame, pred_text, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                        # Add confidence bar
                        bar_width = 200
                        bar_height = 20
                        bar_x, bar_y = 10, 90
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                        fill_width = int(confidence * bar_width)
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
                else:
                    cv2.putText(frame, "No pose detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Add FPS counter
                fps = frame_count / (time.time() - start_time)
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (frame.shape[1] - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Show frame
                cv2.imshow('Live PIM Detection Demo', frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"❌ Error during demo: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Demo ended")

if __name__ == "__main__":
    demo = LivePIMDemo()
    demo.run_live_demo()