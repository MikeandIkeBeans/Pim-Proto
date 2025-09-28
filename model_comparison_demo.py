#!/usr/bin/env python3
"""
Model Comparison Demo - Test Multiple PIM Detection Models
Allows switching between different model architectures for comparison
"""

import cv2
import mediapipe as mp
import torch
import numpy as np
import time
from collections import deque
import sys
import os
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pim_detection_system import load_trained_model

class ModelComparisonDemo:
    def __init__(self, model_configs, sequence_length=30):
        self.sequence_length = sequence_length
        self.pose_sequences = deque(maxlen=sequence_length)
        
        # Performance optimization for ensemble voting
        self.ensemble_frame_skip = 3  # Only run ensemble every 3 frames
        self.frame_counter = 0
        self.last_ensemble_result = ("Collecting data...", 0.0)
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Load all models
        self.models = {}
        self.model_info = {}

        for name, config in model_configs.items():
            try:
                print(f"Loading {name}...")
                
                # Special handling for comprehensive ST-GCN model
                if name == 'stgcn_full':
                    checkpoint = torch.load(config['path'], map_location='cpu')
                    
                    movements = checkpoint['classes']
                    
                    from stgcn_model import STGCNTwoStream
                    from stgcn_graph import build_partitions
                    A = build_partitions()
                    model = STGCNTwoStream(num_classes=len(movements), A=A)
                    # Handle different checkpoint formats
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        raise KeyError("Checkpoint missing state_dict or model_state_dict")
                    
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model.to(device)
                    model.eval()
                    
                    self.models[name] = {
                        'model': model,
                        'movements': movements,
                        'model_type': 'stgcntwostream',
                        'device': device
                    }
                    self.model_info[name] = config
                    
                    print(f"✅ {name}: stgcntwostream model loaded")
                    print(f"   Classes: {movements}")
                    continue
                
                # Special handling for enhanced ST-GCN model
                if name == 'stgcn_enhanced':
                    checkpoint = torch.load(config['path'], map_location='cpu')
                    
                    movements = checkpoint['classes']
                    
                    from stgcn_model import STGCNTwoStreamEnhanced
                    from stgcn_graph import build_partitions
                    A = build_partitions()
                    
                    # Use the enhanced architecture that matches the checkpoint
                    model = STGCNTwoStreamEnhanced(num_classes=len(movements), A=A)
                    
                    # Handle different checkpoint formats
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        raise KeyError("Checkpoint missing state_dict or model_state_dict")
                    
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model.to(device)
                    model.eval()
                    
                    self.models[name] = {
                        'model': model,
                        'movements': movements,
                        'model_type': 'stgcntwostreamenhanced',
                        'device': device
                    }
                    self.model_info[name] = config
                    
                    print(f"✅ {name}: stgcntwostream enhanced model loaded")
                    print(f"   Classes: {movements}")
                    continue
                
                # Special handling for ensemble mode
                if name == 'ensemble':
                    # Load all ensemble models for voting
                    ensemble_models = []
                    ensemble_movements = None
                    
                    # Load joint-bone ensemble models
                    for i in range(3):
                        ensemble_name = f'ensemble_{i}'
                        if ensemble_name in model_configs:
                            ensemble_config = model_configs[ensemble_name]
                            checkpoint = torch.load(ensemble_config['path'])
                            
                            movements = checkpoint['classes']
                            if ensemble_movements is None:
                                ensemble_movements = movements
                            
                            from mediapipe_processor import JointBoneEnsembleLSTM
                            model = JointBoneEnsembleLSTM(num_classes=len(movements))
                            model.load_state_dict(checkpoint['state_dict'])
                            
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            model.to(device)
                            model.eval()
                            
                            ensemble_models.append({
                                'model': model,
                                'model_type': 'jointboneensemblelstm',
                                'device': device
                            })
                    
                    # Load the joint_bone model if available
                    if 'joint_bone' in model_configs:
                        joint_bone_config = model_configs['joint_bone']
                        model, movements, model_type = load_trained_model(joint_bone_config['path'])
                        
                        if ensemble_movements is None:
                            ensemble_movements = movements
                        
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model.to(device)
                        model.eval()
                        
                        ensemble_models.append({
                            'model': model,
                            'model_type': model_type,
                            'device': device
                        })
                    
                    # Load ST-GCN ensemble models (excluding stgcn_2 due to chorea bias)
                    for i in range(3):
                        if i == 2:  # Skip stgcn_2 due to heavy chorea bias
                            continue
                        stgcn_name = f'stgcn_{i}'
                        if stgcn_name in model_configs:
                            stgcn_config = model_configs[stgcn_name]
                            checkpoint = torch.load(stgcn_config['path'])
                            
                            movements = checkpoint['classes']
                            
                            from stgcn_model import STGCNTwoStream
                            from stgcn_graph import build_partitions
                            A = build_partitions()
                            model = STGCNTwoStream(num_classes=len(movements), A=A)
                            model.load_state_dict(checkpoint['state_dict'])
                            
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            model.to(device)
                            model.eval()
                            
                            ensemble_models.append({
                                'model': model,
                                'model_type': 'stgcntwostream',
                                'device': device
                            })
                    
                    # Load ST-GCN ensemble models
                    for i in range(2):
                        stgcn_name = f'stgcn_{i}'
                        if stgcn_name in model_configs:
                            stgcn_config = model_configs[stgcn_name]
                            checkpoint = torch.load(stgcn_config['path'])
                            
                            movements = checkpoint['classes']
                            if ensemble_movements is None:
                                ensemble_movements = movements
                            
                            from stgcn_model import STGCNTwoStream
                            from stgcn_graph import build_partitions
                            A = build_partitions()
                            model = STGCNTwoStream(num_classes=len(movements), A=A)
                            model.load_state_dict(checkpoint['state_dict'])
                            
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            model.to(device)
                            model.eval()
                            
                            ensemble_models.append({
                                'model': model,
                                'model_type': 'stgcntwostream',
                                'device': device
                            })
                    
                    # Load comprehensive ST-GCN model
                    if 'stgcn_full' in model_configs:
                        stgcn_config = model_configs['stgcn_full']
                        checkpoint = torch.load(stgcn_config['path'])
                        
                        movements = checkpoint['classes']
                        if ensemble_movements is None:
                            ensemble_movements = movements
                        
                        from stgcn_model import STGCNTwoStream
                        from stgcn_graph import build_partitions
                        A = build_partitions()
                        model = STGCNTwoStream(num_classes=len(movements), A=A)
                        # Handle different checkpoint formats
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        elif 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            raise KeyError("Checkpoint missing state_dict or model_state_dict")
                        
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model.to(device)
                        model.eval()
                        
                        ensemble_models.append({
                            'model': model,
                            'model_type': 'stgcntwostream',
                            'device': device
                        })
                    
                    self.models[name] = {
                        'ensemble_models': ensemble_models,
                        'movements': ensemble_movements,
                        'model_type': 'ensemble_voting',
                        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    }
                    
                    print(f"✅ {name}: Ensemble voting model loaded ({len(ensemble_models)} models)")
                    print(f"   Classes: {ensemble_movements}")
                    continue
                
                # Regular model loading
                checkpoint = torch.load(config['path'])

                # Handle different model save formats
                if 'state_dict' in checkpoint:
                    # Ensemble model format
                    movements = checkpoint['classes']
                    model_type = checkpoint.get('architecture', 'unknown').lower()

                    # Create model instance
                    if 'stgcn' in model_type.lower():
                        from stgcn_model import STGCNTwoStream
                        from stgcn_graph import build_partitions
                        A = build_partitions()
                        model = STGCNTwoStream(num_classes=len(movements), A=A)
                    elif 'jointbone' in model_type.lower():
                        from mediapipe_processor import JointBoneEnsembleLSTM
                        model = JointBoneEnsembleLSTM(num_classes=len(movements))
                    else:
                        # Fallback to joint bone model
                        from mediapipe_processor import JointBoneEnsembleLSTM
                        model = JointBoneEnsembleLSTM(num_classes=len(movements))

                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Standard model format - use load_trained_model
                    model, movements, model_type = load_trained_model(config['path'])

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                model.eval()

                self.models[name] = {
                    'model': model,
                    'movements': movements,
                    'model_type': model_type,
                    'device': device
                }
                self.model_info[name] = config

                print(f"✅ {name}: {model_type} model loaded")
                print(f"   Classes: {movements}")

            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")

        # Set default active model
        self.active_model = list(self.models.keys())[0] if self.models else None
        print(f"\n🎯 Active model: {self.active_model}")

    def switch_model(self, model_name):
        """Switch to a different model"""
        print(f"Attempting to switch to model: {model_name}")
        if model_name in self.models:
            self.active_model = model_name
            print(f"🔄 Switched to model: {model_name}")
            return True
        else:
            print(f"❌ Model not found: {model_name}")
            return False

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
        """Predict PIM movement using active model"""
        if len(self.pose_sequences) < self.sequence_length:
            return "Collecting data...", 0.0

        if not self.active_model or self.active_model not in self.models:
            return "No model loaded", 0.0

        try:
            model_data = self.models[self.active_model]
            movement_names = {i: movement for i, movement in enumerate(model_data['movements'])}
            
            # Handle ensemble voting with performance optimization
            if self.active_model == 'ensemble':
                self.frame_counter += 1
                # Only run expensive ensemble voting every N frames
                if self.frame_counter % self.ensemble_frame_skip == 0:
                    self.last_ensemble_result = self._predict_ensemble_voting(model_data, movement_names)
                return self.last_ensemble_result
            
            # Single model prediction
            model = model_data['model']
            model_type = model_data['model_type']
            device = model_data['device']

            # Convert sequence to tensor in [batch, seq, 33, 3] format
            sequence = np.array(list(self.pose_sequences))  # Shape: [seq_length, 33, 3]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)  # Shape: [1, seq_length, 33, 3]

            # Make prediction
            with torch.no_grad():
                if 'stgcn' in model_type.lower():
                    # ST-GCN model needs joints and bones in [C, T, V] format
                    from stgcn_features import sequences_to_stgcn_batches
                    from stgcn_graph import mp_edges
                    
                    # Convert sequence to ST-GCN format
                    edges = mp_edges()
                    seq_list = [sequence]  # List with single sequence
                    X_joints, X_bones = sequences_to_stgcn_batches(seq_list, edges)
                    
                    # Convert to tensors
                    joints_tensor = torch.FloatTensor(X_joints).to(device)
                    bones_tensor = torch.FloatTensor(X_bones).to(device)
                    
                    outputs = model(joints_tensor, bones_tensor)
                else:
                    # Regular model or joint bone model
                    outputs = model(sequence_tensor)
                    
                    # Handle models that return (logits, confidence) tuple
                    if isinstance(outputs, tuple):
                        outputs, _ = outputs

                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)

            movement_name = movement_names[int(predicted_class.item())]
            confidence_score = confidence.item()

            return movement_name, confidence_score

        except Exception as e:
            print(f"Prediction error with {self.active_model}: {e}")
            return "Prediction error", 0.0

    def _predict_ensemble_voting(self, model_data, movement_names):
        """Predict using ensemble voting across all models with parallel processing"""
        ensemble_models = model_data['ensemble_models']
        
        # Convert sequence to tensor
        sequence = np.array(list(self.pose_sequences))  # Shape: [seq_length, 33, 3]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Base tensor, will be moved to device per model
        
        predictions = []
        confidences = []
        
        def predict_single_model(model_info):
            """Helper function to predict with a single model"""
            try:
                model = model_info['model']
                model_type = model_info['model_type']
                device = model_info['device']
                
                # Move tensor to correct device
                seq_tensor = sequence_tensor.to(device)
                
                with torch.no_grad():
                    if 'stgcn' in model_type.lower():
                        # ST-GCN model needs joints and bones in [C, T, V] format
                        from stgcn_features import sequences_to_stgcn_batches
                        from stgcn_graph import mp_edges
                        
                        edges = mp_edges()
                        seq_list = [sequence]
                        X_joints, X_bones = sequences_to_stgcn_batches(seq_list, edges)
                        
                        joints_tensor = torch.FloatTensor(X_joints).to(device)
                        bones_tensor = torch.FloatTensor(X_bones).to(device)
                        
                        outputs = model(joints_tensor, bones_tensor)
                    else:
                        # Regular model or joint bone model
                        outputs = model(seq_tensor)
                        
                        # Handle models that return (logits, confidence) tuple
                        if isinstance(outputs, tuple):
                            outputs, _ = outputs
                    
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                    
                    return int(predicted_class.item()), float(confidence.item())
                    
            except Exception as e:
                print(f"Error with ensemble model {model_info['model_type']}: {e}")
                return None
        
        # Run predictions in parallel
        with ThreadPoolExecutor(max_workers=min(len(ensemble_models), 6)) as executor:
            future_to_model = {executor.submit(predict_single_model, model_info): model_info 
                             for model_info in ensemble_models}
            
            for future in as_completed(future_to_model):
                result = future.result()
                if result is not None:
                    pred_idx, conf = result
                    predictions.append(pred_idx)
                    confidences.append(conf)
        
        if not predictions:
            return "Ensemble error", 0.0
        
        # Apply weighted majority voting
        # Give higher weight to higher confidence predictions
        weights = {}
        for pred_idx, conf in zip(predictions, confidences):
            # Apply bias correction for movements that models tend to default to
            movement = movement_names[pred_idx]
            if movement == 'decerebrate':
                weight = conf * 0.7  # Reduce decerebrate influence
            elif movement == 'chorea':
                weight = conf * 0.8  # Reduce chorea influence (less aggressive than decerebrate)
            elif movement == 'decorticate':
                weight = conf * 0.8  # Reduce versive influence (less aggressive than decerebrate)
            else:
                weight = conf
            
            if pred_idx not in weights:
                weights[pred_idx] = 0
            weights[pred_idx] += weight
        
        # Get the prediction with highest weighted score
        best_prediction_idx = max(weights.items(), key=lambda x: x[1])[0]
        total_weight = weights[best_prediction_idx]
        num_models = len(predictions)
        
        # Calculate ensemble confidence more intuitively
        # Use agreement percentage with confidence boost
        agreeing_models = sum(1 for pred_idx in predictions if pred_idx == best_prediction_idx)
        agreement_ratio = agreeing_models / num_models
        
        # Average confidence of agreeing models
        if agreeing_models > 0:
            avg_agreeing_confidence = total_weight / agreeing_models
        else:
            avg_agreeing_confidence = 0
        
        # Boost confidence based on agreement level
        # More models agreeing = higher confidence
        confidence_boost = agreement_ratio ** 0.5  # Square root for smoother scaling
        best_confidence = min(avg_agreeing_confidence * confidence_boost, 1.0)
        
        movement_name = movement_names[best_prediction_idx]
        
        return movement_name, best_confidence

    def run_comparison_demo(self):
        """Run the model comparison demo"""
        print("🎬 Model Comparison Demo")
        print("=" * 50)
        print("Available models:")
        for name, info in self.model_info.items():
            status = "✅" if name in self.models else "❌"
            print(f"  {status} {name}: {info['description']}")
        print(f"\nActive model: {self.active_model}")
        print("Controls:")
        print("  '1': joint_bone | '2': ensemble_0 | '3': ensemble_1 | '4': ensemble_2")
        print("  '5': stgcn_0    | '6': stgcn_1    | '8': stgcn_full  | '9': stgcn_enhanced")
        print("  '7': ENSEMBLE VOTING")
        print("  Note: Ensemble uses 7 models (includes joint_bone + stgcn_full)")
        print("  'q': Quit")
        print("=" * 50)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Could not open webcam")
            print("Available models for testing:")
            for name, info in self.model_info.items():
                if name in self.models:
                    print(f"  • {name}: {info['description']}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
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
                    status = f"Model: {self.active_model} | Frames: {len(self.pose_sequences)}/{self.sequence_length}"
                    cv2.putText(frame, status, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

                # Add model info
                        info_text = f"Type: {self.models[self.active_model]['model_type']}"
                        cv2.putText(frame, info_text, (10, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        
                        # Show available models and keys
                        help_y = 140
                        cv2.putText(frame, "Available Models:", (10, help_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                        help_y += 15
                        
                        model_keys = {
                            '1': 'joint_bone',
                            '2': 'ensemble_0', 
                            '3': 'ensemble_1',
                            '4': 'ensemble_2',
                            '5': 'stgcn_0',
                            '6': 'stgcn_1',
                            '7': 'ensemble',
                            '8': 'stgcn_full'
                        }
                        
                        for key, model_name in model_keys.items():
                            if model_name in self.models:
                                desc = self.model_info.get(model_name, {}).get('description', model_name)
                                cv2.putText(frame, f"{key}: {desc}", (10, help_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
                                help_y += 12
                else:
                    cv2.putText(frame, "No pose detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Update FPS counter
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:  # Update every second
                    self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                
                fps_text = f"FPS: {self.current_fps:.1f}"
                if self.active_model == 'ensemble':
                    fps_text += f" (1/{self.ensemble_frame_skip} frames)"
                cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Show frame
                cv2.imshow('Model Comparison Demo', frame)

                # Check for key presses
                key = cv2.waitKey(1) & 0xFF

                # Debug key presses (remove this after debugging)
                if key != 255:  # 255 is no key pressed
                    print(f"Key pressed: {key} (ord('7')={ord('7')})")

                # Model switching keys
                if key == ord('1') and 'joint_bone' in self.models:
                    self.switch_model('joint_bone')
                elif key == ord('2') and 'ensemble_0' in self.models:
                    self.switch_model('ensemble_0')
                elif key == ord('3') and 'ensemble_1' in self.models:
                    self.switch_model('ensemble_1')
                elif key == ord('4') and 'ensemble_2' in self.models:
                    self.switch_model('ensemble_2')
                elif key == ord('5') and 'stgcn_0' in self.models:
                    self.switch_model('stgcn_0')
                elif key == ord('6') and 'stgcn_1' in self.models:
                    self.switch_model('stgcn_1')
                elif key == ord('8') and 'stgcn_full' in self.models:
                    self.switch_model('stgcn_full')
                elif key == ord('9') and 'stgcn_enhanced' in self.models:
                    self.switch_model('stgcn_enhanced')
                elif key == ord('7') and 'ensemble' in self.models:
                    self.switch_model('ensemble')
                elif key == ord('q'):
                    break

        except Exception as e:
            print(f"❌ Error during demo: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Demo ended")

def main():
    # Define available models
    model_configs = {
        'joint_bone': {
            'path': 'models/pim_model_joint_bone.pth',
            'description': 'Joint-Bone Ensemble LSTM (Advanced)'
        },
        'ensemble_0': {
            'path': 'models/ensemble/ensemble_model_0.pth',
            'description': 'Ensemble Model 0'
        },
        'ensemble_1': {
            'path': 'models/ensemble/ensemble_model_1.pth',
            'description': 'Ensemble Model 1'
        },
        'ensemble_2': {
            'path': 'models/ensemble/ensemble_model_2.pth',
            'description': 'Ensemble Model 2'
        },
        'stgcn_0': {
            'path': 'models/ensemble/ensemble_model_stgcn_0.pth',
            'description': 'ST-GCN Ensemble Model 0'
        },
        'stgcn_1': {
            'path': 'models/ensemble/ensemble_model_stgcn_1.pth',
            'description': 'ST-GCN Ensemble Model 1'
        },
        'stgcn_full': {
            'path': 'models/stgcn_full_comprehensive.pth',
            'description': 'ST-GCN Full Comprehensive Model (All Data)'
        },
        'stgcn_enhanced': {
            'path': 'models/stgcn_enhanced_model.pth',
            'description': 'ST-GCN Enhanced Model (Fresh Training)'
        },
        'ensemble': {
            'path': 'ensemble_voting',  # Special marker for ensemble mode
            'description': 'Combined Ensemble Voting (All Models)'
        }
    }

    demo = ModelComparisonDemo(model_configs)
    demo.run_comparison_demo()

if __name__ == "__main__":
    main()