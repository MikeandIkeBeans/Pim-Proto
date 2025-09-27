#!/usr/bin/env python3
"""
Annotated PIM Video Generator
Creates an annotated video showing real-time PIM movement predictions with enhanced overlays
Combines functionality from video_analyzer.py with improved detection and logging
"""

# Try to import ML components, fallback to mock if not available
try:
    from pim_detection_system import load_trained_model
    from mediapipe_processor import MultiViewPIMProcessor
    from stgcn_features import sequences_to_stgcn_batches
    from stgcn_graph import build_partitions
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe components loaded successfully")
except ImportError as e:
    print(f"⚠️  ML components not available: {e}")
    print("To enable real skeleton detection, install required packages:")
    print("  pip install mediapipe opencv-python torch numpy")
    print("Running in demo mode with mock pose detection")
    MEDIAPIPE_AVAILABLE = False

import os
import cv2
import torch
import numpy as np
from collections import deque, Counter
from datetime import datetime
import time

# Try to import ML components, fallback to mock if not available
try:
    from pim_detection_system import load_trained_model
    from mediapipe_processor import MultiViewPIMProcessor
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe components loaded successfully")
except ImportError as e:
    print(f"⚠️  ML components not available: {e}")
    print("To enable real skeleton detection and predictions, you need to install:")
    print("  pip install mediapipe opencv-python torch numpy")
    print("Running in demo mode with mock pose detection")
    MEDIAPIPE_AVAILABLE = False
    
    # Mock classes for demo (removed - using real MediaPipe now)
    class MockMultiViewPIMProcessor:
        def __init__(self, num_views=1):
            self.num_views = num_views
        
        def crop_video_views(self, frame):
            """Crop frame into views (simulate 3-view vallejo video)"""
            height, width = frame.shape[:2]
            view_width = width // 3  # Changed from 4 to 3 views
            
            # For 3-view videos, typically use the rightmost view (index 2)
            # or center view (index 1) for better pose detection
            selected_view_idx = 2  # Use rightmost view for 3-view setup
            view = frame[:, selected_view_idx*view_width:(selected_view_idx+1)*view_width]
            return [view]  # Return as list for compatibility
        
        def extract_pose_landmarks_from_single_view(self, view_frame):
            """Extract pose landmarks from a single view (mock implementation)"""
            # For demo purposes, return mock landmarks
            # Create 33 mock landmarks in a T-pose, scaled to the view
            height, width = view_frame.shape[:2]
            landmarks = []
            for i in range(33):
                if i < 11:  # Face
                    x = 0.5 + (i - 5) * 0.05
                    y = 0.2
                elif i in [11, 12]:  # Shoulders
                    x = 0.3 if i == 11 else 0.7
                    y = 0.4
                elif i in [13, 14]:  # Elbows
                    x = 0.1 if i == 13 else 0.9
                    y = 0.5
                elif i in [15, 16]:  # Wrists
                    x = 0.05 if i == 15 else 0.95
                    y = 0.6
                elif i in [17, 18, 19, 20, 21, 22]:  # Hands/fingers (simplified)
                    base_x = 0.05 if i < 19 else 0.95
                    y = 0.65 + (i % 4) * 0.02
                    x = base_x + (i % 4) * 0.02
                elif i == 23:  # Left hip
                    x, y = 0.4, 0.7
                elif i == 24:  # Right hip
                    x, y = 0.6, 0.7
                elif i == 25:  # Left knee
                    x, y = 0.35, 0.9
                elif i == 26:  # Right knee
                    x, y = 0.65, 0.9
                elif i == 27:  # Left ankle
                    x, y = 0.3, 1.0
                elif i == 28:  # Right ankle
                    x, y = 0.7, 1.0
                else:  # Other points
                    x, y = 0.5, 0.5
                
                # Ensure coordinates are within [0,1]
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                
                landmarks.append([x, y, 0])  # Keep as normalized coords
                
            return np.array(landmarks)
    
        def extract_pose_landmarks_from_frame(self, frame):
            """Extract pose landmarks from frame (mock implementation)"""
            # First crop to the appropriate view
            views = self.crop_video_views(frame)
            if not views:
                return None
            
            # Use the cropped view for pose detection
            cropped_frame = views[0]
            
            # For demo purposes, let's at least verify the cropping is working
            # by drawing a visible indicator on the frame
            height, width = frame.shape[:2]
            view_width = width // 3
            selected_view_idx = 2
            
            # Draw a colored border around the selected view to show cropping
            start_x = selected_view_idx * view_width
            end_x = (selected_view_idx + 1) * view_width
            cv2.rectangle(frame, (start_x, 0), (end_x, height), (0, 255, 0), 3)  # Green border
            
            # Return mock landmarks for demo (but now we know the cropping works)
            return self.extract_pose_landmarks_from_single_view(cropped_frame)
        def eval(self):
            pass
        
        def __call__(self, x):
            # Return mock predictions
            batch_size = x.shape[0]
            num_classes = 9  # Number of movements
            # Return random logits
            return torch.randn(batch_size, num_classes)
    
    def load_trained_model(path):
        return MockModel(), ['mock_movement'], 'mock'
    
    MultiViewPIMProcessor = MockMultiViewPIMProcessor

class MockModel:
    """Mock model for demo purposes"""
    def eval(self):
        pass
    
    def __call__(self, x):
        # Return mock predictions
        batch_size = x.shape[0] if hasattr(x, 'shape') else 1
        num_classes = 9  # Number of movements
        # Return random logits
        return torch.randn(batch_size, num_classes)

class AnnotatedVideoGenerator:
    def __init__(self, model_path=None, 
                 confidence_threshold=0.7, sequence_length=30, use_ensemble=False):
        """
        Initialize the annotated video generator
        
        Args:
            model_path: Path to trained model (or ensemble config for ensemble mode)
            confidence_threshold: Minimum confidence for detection
            sequence_length: Number of frames for temporal analysis
            use_ensemble: Whether to use ensemble voting instead of single model
        """
        self.confidence_threshold = confidence_threshold
        self.sequence_length = sequence_length
        self.use_ensemble = use_ensemble
        
        # Load the trained model(s)
        print(f"Loading model from: {model_path if model_path else 'ensemble mode'}")
        if MEDIAPIPE_AVAILABLE:
            if use_ensemble:
                self.model, self.movements, self.model_type = self._load_ensemble_models()
            else:
                if model_path is None:
                    raise ValueError("model_path must be provided when use_ensemble=False")
                self.model, self.movements, self.model_type = load_trained_model(model_path)
                self.model.eval()
        else:
            # Mock mode
            if use_ensemble:
                # Create mock ensemble (list of mock models)
                self.model = [{'model': MockModel(), 'model_type': 'mock', 'device': torch.device('cpu')} for _ in range(3)]
                self.model_type = 'ensemble_voting'
            else:
                self.model = MockModel()
                self.model_type = 'mock'
            self.movements = ['decorticate', 'decerebrate', 'versive head', 'fencer posture', 'ballistic', 'chorea', 'tremor', 'dystonia', 'myoclonus']
        
        self.movement_names = {i: movement for i, movement in enumerate(self.movements)}
        print(f"Loaded model with movements: {self.movement_names}")
        
        # Color scheme for different movements
        self.movement_colors = {
            'decorticate': (0, 0, 255),      # Red
            'decerebrate': (0, 100, 255),   # Orange
            'versive head': (0, 255, 255),  # Yellow  
            'fencer posture': (0, 255, 0),  # Green
            'ballistic': (255, 255, 0),     # Cyan
            'chorea': (255, 0, 255),        # Magenta
            'tremor': (128, 0, 128),        # Purple
            'dystonia': (255, 100, 0),      # Blue
            'myoclonus': (0, 255, 128),     # Light Green
        }
        
        # MediaPipe pose connections for skeleton drawing
        self.pose_connections = [
            (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right arm
            (11, 23), (12, 24), (23, 24),  # Shoulders to hips
            (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
            (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
        ]
        
        # Initialize MediaPipe processor
        if MEDIAPIPE_AVAILABLE:
            self.processor = MultiViewPIMProcessor(
                pose_detection=True,
                hand_detection=False,
                face_detection=False,
                num_views=3  # Process 3 views for comprehensive PIM analysis
            )
        else:
            self.processor = MockMultiViewPIMProcessor(num_views=3)
    
    def _load_ensemble_models(self, config_path=None):
        """Load ensemble models for voting - comprehensive ensemble like model_comparison_demo"""
        from concurrent.futures import ThreadPoolExecutor
        import torch
        
        # Define model configs like in model_comparison_demo.py
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
            }
        }
        
        ensemble_models = []
        ensemble_movements = None
        
        # Load joint-bone ensemble models
        for i in range(3):
            ensemble_name = f'ensemble_{i}'
            if ensemble_name in model_configs:
                ensemble_config = model_configs[ensemble_name]
                if os.path.exists(ensemble_config['path']):
                    try:
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
                        
                        print(f"✅ Loaded joint-bone ensemble model: {ensemble_name}")
                        
                    except Exception as e:
                        print(f"❌ Failed to load {ensemble_name}: {e}")
                        continue
        
        # Load the joint_bone model if available
        if 'joint_bone' in model_configs:
            joint_bone_config = model_configs['joint_bone']
            if os.path.exists(joint_bone_config['path']):
                try:
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
                    
                    print(f"✅ Loaded joint-bone model")
                    
                except Exception as e:
                    print(f"❌ Failed to load joint_bone: {e}")
        
        # Load ST-GCN ensemble models (excluding stgcn_2 due to chorea bias)
        for i in range(2):  # Only load stgcn_0 and stgcn_1, skip stgcn_2
            stgcn_name = f'stgcn_{i}'
            if stgcn_name in model_configs:
                stgcn_config = model_configs[stgcn_name]
                if os.path.exists(stgcn_config['path']):
                    try:
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
                        
                        print(f"✅ Loaded ST-GCN ensemble model: {stgcn_name}")
                        
                    except Exception as e:
                        print(f"❌ Failed to load {stgcn_name}: {e}")
                        continue
        
        # Load comprehensive ST-GCN model
        if 'stgcn_full' in model_configs:
            stgcn_config = model_configs['stgcn_full']
            if os.path.exists(stgcn_config['path']):
                try:
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
                    
                    print(f"✅ Loaded ST-GCN full comprehensive model")
                    
                except Exception as e:
                    print(f"❌ Failed to load stgcn_full: {e}")
        
        if not ensemble_models:
            raise ValueError("No ensemble models could be loaded!")
        
        print(f"🎯 Ensemble loaded: {len(ensemble_models)} models")
        return ensemble_models, ensemble_movements, 'ensemble_voting'
    
    def _predict_ensemble_voting(self, frame_buffer):
        """Predict using ensemble voting across all models"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Convert sequence to tensor
        sequence = np.array(list(frame_buffer))  # Shape: [seq_length, 33, 3]
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
        with ThreadPoolExecutor(max_workers=min(len(self.model), 6)) as executor:
            future_to_model = {executor.submit(predict_single_model, model_info): model_info 
                             for model_info in self.model}
            
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
            movement = self.movement_names[pred_idx]
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
        
        movement_name = self.movement_names[best_prediction_idx]
        
        # Debug: print ensemble voting results occasionally
        if np.random.random() < 0.01:  # 1% chance to print
            print(f"🎯 Ensemble: {movement_name} ({best_confidence:.3f}) - Votes: {predictions}")
        
        return movement_name, best_confidence
        
        # Color scheme for different movements
        self.movement_colors = {
            'decorticate': (0, 0, 255),      # Red
            'decerebrate': (0, 100, 255),   # Orange
            'versive head': (0, 255, 255),  # Yellow  
            'fencer posture': (0, 255, 0),  # Green
            'ballistic': (255, 255, 0),     # Cyan
            'chorea': (255, 0, 255),        # Magenta
            'tremor': (128, 0, 128),        # Purple
            'dystonia': (255, 100, 0),      # Blue
            'myoclonus': (0, 255, 128),     # Light Green
        }
        
        # MediaPipe pose connections for skeleton drawing
        
    def create_overlay_text(self, frame, prediction, confidence, timestamp, 
                           frame_count, detection_history, pose_detected=True):
        """Create overlay text on the frame"""
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Background panel
        panel_height = 180
        panel_width = width
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
        
        # Title
        cv2.putText(overlay, "PIM MOVEMENT DETECTION", (10, 25), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Current prediction with color coding
        pred_color = self.movement_colors.get(prediction, (255, 255, 255))
        cv2.putText(overlay, f"CURRENT: {prediction.upper()}", (10, 55), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, pred_color, 2)
        
        # Confidence
        conf_color = (0, 255, 0) if confidence >= 0.8 else (0, 255, 255) if confidence >= 0.7 else (0, 165, 255)
        cv2.putText(overlay, f"CONFIDENCE: {confidence:.3f}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
        
        # Time and frame info
        cv2.putText(overlay, f"TIME: {timestamp:.1f}s  FRAME: {frame_count}", (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Pose detection status
        pose_status = "POSE DETECTED" if pose_detected else "NO POSE"
        pose_color = (0, 255, 0) if pose_detected else (0, 0, 255)
        cv2.putText(overlay, pose_status, (10, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 1)
        
        # Recent detection history (last 10)
        if detection_history:
            history_text = "HISTORY: " + " -> ".join([d['movement'][:4].upper() for d in detection_history[-10:]])
            cv2.putText(overlay, history_text[:80] + "..." if len(history_text) > 80 else history_text, 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Blend overlay with original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def draw_pose_skeleton(self, frame, landmarks):
        """Draw MediaPipe pose skeleton on the frame"""
        height, width = frame.shape[:2]
        
        # Draw connections
        for connection in self.pose_connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                
                # Convert normalized coordinates to pixel coordinates
                start_x = int(start_point[0] * width)
                start_y = int(start_point[1] * height)
                end_x = int(end_point[0] * width)
                end_y = int(end_point[1] * height)
                
                # Draw line
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
        
        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            x = int(landmark[0] * width)
            y = int(landmark[1] * height)
            
            # Different colors for different body parts
            if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Face
                color = (0, 255, 255)  # Yellow
            elif i in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:  # Arms
                color = (255, 0, 0)  # Blue
            else:  # Legs and torso
                color = (0, 255, 0)  # Green
            
            cv2.circle(frame, (x, y), 6, color, -1)
        
        return frame
    
    def generate_annotated_video(self, input_video_path, output_video_path=None, 
                                max_duration=None, start_time=0):
        """
        Generate an annotated video with PIM predictions
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to save annotated video (None = auto-generate)
            max_duration: Maximum duration to process in seconds (None = full video)
            start_time: Start processing from this time in seconds
        """
        
        if output_video_path is None:
            base_name = os.path.splitext(os.path.basename(input_video_path))[0]
            output_video_path = f"annotated_{base_name}.mp4"
        
        print(f"Input video: {input_video_path}")
        print(f"Output video: {output_video_path}")
        
        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {width}x{height}, {fps:.1f} FPS, {duration:.1f}s, {total_frames} frames")
        
        # Auto-detect if this is a multi-view video and crop to single view
        num_views = 1  # Default to single view
        view_width = width
        
        # Try to detect multi-view setup
        possible_views = [3, 4, 2]  # Prioritize 3 views for PIM videos
        for views in possible_views:
            if width % views == 0:
                test_view_width = width // views
                # Check if view width is reasonable (not too narrow)
                if test_view_width >= 320:  # Minimum reasonable width for pose detection
                    num_views = views
                    view_width = test_view_width
                    break
        
        selected_view_idx = 2 if num_views == 3 else 0  # Use rightmost view for 3-view, first for others
        print(f"Detected {num_views} views, using view {selected_view_idx} (width: {view_width})")
        
        # Calculate processing range
        start_frame = int(start_time * fps) if start_time > 0 else 0
        if max_duration:
            end_frame = min(start_frame + int(max_duration * fps), total_frames)
        else:
            end_frame = total_frames
            
        processing_frames = end_frame - start_frame
        processing_duration = processing_frames / fps
        
        print(f"Processing: {start_time:.1f}s to {(end_frame/fps):.1f}s ({processing_duration:.1f}s, {processing_frames} frames)")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if num_views > 1:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (view_width, height))
            print(f"Output will be cropped to single view: {view_width}x{height}")
        else:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Initialize tracking variables
        frame_buffer = deque(maxlen=self.sequence_length)
        detection_history = []
        frame_count = 0
        processed_frames = 0
        
        # Jump to start position
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print(f"\n🎬 Starting video annotation...")
        start_time_process = time.time()
        
        try:
            while frame_count < processing_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_frame_num = start_frame + frame_count
                timestamp = current_frame_num / fps if fps > 0 else current_frame_num
                
                # Process frame with MediaPipe
                # First crop to the selected view if multi-view video
                if num_views > 1:
                    cropped_frame = frame[:, selected_view_idx*view_width:(selected_view_idx+1)*view_width]
                else:
                    cropped_frame = frame
                
                landmarks = self.processor.extract_pose_landmarks_from_single_view(cropped_frame)
                pose_detected = landmarks is not None
                
                current_prediction = "NO POSE"
                current_confidence = 0.0
                
                if pose_detected:
                    # Add to buffer
                    frame_buffer.append(landmarks)
                    
                    # When buffer is full, make prediction
                    if len(frame_buffer) == self.sequence_length:
                        if self.use_ensemble:
                            # Ensemble voting prediction
                            current_prediction, current_confidence = self._predict_ensemble_voting(frame_buffer)
                        elif self.model_type == 'stgcn':
                            # STGCN model: convert sequence to joints and bones
                            sequence_array = np.array(list(frame_buffer))  # [T, 33, 3]
                            from stgcn_graph import mp_edges
                            edges = mp_edges()
                            Xj, Xb = sequences_to_stgcn_batches([sequence_array], edges)
                            joints_tensor = torch.FloatTensor(Xj[0]).unsqueeze(0)  # [1, C, T, V]
                            bones_tensor = torch.FloatTensor(Xb[0]).unsqueeze(0)   # [1, C, T, V]
                            
                            with torch.no_grad():
                                output = self.model(joints_tensor, bones_tensor)
                        else:
                            # LSTM model: single sequence tensor
                            sequence_tensor = torch.FloatTensor(list(frame_buffer)).unsqueeze(0)
                            
                            with torch.no_grad():
                                output = self.model(sequence_tensor)
                        
                        if not self.use_ensemble:
                            # Handle tuple output
                            if isinstance(output, tuple):
                                output = output[0]
                                
                            probabilities = torch.softmax(output, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)
                            
                            pred_class = int(predicted.item())
                            current_confidence = confidence.item()
                            current_prediction = self.movement_names.get(pred_class, f"unknown_{pred_class}")
                        
                        # Record high-confidence detections
                        if current_confidence >= self.confidence_threshold:
                            detection_history.append({
                                'frame': current_frame_num,
                                'timestamp': timestamp,
                                'movement': current_prediction,
                                'confidence': current_confidence
                            })
                    else:
                        current_prediction = "BUFFERING..."
                
                # Create annotated frame - use cropped frame for multi-view
                annotated_frame = cropped_frame.copy()
                
                # Add text overlay first
                annotated_frame = self.create_overlay_text(
                    annotated_frame, current_prediction, current_confidence, 
                    timestamp, current_frame_num, detection_history, pose_detected
                )
                
                # Draw pose skeleton on top if pose detected
                if pose_detected:
                    annotated_frame = self.draw_pose_skeleton(annotated_frame, landmarks)
                
                # Write frame to output video
                out.write(annotated_frame)
                
                frame_count += 1
                processed_frames += 1
                
                # Progress update every 5 seconds
                if processed_frames % (int(fps) * 5) == 0:
                    progress = (processed_frames / processing_frames) * 100
                    elapsed = time.time() - start_time_process
                    remaining = (elapsed / processed_frames) * (processing_frames - processed_frames)
                    print(f"Progress: {progress:.1f}% ({processed_frames}/{processing_frames} frames) "
                          f"- ETA: {remaining:.0f}s")

                    # Memory management: periodic cleanup every 1000 frames
                    if processed_frames % 1000 == 0:
                        import gc
                        gc.collect()
                        print(f"🧹 Memory cleanup performed")
        
        finally:
            cap.release()
            out.release()
        
        total_time = time.time() - start_time_process
        print(f"\n✅ Annotation complete!")
        print(f"Processing time: {total_time:.1f}s")
        print(f"Output saved: {output_video_path}")
        
        # Summary statistics
        if detection_history:
            movement_counts = Counter([d['movement'] for d in detection_history])
            print(f"\nDetection Summary:")
            for movement, count in movement_counts.most_common():
                percentage = (count / len(detection_history)) * 100
                print(f"  {movement}: {count} ({percentage:.1f}%)")
        
        return output_video_path

    def generate_annotated_videos_for_all_views(self, input_video_path, output_dir=None, 
                                               max_duration=None, start_time=0, num_views=None):
        """
        Generate separate annotated videos for each view in a multi-view video
        
        Args:
            input_video_path: Path to input multi-view video
            output_dir: Directory to save annotated videos (None = auto-generate)
            max_duration: Maximum duration to process in seconds (None = full video)
            start_time: Start processing from this time in seconds
            num_views: Number of views to extract (None = auto-detect from video width)
        """
        
        if output_dir is None:
            base_name = os.path.splitext(os.path.basename(input_video_path))[0]
            output_dir = f"annotated_views_{base_name}"
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Input video: {input_video_path}")
        print(f"Output directory: {output_dir}")
        
        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {width}x{height}, {fps:.1f} FPS, {duration:.1f}s, {total_frames} frames")
        
        # Auto-detect number of views if not specified
        if num_views is None:
            # Try common view counts (3, 4, 2) - prefer 3 views when ambiguous
            possible_views = [3, 4, 2]  # Prioritize 3 views for PIM videos
            detected_views = None
            
            for views in possible_views:
                if width % views == 0:
                    view_width = width // views
                    # Check if view width is reasonable (not too narrow)
                    if view_width >= 320:  # Minimum reasonable width for pose detection
                        detected_views = views
                        break
            
            if detected_views is None:
                # Fallback: prefer 3 views if width is divisible by 3, otherwise 4
                detected_views = 3 if width % 3 == 0 else 4
            
            num_views = detected_views
        
        print(f"Detected/using {num_views} views")
        
        # Validate view configuration
        if width % num_views != 0:
            print(f"⚠️  Warning: Video width ({width}) not evenly divisible by {num_views} views")
            print("   This may cause cropping issues. Consider specifying num_views manually.")
        
        view_width = width // num_views
        
        # Generate view names based on count
        if num_views == 2:
            view_names = ['left', 'right']
        elif num_views == 3:
            view_names = ['left', 'center', 'right']
        elif num_views == 4:
            view_names = ['left', 'center_left', 'center_right', 'right']
        else:
            view_names = [f'view_{i+1}' for i in range(num_views)]
        
        print(f"View configuration: {num_views} views, {view_width}x{height} each")
        print(f"View names: {view_names}")
        
        # Calculate processing range
        start_frame = int(start_time * fps) if start_time > 0 else 0
        if max_duration:
            end_frame = min(start_frame + int(max_duration * fps), total_frames)
        else:
            end_frame = total_frames
            
        processing_frames = end_frame - start_frame
        processing_duration = processing_frames / fps
        
        print(f"Processing: {start_time:.1f}s to {(end_frame/fps):.1f}s ({processing_duration:.1f}s, {processing_frames} frames)")
        
        # Initialize tracking for each view
        view_buffers = [deque(maxlen=self.sequence_length) for _ in range(num_views)]
        view_detections = [[] for _ in range(num_views)]
        view_frame_counts = [0] * num_views
        
        # Setup video writers for each view
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        view_writers = []
        for i, view_name in enumerate(view_names):
            output_path = os.path.join(output_dir, f"annotated_{view_name}.mp4")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (view_width, height))
            view_writers.append(writer)
            print(f"View {i+1} ({view_name}): {view_width}x{height} -> {output_path}")
        
        # Jump to start position
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print(f"\n🎬 Starting multi-view annotation...")
        start_time_process = time.time()
        
        try:
            while view_frame_counts[0] < processing_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_frame_num = start_frame + view_frame_counts[0]
                timestamp = current_frame_num / fps if fps > 0 else current_frame_num
                
                # Crop frame into views
                views = []
                for i in range(num_views):
                    view = frame[:, i*view_width:(i+1)*view_width]
                    views.append(view)
                
                # Process each view
                for view_idx, view_frame in enumerate(views):
                    # Extract pose landmarks for this view
                    landmarks = self.processor.extract_pose_landmarks_from_single_view(view_frame)
                    pose_detected = landmarks is not None
                    
                    current_prediction = "NO POSE"
                    current_confidence = 0.0
                    
                    if pose_detected:
                        # Add to buffer for this view
                        view_buffers[view_idx].append(landmarks)
                        
                        # When buffer is full, make prediction
                        if len(view_buffers[view_idx]) == self.sequence_length:
                            if self.use_ensemble:
                                # Use ensemble voting
                                current_prediction, current_confidence = self._predict_ensemble_voting(view_buffers[view_idx])
                            elif self.model_type == 'stgcn':
                                # STGCN model: convert sequence to joints and bones
                                sequence_array = np.array(list(view_buffers[view_idx]))  # [T, 33, 3]
                                from stgcn_graph import mp_edges
                                edges = mp_edges()
                                Xj, Xb = sequences_to_stgcn_batches([sequence_array], edges)
                                joints_tensor = torch.FloatTensor(Xj[0]).unsqueeze(0)  # [1, C, T, V]
                                bones_tensor = torch.FloatTensor(Xb[0]).unsqueeze(0)   # [1, C, T, V]
                                
                                with torch.no_grad():
                                    output = self.model(joints_tensor, bones_tensor)
                            else:
                                # LSTM model: single sequence tensor
                                sequence_tensor = torch.FloatTensor(list(view_buffers[view_idx])).unsqueeze(0)
                                
                                with torch.no_grad():
                                    output = self.model(sequence_tensor)
                            
                            if not self.use_ensemble:
                                if isinstance(output, tuple):
                                    output = output[0]
                                    
                                probabilities = torch.softmax(output, dim=1)
                                confidence, predicted = torch.max(probabilities, 1)
                                
                                pred_class = int(predicted.item())
                                current_confidence = confidence.item()
                                current_prediction = self.movement_names.get(pred_class, f"unknown_{pred_class}")
                            
                            # Record high-confidence detections
                            if current_confidence >= self.confidence_threshold:
                                view_detections[view_idx].append({
                                    'frame': current_frame_num,
                                    'timestamp': timestamp,
                                    'movement': current_prediction,
                                    'confidence': current_confidence
                                })
                        else:
                            current_prediction = "BUFFERING..."
                    
                    # Create annotated frame for this view
                    annotated_frame = view_frame.copy()
                    
                    # Draw pose skeleton if pose detected
                    if pose_detected:
                        annotated_frame = self.draw_pose_skeleton(annotated_frame, landmarks)
                    
                    # Add text overlay
                    annotated_frame = self.create_overlay_text(
                        annotated_frame, current_prediction, current_confidence, 
                        timestamp, current_frame_num, view_detections[view_idx], pose_detected
                    )
                    
                    # Write frame to this view's video
                    view_writers[view_idx].write(annotated_frame)
                    
                    view_frame_counts[view_idx] += 1
                
                # Progress update
                if view_frame_counts[0] % (int(fps) * 5) == 0:
                    progress = (view_frame_counts[0] / processing_frames) * 100
                    elapsed = time.time() - start_time_process
                    remaining = (elapsed / view_frame_counts[0]) * (processing_frames - view_frame_counts[0])
                    print(f"Progress: {progress:.1f}% ({view_frame_counts[0]}/{processing_frames} frames) "
                          f"- ETA: {remaining:.0f}s")
        
        finally:
            cap.release()
            for writer in view_writers:
                writer.release()
        
        total_time = time.time() - start_time_process
        print(f"\n✅ Multi-view annotation complete!")
        print(f"Processing time: {total_time:.1f}s")
        print(f"Output directory: {output_dir}")
        
        # Summary statistics for each view
        for view_idx, (view_name, detections) in enumerate(zip(view_names, view_detections)):
            if detections:
                movement_counts = Counter([d['movement'] for d in detections])
                print(f"\n{view_name.upper()} VIEW ({len(detections)} detections):")
                for movement, count in movement_counts.most_common():
                    percentage = (count / len(detections)) * 100
                    # Calculate confidence statistics for this movement
                    movement_confidences = [d['confidence'] for d in detections if d['movement'] == movement]
                    if movement_confidences:
                        avg_conf = sum(movement_confidences) / len(movement_confidences)
                        min_conf = min(movement_confidences)
                        max_conf = max(movement_confidences)
                        print(f"  {movement}: {count} ({percentage:.1f}%) | Conf: {avg_conf:.3f} (min: {min_conf:.3f}, max: {max_conf:.3f})")
                    else:
                        print(f"  {movement}: {count} ({percentage:.1f}%)")
        
        return output_dir

if __name__ == "__main__":
    # Run the annotated video generator with ensemble voting on a napa video
    ensemble_config = "models/ensemble/ensemble_config.pth"  # Path to ensemble configuration
    input_video = "data/napa/2025-06-01 10-22-32.mkv"
    
    print("🎬 Starting Annotated Video Generator with Ensemble Voting")
    print(f"Ensemble Config: {ensemble_config}")
    print(f"Video: {input_video}")
    
    try:
        generator = AnnotatedVideoGenerator(
            model_path=None,  # Not used for ensemble mode
            confidence_threshold=0.7,
            sequence_length=30,
            use_ensemble=True
        )
        
        output_path = generator.generate_annotated_video(
            input_video_path=input_video,
            max_duration=60,  # Process only first 60 seconds for testing
            start_time=0
        )
        
        print(f"\n✅ Success! Annotated video saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()