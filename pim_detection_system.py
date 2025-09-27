"""
Complete PIM (Patient Involuntary Movement) Detection System
Integrated with multi-view video processing capabilities.
"""

import cv2
import mediapipe as mp
import time
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import collections
import argparse
import json
import glob
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import sys
from mediapipe_processor import MultiViewPIMProcessor, PIMDetectorLSTM, JointBoneEnsembleLSTM, prepare_sequences, PIM_MOVEMENTS
import sys
from mediapipe_processor import MultiViewPIMProcessor, PIMDetectorLSTM, JointBoneEnsembleLSTM, prepare_sequences, PIM_MOVEMENTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_multi_view_video_file(video_path, movement_name, output_dir="pose_data", num_views=3):
    """Process a multi-view video file to extract pose landmarks and save to CSV."""
    processor = MultiViewPIMProcessor(
        pose_detection=True,
        hand_detection=False,
        face_detection=False,
        num_views=num_views
    )
    
    return processor.process_multi_view_video_for_pim(video_path, movement_name, output_dir)

def process_video_file(video_path, movement_name, output_dir="pose_data"):
    """Process a single-view video file to extract pose landmarks and save to CSV."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    video_id = os.path.basename(video_path).replace('.mkv', '').replace('.mp4', '').replace(' ', '_')
    timestamp = int(time.time())
    output_file = os.path.join(output_dir, f"{movement_name}_{video_id}_{timestamp}_data.csv")

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, ['timestamp', 'landmark_id', 'x', 'y', 'z'])
        writer.writeheader()
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Processing video: {video_path} ({frame_count} frames @ {fps} FPS)")
        
        for frame_idx in tqdm(range(frame_count), desc=f"Processing {movement_name}"):
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate timestamp from frame index and fps
            timestamp = frame_idx / fps
            
            # Process frame with MediaPipe
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                # Round to 3 decimals to match sequence prep
                ts = round(timestamp, 3)
                for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
                    writer.writerow({
                        'timestamp': ts, 
                        'landmark_id': landmark_id,
                        'x': landmark.x, 
                        'y': landmark.y, 
                        'z': landmark.z
                    })

    cap.release()
    print(f"Processing complete. Data saved to {output_file}")
    return output_file

def process_video_directory(input_dir, movement_name, output_dir="pose_data", multi_view=False):
    """Process all video files in a directory for a specific movement."""
    video_extensions = ["*.mkv", "*.mp4", "*.avi", "*.mov"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not video_files:
        print(f"No video files found in directory: {input_dir}")
        return False
    
    print(f"Found {len(video_files)} video files for movement: {movement_name}")
    success_count = 0
    
    for video_file in video_files:
        if multi_view:
            if process_multi_view_video_file(video_file, movement_name, output_dir):
                success_count += 1
        else:
            if process_video_file(video_file, movement_name, output_dir):
                success_count += 1
    
    print(f"Successfully processed {success_count}/{len(video_files)} videos for {movement_name}")
    return success_count > 0

def train_model(data_dir="pose_data", movements=None, epochs=50, model_type="normal", patience=10):
    """Train PIM detection model on processed landmark data."""
    # Infer movement names from files if not provided
    if not movements:
        movements = set()
        for filename in os.listdir(data_dir):
            if filename.endswith('_data.csv'):
                parts = filename.split('_')
                if parts[-1] == "data.csv":
                    if len(parts) == 2: 
                        movement = parts[0]
                    else: 
                        movement = '_'.join(parts[:-3])
                    if movement in PIM_MOVEMENTS:
                        movements.add(movement)
        movements = list(movements)

    print(f"Training model on movements: {movements}\nModel type: {model_type}")

    # Process data
    X, y = [], []
    for idx, movement in enumerate(movements):
        movement_files = [f for f in os.listdir(data_dir) if f.startswith(f"{movement}_") and f.endswith('_data.csv')]
        if not movement_files:
            print(f"Warning: No files found for movement: {movement}")
            continue

        print(f"Found {len(movement_files)} files for movement: {movement}")
        for file in movement_files:
            sequences = prepare_sequences(os.path.join(data_dir, file))
            if len(sequences) > 0:
                X.append(sequences)
                y.extend([idx] * len(sequences))
                print(f"  - {file}: {len(sequences)} sequences")
            else:
                print(f"  - {file}: No valid sequences extracted")

    if not X:
        print("Error: No valid training data found")
        return None

    X = np.vstack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    print(f"Total sequences: {len(X)}, Shape: {X.shape}")

    # Convert to tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Stratified split on indices to keep dtype controlled
    idx = np.arange(len(X))
    try:
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=42, stratify=y.numpy()
        )
    except Exception:
        # Fallback if classes are too small for stratify
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Initialize model
    model = JointBoneEnsembleLSTM(input_dim=3, hidden_dim=128, num_layers=3, num_classes=len(movements)) \
            if model_type == "joint_bone" else \
            PIMDetectorLSTM(input_dim=3, hidden_dim=128, num_layers=3, num_classes=len(movements))
    model.to(device)

    # Handle tiny datasets gracefully
    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Not enough data to split train/test. Record more samples.")
        return None

    batch_size = max(1, min(16, len(X_train)))
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_seen = 0

        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size].to(device, non_blocking=True)
            yb = y_train[i:i+batch_size].to(device, non_blocking=True)

            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_loss += loss.item() * bs
            total_seen += bs

        epoch_loss = running_loss / max(1, total_seen)
        
        # Calculate validation loss
        model.eval()
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                xb = X_test[i:i+batch_size].to(device, non_blocking=True)
                yb = y_test[i:i+batch_size].to(device, non_blocking=True)
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_total += xb.size(0)
        val_loss = val_loss / max(1, val_total)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {time.time()-start_time:.1f}s')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                break

    # Evaluate final model
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for i in range(0, len(X_test), batch_size):
            xb = X_test[i:i+batch_size].to(device, non_blocking=True)
            yb = y_test[i:i+batch_size].to(device, non_blocking=True)
            logits, _ = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        accuracy = correct / max(1, total)
        print(f'Final Test Accuracy: {accuracy:.4f}')

    # Save best model (from early stopping)
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', f'pim_model_{model_type}.pth')
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Saving best model from early stopping (val loss: {best_val_loss:.4f})")
    torch.save({'model_state_dict': model.state_dict(), 'movements': movements, 'model_type': model_type}, model_path)
    print(f"Model saved to {model_path}")
    return model, movements

def comprehensive_video_analysis(video_path, model_path='models/pim_model_joint_bone.pth', output_csv=None):
    """Comprehensive video analysis with detailed temporal breakdown and statistics."""
    print("🔬 COMPREHENSIVE PIM VIDEO ANALYSIS SUITE")
    print("=" * 60)

    # Process video if needed
    output_dir = 'pose_data'
    movement_name = 'comprehensive_analysis'
    output_file = process_video_file(video_path, movement_name, output_dir)
    if not output_file:
        print(f"❌ Failed to process video: {video_path}")
        return

    # Load model
    try:
        model, movements, model_type = load_trained_model(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load and analyze ALL sequences
    sequences = prepare_sequences(output_file)
    if len(sequences) == 0:
        print("❌ No sequences extracted from video")
        return

    print(f"📊 Processing {len(sequences)} movement sequences...")
    print(f"📹 Video: {os.path.basename(video_path)}")
    print(f"⏱️  Duration: {len(sequences) * 0.5:.1f} seconds ({len(sequences)//120:.0f} minutes)")
    print()

    # Analyze ALL predictions
    predictions = []
    confidences = []

    print("🤖 Running model predictions...")
    for i, seq in enumerate(sequences):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{len(sequences)} sequences analyzed...")
        seq_tensor = torch.from_numpy(seq).unsqueeze(0).float().to(device)
        pred, conf, probs = predict_sequence(model, model_type, seq_tensor)
        predictions.append(pred)
        confidences.append(conf)

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    # === COMPREHENSIVE STATISTICS ===
    print("\n📈 COMPREHENSIVE STATISTICS")
    print("-" * 40)

    total_sequences = len(sequences)
    unique_preds, counts = np.unique(predictions, return_counts=True)

    # Overall statistics
    print(f"Total Sequences Analyzed: {total_sequences:,}")
    print(f"Sequence Duration: 0.5 seconds each")
    print(f"Total Video Time: {total_sequences * 0.5 / 60:.1f} minutes")
    print(f"Average Confidence: {confidences.mean():.3f}")
    print(f"Confidence Range: {confidences.min():.3f} - {confidences.max():.3f}")
    print()

    # Movement breakdown
    print("MOVEMENT DISTRIBUTION:")
    print("-" * 25)
    movement_stats = []
    for pred_class, count in zip(unique_preds, counts):
        movement_name = movements[pred_class]
        percentage = count / total_sequences * 100
        avg_conf = confidences[predictions == pred_class].mean()
        std_conf = confidences[predictions == pred_class].std()
        movement_stats.append((movement_name, count, percentage, avg_conf, std_conf))

        confidence_indicator = "🎯" if avg_conf > 0.8 else "⚠️" if avg_conf > 0.6 else "❓"
        print(f"{confidence_indicator} {movement_name.upper():<15} {count:>6,} seq ({percentage:>5.1f}%) | Conf: {avg_conf:.3f} ± {std_conf:.3f}")

    print()

    # === TEMPORAL ANALYSIS - COMPLETE VIDEO ===
    print("⏰ TEMPORAL ANALYSIS - COMPLETE VIDEO BREAKDOWN")
    print("-" * 50)

    # Analyze in 1-minute windows (120 sequences = 60 seconds)
    window_size = 120  # 1 minute windows
    time_windows = []

    print("Analyzing movement patterns over time...")
    for i in range(0, len(predictions), window_size):
        window_preds = predictions[i:i+window_size]
        window_confs = confidences[i:i+window_size]

        if len(window_preds) > 0:
            # Get dominant movement in this window
            most_common = collections.Counter(window_preds).most_common(1)[0]
            movement = movements[most_common[0]]
            percentage = most_common[1] / len(window_preds) * 100
            avg_conf = window_confs.mean()

            start_time = i * 0.5  # Convert sequences to seconds
            end_time = min((i + window_size) * 0.5, len(sequences) * 0.5)

            time_windows.append({
                'start_time': start_time,
                'end_time': end_time,
                'dominant_movement': movement,
                'percentage': percentage,
                'avg_confidence': avg_conf,
                'total_sequences': len(window_preds)
            })

    # Display temporal breakdown
    print(f"{'Time Range':<12} {'Movement':<15} {'Dominance':<10} {'Confidence':<12} {'Sequences'}")
    print("-" * 70)

    for window in time_windows:
        start_min = int(window['start_time'] // 60)
        start_sec = int(window['start_time'] % 60)
        end_min = int(window['end_time'] // 60)
        end_sec = int(window['end_time'] % 60)

        time_range = f"{start_min:02d}:{start_sec:02d}-{end_min:02d}:{end_sec:02d}"
        movement = window['dominant_movement'].upper()
        dominance = f"{window['percentage']:.1f}%"
        confidence = f"{window['avg_confidence']:.3f}"
        sequences = f"{window['total_sequences']}"

        print(f"{time_range:<12} {movement:<15} {dominance:<10} {confidence:<12} {sequences}")

    print()

    # === MOVEMENT TRANSITIONS ===
    print("🔄 MOVEMENT TRANSITIONS")
    print("-" * 30)

    transitions = []
    for i in range(1, len(time_windows)):
        prev_movement = time_windows[i-1]['dominant_movement']
        curr_movement = time_windows[i]['dominant_movement']
        if prev_movement != curr_movement:
            transition_time = time_windows[i]['start_time']
            transitions.append((transition_time, prev_movement, curr_movement))

    if transitions:
        print(f"Found {len(transitions)} movement transitions:")
        for time_sec, from_mov, to_mov in transitions:
            minutes = int(time_sec // 60)
            seconds = int(time_sec % 60)
            print(f"  {minutes:02d}:{seconds:02d} - {from_mov.upper()} → {to_mov.upper()}")
    else:
        print("No significant movement transitions detected")

    print()

    # === DETAILED CONFIDENCE ANALYSIS ===
    print("🎯 CONFIDENCE ANALYSIS")
    print("-" * 25)

    # Confidence distribution
    conf_ranges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    conf_labels = ["Very Low (0.0-0.5)", "Low (0.5-0.7)", "High (0.7-0.9)", "Very High (0.9-1.0)"]

    print("Confidence Distribution:")
    for (low, high), label in zip(conf_ranges, conf_labels):
        count = np.sum((confidences >= low) & (confidences < high))
        percentage = count / len(confidences) * 100
        print(f"  {label}: {count:,} sequences ({percentage:.1f}%)")

    print()

    # === SUMMARY INSIGHTS ===
    print("💡 ANALYSIS INSIGHTS")
    print("-" * 25)

    # Find dominant movement
    dominant_movement = movements[unique_preds[np.argmax(counts)]]
    dominant_percentage = counts[np.argmax(counts)] / total_sequences * 100

    print(f"🏆 PRIMARY MOVEMENT: {dominant_movement.upper()} ({dominant_percentage:.1f}% of video)")

    # Movement diversity
    movement_count = len(unique_preds)
    if movement_count == 1:
        print("🎯 MOVEMENT CONSISTENCY: Video shows consistent movement patterns")
    elif movement_count <= 3:
        print(f"🔄 MODERATE VARIETY: {movement_count} different movement types detected")
    else:
        print(f"🌈 HIGH VARIETY: {movement_count} different movement types detected")

    # Confidence assessment
    high_conf_percentage = np.sum(confidences > 0.8) / len(confidences) * 100
    if high_conf_percentage > 80:
        print("✅ HIGH CONFIDENCE: Model is very confident in its predictions")
    elif high_conf_percentage > 60:
        print("⚠️  MODERATE CONFIDENCE: Model shows reasonable confidence")
    else:
        print("❓ LOW CONFIDENCE: Model predictions may be uncertain")

    # Medical context
    medical_movements = ['decerebrate', 'decorticate', 'ballistic']
    detected_medical = [mov for mov in medical_movements if mov in [movements[p] for p in unique_preds]]

    if detected_medical:
        print(f"🏥 MEDICAL CONTEXT: Detected {len(detected_medical)} neurological movement patterns: {', '.join(detected_medical).upper()}")

    # === EXPORT DATA ===
    if output_csv:
        print(f"\n💾 Exporting detailed data to: {output_csv}")
        export_comprehensive_data(time_windows, movement_stats, transitions, output_csv)

    print("\n" + "=" * 60)
    print("🎉 COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 60)

def run_comprehensive_testing_suite(video_paths, model_path='models/pim_model_joint_bone.pth', output_dir='testing_results'):
    """Run comprehensive analysis on multiple videos and generate comparison report."""
    os.makedirs(output_dir, exist_ok=True)

    print("🧪 COMPREHENSIVE PIM TESTING SUITE")
    print("=" * 70)
    print(f"📊 Analyzing {len(video_paths)} videos...")
    print(f"🎯 Model: {model_path}")
    print(f"📁 Output directory: {output_dir}")
    print()

    all_results = []
    summary_data = []

    for i, video_path in enumerate(video_paths, 1):
        print(f"🎬 VIDEO {i}/{len(video_paths)}: {os.path.basename(video_path)}")
        print("-" * 50)

        try:
            # Run comprehensive analysis
            result = analyze_video_comprehensive(video_path, model_path, save_csv=False)
            if result:
                all_results.append(result)

                # Extract key metrics for summary
                summary_data.append({
                    'video': os.path.basename(video_path),
                    'duration_minutes': result['total_time_minutes'],
                    'total_sequences': result['total_sequences'],
                    'dominant_movement': result['dominant_movement'],
                    'dominant_percentage': result['dominant_percentage'],
                    'movement_variety': result['movement_variety'],
                    'avg_confidence': result['avg_confidence'],
                    'high_conf_percentage': result['high_conf_percentage'],
                    'transitions_count': result['transitions_count'],
                    'medical_movements': ', '.join(result['medical_movements'])
                })

                print(f"✅ Analysis complete for {os.path.basename(video_path)}")
            else:
                print(f"❌ Analysis failed for {os.path.basename(video_path)}")

        except Exception as e:
            print(f"❌ Error analyzing {os.path.basename(video_path)}: {e}")

        print()

    if not all_results:
        print("❌ No videos were successfully analyzed")
        return

    # Generate comparison report
    generate_comparison_report(summary_data, all_results, output_dir)

    print("🎉 TESTING SUITE COMPLETE")
    print(f"📊 Results saved to: {output_dir}")

def analyze_video_comprehensive(video_path, model_path, save_csv=True):
    """Run comprehensive analysis and return structured results."""
    # Process video
    output_dir = 'pose_data'
    movement_name = f"test_{os.path.basename(video_path).replace('.', '_')}"
    output_file = process_video_file(video_path, movement_name, output_dir)
    if not output_file:
        return None

    # Load model
    try:
        model, movements, model_type = load_trained_model(model_path)
    except Exception as e:
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load sequences
    sequences = prepare_sequences(output_file)
    if len(sequences) == 0:
        return None

    # Run predictions
    predictions = []
    confidences = []

    for seq in sequences:
        seq_tensor = torch.from_numpy(seq).unsqueeze(0).float().to(device)
        pred, conf, probs = predict_sequence(model, model_type, seq_tensor)
        predictions.append(pred)
        confidences.append(conf)

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    # Calculate statistics
    total_sequences = len(sequences)
    total_time_minutes = total_sequences * 0.5 / 60
    unique_preds, counts = np.unique(predictions, return_counts=True)

    # Movement distribution
    movement_stats = []
    for pred_class, count in zip(unique_preds, counts):
        movement_name = movements[pred_class]
        percentage = count / total_sequences * 100
        avg_conf = confidences[predictions == pred_class].mean()
        std_conf = confidences[predictions == pred_class].std()
        movement_stats.append((movement_name, count, percentage, avg_conf, std_conf))

    # Dominant movement
    dominant_movement = movements[unique_preds[np.argmax(counts)]]
    dominant_percentage = counts[np.argmax(counts)] / total_sequences * 100

    # Temporal analysis (1-minute windows)
    time_windows = []
    window_size = 120

    for i in range(0, len(predictions), window_size):
        window_preds = predictions[i:i+window_size]
        window_confs = confidences[i:i+window_size]

        if len(window_preds) > 0:
            most_common = collections.Counter(window_preds).most_common(1)[0]
            movement = movements[most_common[0]]
            percentage = most_common[1] / len(window_preds) * 100
            avg_conf = window_confs.mean()

            start_time = i * 0.5
            end_time = min((i + window_size) * 0.5, len(sequences) * 0.5)

            time_windows.append({
                'start_time': start_time,
                'end_time': end_time,
                'dominant_movement': movement,
                'percentage': percentage,
                'avg_confidence': avg_conf,
                'total_sequences': len(window_preds)
            })

    # Movement transitions
    transitions = []
    for i in range(1, len(time_windows)):
        prev_movement = time_windows[i-1]['dominant_movement']
        curr_movement = time_windows[i]['dominant_movement']
        if prev_movement != curr_movement:
            transition_time = time_windows[i]['start_time']
            transitions.append((transition_time, prev_movement, curr_movement))

    # Confidence analysis
    high_conf_percentage = np.sum(confidences > 0.8) / len(confidences) * 100

    # Medical context
    medical_movements = ['decerebrate', 'decorticate', 'ballistic']
    detected_medical = [mov for mov in medical_movements if mov in [movements[p] for p in unique_preds]]

    # Movement variety
    movement_variety = len(unique_preds)
    if movement_variety == 1:
        variety_desc = "Consistent"
    elif movement_variety <= 3:
        variety_desc = "Moderate"
    else:
        variety_desc = "High"

    # Compile results
    results = {
        'video_path': video_path,
        'total_sequences': total_sequences,
        'total_time_minutes': total_time_minutes,
        'avg_confidence': float(confidences.mean()),
        'confidence_range': (float(confidences.min()), float(confidences.max())),
        'movement_stats': movement_stats,
        'dominant_movement': dominant_movement,
        'dominant_percentage': dominant_percentage,
        'movement_variety': movement_variety,
        'variety_description': variety_desc,
        'time_windows': time_windows,
        'transitions': transitions,
        'transitions_count': len(transitions),
        'high_conf_percentage': high_conf_percentage,
        'medical_movements': detected_medical,
        'movements_detected': [movements[p] for p in unique_preds]
    }

    # Save individual CSV if requested
    if save_csv:
        csv_filename = f"{os.path.basename(video_path).replace('.', '_')}_analysis.csv"
        csv_path = os.path.join('testing_results', csv_filename)
        export_comprehensive_data(time_windows, movement_stats, transitions, csv_path)

    return results

def generate_comparison_report(summary_data, all_results, output_dir):
    """Generate a comprehensive comparison report across all analyzed videos."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    report_file = os.path.join(output_dir, f'comprehensive_comparison_report_{timestamp}.txt')
    csv_file = os.path.join(output_dir, f'comparison_summary_{timestamp}.csv')

    # Text report
    with open(report_file, 'w') as f:
        f.write("🧪 COMPREHENSIVE PIM TESTING SUITE - COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Videos Analyzed: {len(summary_data)}\n\n")

        # Summary statistics
        f.write("📊 OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        total_videos = len(summary_data)
        avg_duration = np.mean([d['duration_minutes'] for d in summary_data])
        avg_sequences = np.mean([d['total_sequences'] for d in summary_data])
        avg_confidence = np.mean([d['avg_confidence'] for d in summary_data])

        f.write(f"Total Videos: {total_videos}\n")
        f.write(f"Average Duration: {avg_duration:.1f} minutes\n")
        f.write(f"Average Sequences: {avg_sequences:,.0f}\n")
        f.write(f"Average Confidence: {avg_confidence:.3f}\n\n")

        # Movement patterns across videos
        f.write("🎯 MOVEMENT PATTERNS ACROSS VIDEOS\n")
        f.write("-" * 40 + "\n")

        all_dominant = [d['dominant_movement'] for d in summary_data]
        dominant_counts = collections.Counter(all_dominant)

        f.write("Most Common Dominant Movements:\n")
        for movement, count in dominant_counts.most_common():
            percentage = count / total_videos * 100
            f.write(f"  {movement.upper()}: {count} videos ({percentage:.1f}%)\n")

        f.write("\n")

        # Individual video summaries
        f.write("📹 INDIVIDUAL VIDEO SUMMARIES\n")
        f.write("-" * 40 + "\n")

        for data in summary_data:
            f.write(f"\n🎬 {data['video']}\n")
            f.write(f"   Duration: {data['duration_minutes']:.1f} minutes\n")
            f.write(f"   Sequences: {data['total_sequences']:,}\n")
            f.write(f"   Dominant: {data['dominant_movement'].upper()} ({data['dominant_percentage']:.1f}%)\n")
            f.write(f"   Movement Types: {data['movement_variety']}\n")
            f.write(f"   Avg Confidence: {data['avg_confidence']:.3f}\n")
            f.write(f"   High Confidence: {data['high_conf_percentage']:.1f}%\n")
            f.write(f"   Transitions: {data['transitions_count']}\n")
            if data['medical_movements']:
                f.write(f"   Medical Movements: {data['medical_movements']}\n")

        # Comparative insights
        f.write("\n💡 COMPARATIVE INSIGHTS\n")
        f.write("-" * 40 + "\n")

        # Confidence comparison
        conf_levels = [(d['avg_confidence'], d['video']) for d in summary_data]
        conf_levels.sort(reverse=True)
        f.write("Confidence Ranking (highest to lowest):\n")
        for i, (conf, video) in enumerate(conf_levels[:5], 1):
            f.write(f"  {i}. {video}: {conf:.3f}\n")

        # Movement diversity
        diversity_levels = [(d['movement_variety'], d['video']) for d in summary_data]
        diversity_levels.sort(reverse=True)
        f.write("\nMovement Diversity Ranking (most to least diverse):\n")
        for i, (diversity, video) in enumerate(diversity_levels[:5], 1):
            f.write(f"  {i}. {video}: {diversity} movement types\n")

        # Duration comparison
        duration_levels = [(d['duration_minutes'], d['video']) for d in summary_data]
        duration_levels.sort(reverse=True)
        f.write("\nDuration Ranking (longest to shortest):\n")
        for i, (duration, video) in enumerate(duration_levels[:5], 1):
            f.write(f"  {i}. {video}: {duration:.1f} minutes\n")

    # CSV summary
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['video', 'duration_minutes', 'total_sequences', 'dominant_movement',
                     'dominant_percentage', 'movement_variety', 'avg_confidence',
                     'high_conf_percentage', 'transitions_count', 'medical_movements']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in summary_data:
            writer.writerow(data)

    print(f"📄 Detailed report saved to: {report_file}")
    print(f"📊 Summary CSV saved to: {csv_file}")

def load_trained_model(model_path):
    """Load a trained PIM detection model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu')
    
    # Handle different model save formats
    if 'classes' in ckpt:
        # STGCN model format
        movements = ckpt['classes']
        model_type = 'stgcn'
        from stgcn_model import STGCNTwoStream
        from stgcn_graph import build_partitions
        edges = build_partitions()
        model = STGCNTwoStream(len(movements), edges)
    elif 'movements' in ckpt:
        # LSTM model format
        movements, model_type = ckpt['movements'], ckpt.get('model_type', 'normal')
        model = JointBoneEnsembleLSTM(input_dim=3, hidden_dim=128, num_layers=3, num_classes=len(movements)) \
                if model_type == 'joint_bone' else \
                PIMDetectorLSTM(input_dim=3, hidden_dim=128, num_layers=3, num_classes=len(movements))
    else:
        raise ValueError(f"Unknown model format in {model_path}. Expected 'classes' (STGCN) or 'movements' (LSTM) key.")

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, movements, model_type

def predict_sequence(model, model_type, sequence_tensor):
    """Return (predicted_class_index, confidence, probability_vector)."""
    with torch.no_grad():
        if model_type == 'stgcn':
            # ST-GCN expects [N,C,T,V] format
            from stgcn_graph import mp_edges
            from stgcn_features import sequences_to_stgcn_batches
            EDGES = mp_edges()
            Xj, Xb = sequences_to_stgcn_batches([sequence_tensor.numpy()], EDGES)
            xj = torch.from_numpy(Xj).to(sequence_tensor.device)
            xb = torch.from_numpy(Xb).to(sequence_tensor.device)

            if hasattr(model, 'bones'):  # two-stream
                logits = model(xj, xb)
            else:
                logits = model(xj)
        else:
            # Original joint_bone model
            logits, _ = model(sequence_tensor)

        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return int(pred.item()), float(conf.item()), probs.squeeze(0).cpu().numpy()

# Color coding for movement severity
COLOR_RULES = {
    'critical': (0, 0, 255), 'concerning': (0, 165, 255),
    'minor': (0, 255, 255), 'normal': (0, 255, 0)
}

def movement_color(name):
    """Get color for movement based on severity."""
    if name in ['seizure', 'myoclonus']: return COLOR_RULES['critical']
    if name in ['tremor', 'dystonia', 'chorea']: return COLOR_RULES['concerning']
    if name == 'tics': return COLOR_RULES['minor']
    return COLOR_RULES['normal']

def detection_loop(model_path, patient_id=None, debug=False):
    """Main real-time detection loop."""
    try:
        model, movements, model_type = load_trained_model(model_path)
    except Exception as e:
        print(f"❌ {e}")
        return False

    # Move to best available device for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    pose = mp.solutions.pose.Pose()
    frame_buffer = collections.deque(maxlen=30)
    pred_q, conf_q = collections.deque(maxlen=10), collections.deque(maxlen=10)
    raw_preds = collections.deque(maxlen=50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Could not open camera.')
        return False

    prev_time, detection_count = 0, 0
    mode_label = 'DEBUG MODE' if debug else 'DETECTION'
    print(f"🏥 PIM System {mode_label} | Movements: {movements}\nKeys: q=quit d=stats (debug only)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
            frame_buffer.append(landmarks)
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            cv2.putText(frame, f"Buffer: {len(frame_buffer)}/30", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            if len(frame_buffer) == 30:  # Full buffer
                seq = np.array(frame_buffer, dtype=np.float32)
                seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(device, non_blocking=True)
                pred, conf, probs = predict_sequence(model, model_type, seq_tensor)

                pred_q.append(pred); conf_q.append(conf)
                final_pred = max(set(pred_q), key=pred_q.count)
                avg_conf = sum(conf_q)/len(conf_q)
                movement = movements[final_pred]

                # Display the detected movement
                cv2.putText(frame, f"{movement.upper()} {avg_conf:.2f}", (10,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, movement_color(movement), 2)

                if debug:
                    raw_preds.append({'pred': final_pred, 'conf': avg_conf, 'scores': probs, 'ts': time.time()})
                    top_idx = np.argsort(probs)[-3:][::-1]
                    for i, idx in enumerate(top_idx):
                        cv2.putText(frame, f"{i+1}.{movements[idx]}:{probs[idx]:.2f}",
                                    (10,250+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                detection_count += 1

        # HUD
        current = time.time()
        fps = 1/(current-prev_time) if prev_time else 0
        prev_time = current
        cv2.putText(frame, f'FPS:{int(fps)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f'Patient:{patient_id or "Unknown"}', (10,190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f'Detections:{detection_count}', (10,215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow(f'PIM {mode_label}', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif debug and k == ord('d'):
            print('\n🔍 Stats:')
            counts = collections.Counter([movements[p['pred']] for p in raw_preds])
            for m, c in counts.most_common(): print(f"  {m}: {c}")

    cap.release(); cv2.destroyAllWindows()
    print(f"\n🏥 Session Complete | detections={detection_count}")
    return True

def real_time_pim_detection(model_path='models/pim_model_normal.pth', patient_id=None):
    """Run real-time PIM detection."""
    return detection_loop(model_path, patient_id, debug=False)

def debug_real_time_detection(model_path='models/pim_model_normal.pth', patient_id=None):
    """Run real-time PIM detection with debug information."""
    return detection_loop(model_path, patient_id, debug=True)

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Multi-View PIM Detection System')
    parser.add_argument('--mode', type=str, default='help',
                        choices=['process_video', 'process_directory', 'train', 'detect', 'debug', 'analyze', 'test_suite', 'help'],
                        help='Operation mode')
    parser.add_argument('--movement', type=str, help='Movement name for processing')
    parser.add_argument('--patient_id', type=str, help='Patient ID for detection mode')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--model_type', type=str, default='normal',
                        choices=['normal', 'joint_bone'],
                        help='Model type: normal or joint_bone (ensemble)')
    parser.add_argument('--model_path', type=str, help='Path to model file for detection')
    parser.add_argument('--video_path', type=str, help='Path to video file for processing')
    parser.add_argument('--video_dir', type=str, help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, default='testing_results', help='Output directory for test suite results')
    
    args = parser.parse_args()
    
    # Create necessary directories
    for dir_name in ['pose_data', 'models', 'output']: 
        os.makedirs(dir_name, exist_ok=True)

    if args.mode == 'help' or len(sys.argv) == 1:
        print("🏥 Multi-View PIM Detection System")
        print("=" * 55)
        print("\nQuick Start Guide:")
        print("1. Process multi-view videos: python pim_detection_system.py --mode process_directory --movement normal --video_dir /videos/normal/ --multi_view")
        print("2. Train PIM model:          python pim_detection_system.py --mode train --model_type joint_bone")
        print("3. Start detection:          python pim_detection_system.py --mode detect --patient_id John_Doe")
        print("\nAvailable modes:")
        print("  • process_video     - Process a single video file")
        print("  • process_directory - Process all videos in a directory")  
        print("  • train             - Train the PIM detection model")
        print("  • debug             - Run detection with detailed debugging info")
        print("  • detect            - Run normal PIM detection")
        print("  • analyze           - Comprehensive video analysis with detailed output")
        print("  • test_suite        - Run comprehensive testing suite on multiple videos")
        print("\nModel types:")
        print("  • normal     - PIMDetectorLSTM (default)")
        print("  • joint_bone - Ensemble of joint + bone features (recommended)")
        print("\nAvailable PIM movements:", list(PIM_MOVEMENTS.keys()))
        print("\nMulti-view processing:")
        print("  Use --multi_view flag to process 4-view videos (crops rightmost view)")
        print("  Without flag: processes standard single-view videos")
        return

    if args.mode == 'process_video':
        if not args.movement or not args.video_path:
            print("Error: Must specify --movement name and --video_path")
            print("Available PIM movements:", list(PIM_MOVEMENTS.keys()))
            return
        
        if args.multi_view:
            process_multi_view_video_file(args.video_path, args.movement)
        else:
            process_video_file(args.video_path, args.movement)
        
    elif args.mode == 'process_directory':
        if not args.movement or not args.video_dir:
            print("Error: Must specify --movement name and --video_dir")
            print("Available PIM movements:", list(PIM_MOVEMENTS.keys()))
            return
        
        process_video_directory(args.video_dir, args.movement, multi_view=args.multi_view)

    elif args.mode == 'train':
        # Check which movements have data
        available_movements = []
        for movement in PIM_MOVEMENTS:
            movement_files = [f for f in os.listdir('pose_data')
                              if f.startswith(f"{movement}_") and f.endswith('_data.csv')]
            if movement_files: 
                available_movements.append(movement)

        if not available_movements:
            print("❌ No PIM movement data files found in pose_data directory")
            print("Please process video files first using:")
            print("  python pim_detection_system.py --mode process_video --movement normal --video_path /path/to/video.mp4 --multi_view")
            return

        print(f"✅ Found data for PIM movements: {available_movements}")
        train_model(movements=available_movements, epochs=args.epochs, model_type=args.model_type, patience=args.patience)

    elif args.mode in ['detect', 'debug']:
        model_path = args.model_path or os.path.join('models', f'pim_model_{args.model_type}.pth')
        if not os.path.exists(model_path):
            print(f"❌ No trained model found at {model_path}")
            print(f"Please train a {args.model_type} model first using --mode train --model_type {args.model_type}")
            return

        patient_id = args.patient_id or f"Patient_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"{'🔍' if args.mode=='debug' else '🏥'} Starting {'DEBUG ' if args.mode=='debug' else ''}PIM Detection for Patient: {patient_id}")

    elif args.mode == 'analyze':
        if not args.video_path:
            print("Error: Must specify --video_path for analysis mode")
            return
        
        model_path = args.model_path or os.path.join('models', f'pim_model_{args.model_type}.pth')
        if not os.path.exists(model_path):
            print(f"❌ No trained model found at {model_path}")
            return

        print(f"🔍 Analyzing video: {args.video_path}")
        comprehensive_video_analysis(args.video_path, model_path)

    elif args.mode == 'test_suite':
        if not args.video_dir:
            print("Error: Must specify --video_dir for test_suite mode")
            print("Example: python pim_detection_system.py --mode test_suite --video_dir 'C:\\Users\\Mike\\Desktop\\napa'")
            return

        model_path = args.model_path or os.path.join('models', f'pim_model_{args.model_type}.pth')
        if not os.path.exists(model_path):
            print(f"❌ No trained model found at {model_path}")
            return

        # Find all video files in directory
        video_extensions = ['*.mkv', '*.mp4', '*.avi', '*.mov']
        video_paths = []
        for ext in video_extensions:
            video_paths.extend(glob.glob(os.path.join(args.video_dir, ext)))

        if not video_paths:
            print(f"❌ No video files found in {args.video_dir}")
            print(f"Supported formats: {', '.join(video_extensions)}")
            return

        print(f"📹 Found {len(video_paths)} video files to analyze")
        for video in video_paths:
            print(f"  • {os.path.basename(video)}")

        run_comprehensive_testing_suite(video_paths, model_path, args.output_dir)

if __name__ == "__main__":
    main()