"""
Intelligent PIM Video Clipper
Automatically identifies and extracts training segments from longer videos using the trained PIM detector
"""

import os
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Try to import MoviePy, fallback to subprocess if not available
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    VideoFileClip = None

from pim_detection_system import load_trained_model
from mediapipe_processor import MultiViewPIMProcessor, PIM_MOVEMENTS


class PIMVideoClipper:
    """
    Intelligent video clipper that uses a trained PIM model to identify segments worth extracting.
    """

    def __init__(self, model_path: str = "models/pim_model_joint_bone.pth", confidence_threshold: float = 0.7):
        """
        Initialize the clipper with the trained model.

        Args:
            model_path: Path to trained PIM model.
            confidence_threshold: Minimum confidence to consider a detection valid.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        # ONLY pathological movements - class 0 (normal) is explicitly ignored
        self.pathological_movements = {
            'decorticate': 1, 'dystonia': 2, 'chorea': 3,
            'myoclonus': 4, 'decerebrate': 5, 'fencer posture': 6,
            'ballistic': 7, 'tremor': 8, 'versive head': 9
        }
        self.movement_names = {v: k for k, v in self.pathological_movements.items()}

        # Load the trained model
        print("Loading trained PIM model...")
        self.model, self.movements, self.model_type = load_trained_model(model_path)
        self.model.eval()

        # Initialize MediaPipe processor
        self.processor = MultiViewPIMProcessor(
            pose_detection=True,
            hand_detection=False,
            face_detection=False,
            num_views=1  # Single view for analysis
        )

        print("Model loaded - PATHOLOGICAL MOVEMENTS ONLY:")
        for class_id, name in self.movement_names.items():
            print(f"   {class_id}: {name}")
        print("   Class 0 (normal) - IGNORED")

    def analyze_video_segments(
        self,
        video_path: str,
        segment_length: int = 30,
        overlap: int = 15,
        min_segment_duration: float = 30.0,
        max_segment_duration: float = 40.0
    ) -> Dict:
        """
        Analyze video and identify segments with meaningful PIM activity.

        Logic: if it identifies a specific (non-'normal') posture consistently for >= min_segment_duration seconds,
        it's a candidate for clipping.

        Args:
            video_path: Path to video to analyze.
            segment_length: Length of the model's temporal analysis window in frames.
            overlap: (Reserved) Amount of overlap between windows. Currently not used; sliding window runs every frame.
            min_segment_duration: Minimum duration for a clip in seconds (default 30.0).

        Returns:
            Dictionary with analysis results and suggested clips.
        """
        print(f"\n🎬 Analyzing video: {Path(video_path).name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 1e-3:
            print("⚠️ FPS unreadable; defaulting to 30.0")
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")

        # Storage for analysis results
        frame_predictions: List[int] = []
        frame_confidences: List[float] = []
        frame_timestamps: List[float] = []
        pose_qualities: List[float] = []

        # Process video frame by frame with a sliding buffer
        frame_buffer: deque = deque(maxlen=segment_length)
        frame_idx = 0

        with tqdm(total=total_frames, desc="Analyzing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps

                # Extract pose landmarks
                landmarks = self.processor.extract_pose_landmarks_from_frame(frame)

                if landmarks is not None:
                    # Add to buffer and record quality
                    frame_buffer.append(landmarks)
                    pose_quality = self._calculate_pose_quality(landmarks)
                    pose_qualities.append(pose_quality)

                    if len(frame_buffer) == segment_length:
                        # Prepare sequence for model
                        sequence = np.array(list(frame_buffer))
                        sequence = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension

                        # Get prediction
                        with torch.no_grad():
                            output = self.model(sequence)
                            # Handle tuple output (some models return tuple)
                            if isinstance(output, tuple):
                                output = output[0]
                            probabilities = torch.softmax(output, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)

                        frame_predictions.append(int(predicted.item()))
                        frame_confidences.append(confidence.item())
                        frame_timestamps.append(timestamp)
                    else:
                        # Not enough frames yet
                        frame_predictions.append(-1)  # Unknown
                        frame_confidences.append(0.0)
                        frame_timestamps.append(timestamp)
                        # (Do NOT append extra pose_qualities here)
                else:
                    # No pose detected at this frame
                    frame_predictions.append(-1)
                    frame_confidences.append(0.0)
                    frame_timestamps.append(timestamp)
                    pose_qualities.append(0.0)

                frame_idx += 1
                pbar.update(1)

        cap.release()

        # Analyze results and find clips
        analysis_results = {
            "video_path": video_path,
            "fps": fps,
            "duration": duration,
            "total_frames": total_frames,
            "frame_predictions": frame_predictions,
            "frame_confidences": frame_confidences,
            "frame_timestamps": frame_timestamps,
            "pose_qualities": pose_qualities,
            "movement_counts": self._count_movements(frame_predictions, frame_confidences),
            "suggested_clips": self._find_clip_segments(
                frame_predictions, frame_confidences, frame_timestamps, fps, 
                min_duration=min_segment_duration, max_duration=max_segment_duration
            ),
            "quality_analysis": self._analyze_pose_quality(pose_qualities, frame_timestamps),
        }

        return analysis_results

    def _calculate_pose_quality(self, landmarks: np.ndarray) -> float:
        """Calculate pose detection quality score based on visibility and confidence."""
        if landmarks is None:
            return 0.0

        # landmarks: shape (num_points, 3) -> (x, y, visibility)
        visible_mask = landmarks[:, 2] > 0.5  # Visibility scores
        visible_points = int(np.sum(visible_mask))
        total_points = int(landmarks.shape[0])

        # Average visibility/confidence for visible points
        visible_confidences = landmarks[visible_mask, 2]
        avg_confidence = float(np.mean(visible_confidences)) if visible_confidences.size > 0 else 0.0

        visibility_ratio = (visible_points / total_points) if total_points > 0 else 0.0
        quality_score = (visibility_ratio * 0.6) + (avg_confidence * 0.4)

        return float(quality_score)

    def _count_movements(self, predictions: List[int], confidences: List[float]) -> Dict[str, int]:
        """Count detected movements with confidence filtering."""
        movement_counts: Dict[str, int] = defaultdict(int)

        for pred, conf in zip(predictions, confidences):
            if pred >= 0 and conf >= self.confidence_threshold:
                name = self.movement_names.get(pred, f"unknown_{pred}")
                movement_counts[name] += 1

        return dict(movement_counts)
    
    def _is_rapid_switching(self, predictions: List[int], confidences: List[float], 
                           idx: int, window_size: int = 6) -> bool:
        """
        Check if there's rapid switching around this frame (indicates normal movement)
        """
        if idx < window_size // 2 or idx >= len(predictions) - window_size // 2:
            return False
        
        start_idx = idx - window_size // 2
        end_idx = idx + window_size // 2
        window_preds = predictions[start_idx:end_idx]
        window_confs = confidences[start_idx:end_idx]
        
        # Only consider high confidence predictions
        valid_preds = [p for p, c in zip(window_preds, window_confs) 
                      if p >= 0 and c >= self.confidence_threshold]
        
        if len(valid_preds) < 3:  # Not enough valid predictions
            return False
            
        # Count unique movements in window
        unique_movements = len(set(valid_preds))
        
        # If we have 3+ different movements in a 6-frame window, it's rapid switching (normal)
        return unique_movements >= 3

    def _find_clip_segments(
        self,
        predictions: List[int],
        confidences: List[float],
        timestamps: List[float],
        fps: float,
        min_duration: float = 30.0,
        max_duration: float = 40.0
    ) -> List[Dict]:
        """
        Find continuous segments with high-confidence PIM detections.
        Uses the rule: if a non-'normal' posture persists for >= min_duration seconds, it's a candidate.
        Automatically splits clips longer than max_duration.

        Args:
            min_duration: Minimum duration in seconds for a clip (default 30.0).
            max_duration: Maximum duration in seconds for a clip (default 40.0).
        """
        clips: List[Dict] = []
        current_clip: Optional[Dict] = None
        min_frames = int(min_duration * fps) if fps > 0 else 0

        print(f"🎯 Looking for consistent movements lasting {min_duration}+ seconds (~{min_frames} frames)")

        for i, (pred, conf, timestamp) in enumerate(zip(predictions, confidences, timestamps)):
            is_high_conf = (pred >= 0 and conf >= self.confidence_threshold)
            # Only pathological movements (class 1-9) - exclude class 0 completely
            is_pathological = is_high_conf and pred > 0

            if is_pathological:
                movement_name = self.movement_names.get(pred, f"unknown_{pred}")

                if current_clip is None:
                    # Start new clip
                    current_clip = {
                        "start_time": timestamp,
                        "start_frame": i,
                        "movement_type": movement_name,
                        "movement_id": pred,
                        "max_confidence": conf,
                        "avg_confidence": conf,
                        "frame_count": 1,
                        "confidence_sum": conf,
                        "consistent_duration": 0.0,
                    }
                elif current_clip["movement_id"] == pred:
                    # Continue current clip with same movement
                    current_clip["frame_count"] += 1
                    current_clip["confidence_sum"] += conf
                    current_clip["avg_confidence"] = current_clip["confidence_sum"] / current_clip["frame_count"]
                    current_clip["max_confidence"] = max(current_clip["max_confidence"], conf)
                    current_clip["consistent_duration"] = timestamp - current_clip["start_time"]
                    
                    # Check if clip is getting too long - split it
                    if current_clip["consistent_duration"] >= max_duration:
                        current_clip["end_time"] = timestamp
                        current_clip["end_frame"] = i
                        current_clip["duration"] = current_clip["end_time"] - current_clip["start_time"]
                        clips.append(current_clip)
                        print(
                            f"✅ Found {current_clip['movement_type']} clip (split at max duration): "
                            f"{current_clip['duration']:.1f}s (conf: {current_clip['avg_confidence']:.2f})"
                        )
                        
                        # Start new clip with same movement type
                        current_clip = {
                            "start_time": timestamp,
                            "start_frame": i,
                            "movement_type": movement_name,
                            "movement_id": pred,
                            "max_confidence": conf,
                            "avg_confidence": conf,
                            "frame_count": 1,
                            "confidence_sum": conf,
                            "consistent_duration": 0.0,
                        }
                else:
                    # Different movement detected - end current clip if it meets duration threshold
                    if current_clip is not None and current_clip["consistent_duration"] >= min_duration:
                        current_clip["end_time"] = timestamps[i - 1] if i > 0 else timestamp
                        current_clip["end_frame"] = i - 1
                        current_clip["duration"] = current_clip["end_time"] - current_clip["start_time"]
                        clips.append(current_clip)
                        print(
                            f"✅ Found {current_clip['movement_type']} clip: "
                            f"{current_clip['duration']:.1f}s (conf: {current_clip['avg_confidence']:.2f})"
                        )

                    # Start new clip with new movement
                    current_clip = {
                        "start_time": timestamp,
                        "start_frame": i,
                        "movement_type": movement_name,
                        "movement_id": pred,
                        "max_confidence": conf,
                        "avg_confidence": conf,
                        "frame_count": 1,
                        "confidence_sum": conf,
                        "consistent_duration": 0.0,
                    }
            else:
                # Low confidence - end current clip if it meets duration threshold
                if current_clip is not None and current_clip["consistent_duration"] >= min_duration:
                    current_clip["end_time"] = timestamp
                    current_clip["end_frame"] = i
                    current_clip["duration"] = current_clip["end_time"] - current_clip["start_time"]
                    clips.append(current_clip)
                    print(
                        f"✅ Found {current_clip['movement_type']} clip: "
                        f"{current_clip['duration']:.1f}s (conf: {current_clip['avg_confidence']:.2f})"
                    )

                current_clip = None

        # Handle clip that extends to end of video
        if current_clip is not None and current_clip["consistent_duration"] >= min_duration:
            current_clip["end_time"] = timestamps[-1]
            current_clip["end_frame"] = len(timestamps) - 1
            current_clip["duration"] = current_clip["end_time"] - current_clip["start_time"]
            clips.append(current_clip)
            print(
                f"✅ Found {current_clip['movement_type']} clip: "
                f"{current_clip['duration']:.1f}s (conf: {current_clip['avg_confidence']:.2f})"
            )

        # Sort clips by confidence and remove overlapping
        clips = sorted(clips, key=lambda x: x["avg_confidence"], reverse=True)
        clips = self._remove_overlapping_clips(clips)

        print(f"📋 Final result: {len(clips)} clips meeting {min_duration}+ second duration threshold")

        return clips

    def _remove_overlapping_clips(self, clips: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
        """Remove overlapping clips, keeping the most confident ones."""
        if not clips:
            return clips

        non_overlapping = [clips[0]]  # Start with most confident clip

        for clip in clips[1:]:
            is_overlapping = False

            for existing_clip in non_overlapping:
                # Check for temporal overlap
                overlap_start = max(clip["start_time"], existing_clip["start_time"])
                overlap_end = min(clip["end_time"], existing_clip["end_time"])

                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    clip_duration = clip["end_time"] - clip["start_time"]

                    overlap_ratio = overlap_duration / max(clip_duration, 1e-6)
                    if overlap_ratio > overlap_threshold:
                        is_overlapping = True
                        break

            if not is_overlapping:
                non_overlapping.append(clip)

        return non_overlapping

    def _analyze_pose_quality(self, pose_qualities: List[float], timestamps: List[float]) -> Dict:
        """Analyze overall pose detection quality."""
        if not pose_qualities:
            return {"avg_quality": 0.0, "good_quality_ratio": 0.0, "total_frames_analyzed": 0}

        avg_quality = float(np.mean(pose_qualities))
        good_quality_frames = int(sum(1 for q in pose_qualities if q > 0.7))
        good_quality_ratio = (good_quality_frames / len(pose_qualities)) if pose_qualities else 0.0

        return {
            "avg_quality": avg_quality,
            "good_quality_ratio": good_quality_ratio,
            "total_frames_analyzed": len(pose_qualities),
        }

    def extract_clips(
        self,
        analysis_results: Dict,
        output_dir: str,
        padding_seconds: float = 2.0
    ) -> List[str]:
        """
        Extract suggested clips from the original video using MoviePy (enhanced version)

        Args:
            analysis_results: Results from analyze_video_segments
            output_dir: Directory to save extracted clips
            padding_seconds: Extra time to add before/after each clip

        Returns:
            List of paths to extracted clip files
        """
        video_path = analysis_results["video_path"]
        clips = analysis_results["suggested_clips"]

        if not clips:
            print("⚠️ No clips found to extract")
            return []

        os.makedirs(output_dir, exist_ok=True)
        video_name = Path(video_path).stem
        extracted_files: List[str] = []

        print(f"\n✂️ Extracting {len(clips)} clips using MoviePy to {output_dir}")

        for i, clip in enumerate(clips):
            # Calculate clip boundaries with padding
            start_time = max(0.0, clip["start_time"] - padding_seconds)
            end_time = min(analysis_results["duration"], clip["end_time"] + padding_seconds)
            duration = max(0.0, end_time - start_time)

            # Generate output filename
            movement_type = clip["movement_type"].replace(" ", "_")
            confidence = clip["avg_confidence"]
            output_filename = f"{video_name}_{movement_type}_{i+1:02d}_conf{confidence:.2f}_{int(start_time)}s_{int(end_time)}s.mkv"
            output_path = os.path.join(output_dir, output_filename)

            # Extract clip using MoviePy
            try:
                success = self._clip_mkv_with_moviepy(video_path, output_path, start_time, end_time)

                if success:
                    extracted_files.append(output_path)
                    print(f"✅ Extracted: {output_filename} ({duration:.1f}s, {movement_type}, conf: {confidence:.2f})")
                else:
                    print(f"❌ Failed to extract {output_filename}")

            except Exception as e:
                print(f"❌ Error extracting {output_filename}: {e}")

        return extracted_files

    def _clip_mkv_with_moviepy(self, input_file: str, output_file: str, start_time: float, end_time: float) -> bool:
        """
        Clip an MKV file using MoviePy (fixed: subclip).
        Falls back to ffmpeg on error.
        """
        if not MOVIEPY_AVAILABLE:
            print("❌ MoviePy not available, falling back to ffmpeg...")
            return self._clip_with_ffmpeg(input_file, output_file, start_time, end_time)

        try:
            with VideoFileClip(input_file) as video:
                clip = video.subclip(start_time, end_time)  # <-- fixed API
                clip.write_videofile(
                    output_file,
                    codec="libx264",
                    audio_codec="aac",
                    threads=os.cpu_count() or 4,
                    verbose=False,
                    logger=None
                )
            return True

        except Exception as e:
            print(f"Error clipping video with MoviePy: {e}")
            print("Falling back to ffmpeg...")
            return self._clip_with_ffmpeg(input_file, output_file, start_time, end_time)

    def _clip_with_ffmpeg(
        self,
        input_file: str,
        output_file: str,
        start_time: float,
        end_time: float,
        accurate: bool = True
    ) -> bool:
        """
        Fallback clipping method using ffmpeg.

        accurate=True re-encodes for frame-accurate cuts; False is faster but cuts at keyframes only.
        """
        try:
            import subprocess
            duration = max(0.001, end_time - start_time)

            if accurate:
                # Accurate seek: -ss after -i and re-encode
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i", input_file,
                    "-ss", str(start_time),
                    "-t", str(duration),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k",
                    "-y",
                    output_file
                ]
            else:
                # Fast (keyframe) seek: stream copy
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-ss", str(start_time),
                    "-i", input_file,
                    "-t", str(duration),
                    "-c", "copy",
                    "-y",
                    output_file
                ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return True
            else:
                print(f"FFmpeg error: {result.stderr}")
                return False

        except Exception as e:
            print(f"Error clipping video with ffmpeg: {e}")
            return False

    def generate_analysis_report(self, analysis_results: Dict, output_path: Optional[str] = None) -> str:
        """Generate a detailed analysis report."""
        video_name = Path(analysis_results["video_path"]).name

        if output_path is None:
            output_path = f"analysis_report_{Path(analysis_results['video_path']).stem}.txt"

        report_lines = [
            "🎬 PIM Video Analysis Report",
            "=" * 50,
            f"Video: {video_name}",
            f"Duration: {analysis_results['duration']:.1f} seconds",
            f"FPS: {analysis_results['fps']:.1f}",
            f"Total Frames: {analysis_results['total_frames']}",
            "",
            "📊 Movement Detection Summary:",
            "-" * 30
        ]

        # Add movement counts
        movement_counts = analysis_results["movement_counts"]
        if movement_counts:
            for movement, count in sorted(movement_counts.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"{movement}: {count} frames")
        else:
            report_lines.append("No significant movements detected")

        # Add quality analysis
        quality = analysis_results["quality_analysis"]
        report_lines.extend([
            "",
            "🎯 Pose Detection Quality:",
            "-" * 30,
            f"Average Quality: {quality['avg_quality']:.2f}",
            f"Good Quality Frames: {quality['good_quality_ratio'] * 100:.1f}%",
            f"Total Frames Analyzed: {quality['total_frames_analyzed']}",
        ])

        # Add suggested clips
        clips = analysis_results["suggested_clips"]
        report_lines.extend([
            "",
            f"✂️ Suggested Clips ({len(clips)} found):",
            "-" * 30
        ])

        if clips:
            for i, clip in enumerate(clips, 1):
                report_lines.extend([
                    f"Clip {i}:",
                    f"  Time: {clip['start_time']:.1f}s - {clip['end_time']:.1f}s ({clip['duration']:.1f}s)",
                    f"  Movement: {clip['movement_type']}",
                    f"  Confidence: {clip['avg_confidence']:.2f} (max: {clip['max_confidence']:.2f})",
                    f"  Frames: {clip['frame_count']}",
                    ""
                ])
        else:
            report_lines.append("No clips meet the criteria for extraction")

        # Write report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        return output_path

    def create_visualization(self, analysis_results: Dict, output_path: Optional[str] = None) -> str:
        """Create a visualization of the analysis results."""
        if output_path is None:
            output_path = f"analysis_plot_{Path(analysis_results['video_path']).stem}.png"

        timestamps = analysis_results["frame_timestamps"]
        confidences = analysis_results["frame_confidences"]
        predictions = analysis_results["frame_predictions"]
        pose_qualities = analysis_results["pose_qualities"]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

        # Plot 1: Prediction confidence over time
        ax1.plot(timestamps, confidences, alpha=0.7, linewidth=1)
        ax1.axhline(y=self.confidence_threshold, linestyle="--", alpha=0.7, label="Confidence Threshold")
        ax1.set_ylabel("Prediction Confidence")
        ax1.set_title(f'PIM Analysis: {Path(analysis_results["video_path"]).name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Movement predictions (robust to non-dense class IDs)
        valid_points = [(t, p, c) for t, p, c in zip(timestamps, predictions, confidences)
                        if p >= 0 and c >= self.confidence_threshold]

        if valid_points:
            unique_ids = sorted({p for _, p, _ in valid_points})
            id_to_color = {cid: plt.cm.get_cmap("tab10")(idx % 10) for idx, cid in enumerate(unique_ids)}
            for t, p, _ in valid_points:
                ax2.scatter(t, p, c=[id_to_color[p]], alpha=0.6, s=20)

            ax2.set_ylabel("Movement Type")
            ax2.set_yticks(unique_ids)
            ax2.set_yticklabels([self.movement_names.get(i, f"class_{i}") for i in unique_ids])
        else:
            ax2.set_ylabel("Movement Type")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Pose quality
        ax3.plot(timestamps, pose_qualities, alpha=0.7, linewidth=1)
        ax3.axhline(y=0.7, linestyle="--", alpha=0.7, label="Good Quality Threshold")
        ax3.set_ylabel("Pose Quality")
        ax3.set_xlabel("Time (seconds)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Highlight suggested clips (label only once in ax1)
        added_label = False
        for clip in analysis_results["suggested_clips"]:
            for ax in [ax1, ax2, ax3]:
                ax.axvspan(
                    clip["start_time"], clip["end_time"], alpha=0.2, color="yellow",
                    label="Suggested Clip" if ax is ax1 and not added_label else None
                )
            if not added_label:
                added_label = True

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def extract_clips_to_organized_folders(
        self,
        analysis_results: Dict,
        base_output_dir: str,
        padding_seconds: float = 2.0
    ) -> Dict[str, List[str]]:
        """
        Extract clips to organized folders by movement type.

        Args:
            analysis_results: Results from analyze_video_segments
            base_output_dir: Base directory (e.g., "Desktop/extracted_clips")
            padding_seconds: Extra time to add before/after each clip

        Returns:
            Dictionary mapping movement types to lists of extracted file paths
        """
        clips = analysis_results["suggested_clips"]

        if not clips:
            print("⚠️ No clips found to extract")
            return {}

        # Create organized folder structure - PATHOLOGICAL MOVEMENTS ONLY
        organized_clips: Dict[str, List[str]] = defaultdict(list)
        folder_mapping = {
            "decorticate": "decorticate",
            "dystonia": "dystonia", 
            "chorea": "chorea",
            "myoclonus": "myoclonus",
            "decerebrate": "decerebrate",
            "fencer posture": "fencer posture",
            "ballistic": "ballistic",
            "tremor": "tremor",
            "versive head": "versive head",
        }

        video_path = analysis_results["video_path"]
        video_name = Path(video_path).stem

        print(f"\n📁 Extracting clips to organized folders in {base_output_dir}")

        for i, clip in enumerate(clips):
            movement_type = clip["movement_type"]
            folder_name = folder_mapping.get(movement_type, movement_type.replace(" ", "_")) or "unknown"

            # Create movement-specific folder
            movement_dir = os.path.join(base_output_dir, folder_name)
            os.makedirs(movement_dir, exist_ok=True)

            # Calculate clip boundaries with padding
            start_time = max(0.0, clip["start_time"] - padding_seconds)
            end_time = min(analysis_results["duration"], clip["end_time"] + padding_seconds)

            # Generate output filename
            confidence = clip["avg_confidence"]
            output_filename = (
                f"{video_name}_{movement_type.replace(' ', '_')}_auto_{int(start_time)}s_{int(end_time)}s_conf{confidence:.2f}.mkv"
            )
            output_path = os.path.join(movement_dir, output_filename)

            # Extract clip
            try:
                success = self._clip_mkv_with_moviepy(video_path, output_path, start_time, end_time)

                if success:
                    organized_clips[movement_type].append(output_path)
                    duration = end_time - start_time
                    print(f"✅ Extracted to {folder_name}/: {output_filename} ({duration:.1f}s, conf: {confidence:.2f})")
                else:
                    print(f"❌ Failed to extract {output_filename}")

            except Exception as e:
                print(f"❌ Error extracting {output_filename}: {e}")

        # Summary
        print(f"\n📊 Organized Extraction Summary:")
        for movement_type, files in organized_clips.items():
            folder_name = folder_mapping.get(movement_type, movement_type.replace(" ", "_"))
            print(f"  {folder_name}/: {len(files)} clips")

        return dict(organized_clips)

    def batch_process_videos(
        self,
        video_dir: str,
        output_base_dir: str,
        file_pattern: str = "*.mkv"
    ) -> Dict[str, Dict]:
        """
        Process multiple videos in a directory.

        Args:
            video_dir: Directory containing videos to process
            output_base_dir: Base directory for all outputs
            file_pattern: Pattern to match video files

        Returns:
            Dictionary with results for each processed video
        """
        video_files = glob.glob(os.path.join(video_dir, file_pattern))
        print(f"🎬 Found {len(video_files)} videos to process")

        all_results: Dict[str, Dict] = {}

        for i, video_path in enumerate(video_files, 1):
            print(f"\n{'=' * 60}")
            print(f"🔄 Processing video {i}/{len(video_files)}: {Path(video_path).name}")
            print(f"{'=' * 60}")

            try:
                # Analyze video
                analysis_results = self.analyze_video_segments(video_path)

                # Create video-specific output directory
                video_name = Path(video_path).stem
                video_output_dir = os.path.join(output_base_dir, video_name)

                # Extract clips to organized folders
                extracted_clips = self.extract_clips_to_organized_folders(analysis_results, video_output_dir)

                # Generate reports
                report_path = os.path.join(video_output_dir, f"{video_name}_analysis_report.txt")
                self.generate_analysis_report(analysis_results, report_path)

                viz_path = os.path.join(video_output_dir, f"{video_name}_analysis_plot.png")
                self.create_visualization(analysis_results, viz_path)

                # Store results
                all_results[video_path] = {
                    "analysis": analysis_results,
                    "extracted_clips": extracted_clips,
                    "report_path": report_path,
                    "visualization_path": viz_path,
                }

                print(f"✅ Completed processing: {Path(video_path).name}")

            except Exception as e:
                print(f"❌ Error processing {Path(video_path).name}: {e}")
                all_results[video_path] = {"error": str(e)}

        print(f"\n🎉 Batch processing complete! Processed {len(video_files)} videos")
        return all_results

    def create_excel_style_summary(self, batch_results: Dict[str, Dict], output_path: str) -> str:
        """
        Create an Excel-style summary of all detected clips (like your database sheets)

        Args:
            batch_results: Results from batch_process_videos
            output_path: Path for the Excel file

        Returns:
            Path to created Excel file
        """
        summary_data: List[Dict] = []

        for video_path, results in batch_results.items():
            if "error" in results:
                continue

            video_name = Path(video_path).stem
            analysis = results["analysis"]
            timestamps = np.array(analysis["frame_timestamps"])
            pose_qualities = np.array(analysis["pose_qualities"])

            for clip in analysis.get("suggested_clips", []):
                # mask frames inside [start, end]
                mask = (timestamps >= clip["start_time"]) & (timestamps <= clip["end_time"])
                if mask.any():
                    quality_score = float(np.mean(pose_qualities[mask]))
                else:
                    quality_score = 0.0

                summary_data.append({
                    "video_file": video_name,
                    "start_time": f"0 days {int(clip['start_time']//3600):02d}:{int((clip['start_time']%3600)//60):02d}:{int(clip['start_time']%60):02d}",
                    "end_time":   f"0 days {int(clip['end_time']//3600):02d}:{int((clip['end_time']%3600)//60):02d}:{int(clip['end_time']%60):02d}",
                    "movement_type": clip["movement_type"],
                    "confidence": f"{clip['avg_confidence']:.3f}",
                    "duration_seconds": f"{clip['end_time'] - clip['start_time']:.1f}",
                    "quality_score": f"{quality_score:.3f}",
                    "detection_method": "AI_Intelligent_Clipper",
                    "notes": f"Auto-detected with {clip['avg_confidence']:.1%} confidence",
                })

        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_excel(output_path, index=False, sheet_name="AI_Detected_Clips")
            print(f"📄 Created Excel summary: {output_path}")
            print(f"📊 Total clips detected: {len(summary_data)}")
            # Print summary by movement type
            movement_counts = df["movement_type"].value_counts()
            print("📋 Clips by movement type:")
            for movement, count in movement_counts.items():
                print(f"  {movement}: {count}")

        return output_path


def main():
    parser = argparse.ArgumentParser(description="Intelligent PIM Video Clipper")
    parser.add_argument("video_path", help="Path to video file to analyze")
    parser.add_argument("--model_path", default="models/pim_model_joint_bone.pth",
                        help="Path to trained PIM model")
    parser.add_argument("--output_dir", default="extracted_clips",
                        help="Directory to save extracted clips")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Minimum confidence threshold for detections")
    parser.add_argument("--min_duration", type=float, default=30.0,
                        help="Minimum clip duration in seconds (default 30.0)")
    parser.add_argument("--max_duration", type=float, default=40.0,
                        help="Maximum clip duration in seconds (default 40.0)")
    parser.add_argument("--padding", type=float, default=2.0,
                        help="Padding to add before/after each clip in seconds")
    parser.add_argument("--extract_clips", action="store_true",
                        help="Extract clips in addition to analysis")
    parser.add_argument("--report_only", action="store_true",
                        help="Generate report and visualization only, no extraction")

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"❌ Video file not found: {args.video_path}")
        return

    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return

    # Initialize clipper
    clipper = PIMVideoClipper(args.model_path, args.confidence_threshold)

    # Analyze video
    print(f"\n🔍 Starting analysis of: {args.video_path}")
    analysis_results = clipper.analyze_video_segments(
        args.video_path,
        min_segment_duration=args.min_duration,
        max_segment_duration=args.max_duration
    )

    # Generate report
    report_path = clipper.generate_analysis_report(analysis_results)
    print(f"📄 Analysis report saved to: {report_path}")

    # Create visualization
    plot_path = clipper.create_visualization(analysis_results)
    print(f"📊 Analysis plot saved to: {plot_path}")

    # Extract clips if requested
    if args.extract_clips and not args.report_only:
        extracted_files = clipper.extract_clips(analysis_results, args.output_dir, args.padding)
        if extracted_files:
            print(f"\n✅ Successfully extracted {len(extracted_files)} clips")
            print("📁 Extracted files:")
            for file_path in extracted_files:
                print(f"   - {Path(file_path).name}")
        else:
            print("\n⚠️ No clips were extracted")

    # Summary
    clips = analysis_results["suggested_clips"]
    movements = analysis_results["movement_counts"]

    print(f"\n📈 Analysis Summary:")
    print(f"   Duration analyzed: {analysis_results['duration']:.1f} seconds")
    print(f"   Movements detected: {len(movements)} types")
    print(f"   Clips suggested: {len(clips)}")
    print(f"   Total potential training time: {sum(c['duration'] for c in clips):.1f} seconds")


if __name__ == "__main__":
    main()
