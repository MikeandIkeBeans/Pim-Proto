#!/usr/bin/env python3
"""
Enhanced Batch Vallejo Processor
Processes complete vallejo videos and creates detailed sequence validation reports
"""

import os
import glob
import time
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from annotated_video_generator import AnnotatedVideoGenerator

class EnhancedVallejoProcessor:
    def __init__(self, model_path="models/pim_model_joint_bone.pth", 
                 confidence_threshold=0.6, sequence_length=30):
        """
        Initialize the enhanced processor
        
        Args:
            model_path: Path to trained model
            confidence_threshold: Minimum confidence for detection
            sequence_length: Number of frames for temporal analysis
        """
        self.generator = AnnotatedVideoGenerator(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            sequence_length=sequence_length
        )
        
        # Expected sequence pattern (your reference)
        self.expected_sequence = [
            'decorticate',
            'decerebrate', 
            'versive head', 'versive head',
            'fencer posture', 'fencer posture',
            'ballistic', 'ballistic', 'ballistic', 'ballistic',
            'chorea', 'chorea', 'chorea', 'chorea',
            'tremor', 'tremor', 'tremor', 'tremor',
            'dystonia', 'dystonia', 'dystonia', 'dystonia',
            'myoclonus', 'myoclonus', 'myoclonus', 'myoclonus',
            'decorticate',
            'decerebrate',
            'versive head', 'versive head',
            'fencer posture', 'fencer posture'
        ]
        
    def extract_sequence_from_detections(self, detections, time_window=10.0):
        """
        Extract comprehensive sequence analysis including ALL movements with counts
        
        Args:
            detections: List of detection dictionaries
            time_window: Time window in seconds to group similar detections
        
        Returns:
            Dict with simplified sequence and detailed movement counts
        """
        if not detections:
            return {'sequence': [], 'detailed_counts': [], 'all_movements': Counter()}
            
        sequence = []
        detailed_counts = []
        all_movements = Counter()
        current_time = 0
        
        while current_time < detections[-1]['timestamp']:
            # Get ALL detections in this time window (regardless of confidence)
            window_detections = [
                d for d in detections 
                if current_time <= d['timestamp'] < current_time + time_window
            ]
            
            if window_detections:
                # Count ALL movements in this window
                movements = [d['movement'] for d in window_detections]
                movement_counts = Counter(movements)
                all_movements.update(movements)
                
                # Find most common movement in this window
                most_common = movement_counts.most_common(1)[0][0]
                sequence.append(most_common)
                
                # Store detailed counts for this time window
                window_info = {
                    'time_start': current_time,
                    'time_end': current_time + time_window,
                    'dominant_movement': most_common,
                    'movement_counts': dict(movement_counts),
                    'total_detections': len(window_detections),
                    'high_confidence_count': len([d for d in window_detections if d.get('high_confidence', False)])
                }
                detailed_counts.append(window_info)
            
            current_time += time_window
        
        return {
            'sequence': sequence,
            'detailed_counts': detailed_counts,
            'all_movements': all_movements
        }
    
    def compare_with_expected_sequence(self, sequence_analysis, video_name):
        """
        Compare detected sequence with expected pattern
        
        Args:
            sequence_analysis: Dict from extract_sequence_from_detections
            video_name: Name of the video being analyzed
            
        Returns:
            Dictionary with comparison results
        """
        detected_sequence = sequence_analysis['sequence']
        
        comparison = {
            'video_name': video_name,
            'expected_length': len(self.expected_sequence),
            'detected_length': len(detected_sequence),
            'expected_sequence': self.expected_sequence,
            'detected_sequence': detected_sequence,
            'matches': [],
            'mismatches': [],
            'accuracy_score': 0.0,
            'pattern_analysis': {},
            'all_movements_detected': dict(sequence_analysis['all_movements']),
            'detailed_time_windows': sequence_analysis['detailed_counts'],
            'total_all_detections': sum(sequence_analysis['all_movements'].values())
        }
        
        # Direct sequence comparison (position by position)
        max_len = max(len(self.expected_sequence), len(detected_sequence))
        matches = 0
        
        for i in range(max_len):
            expected = self.expected_sequence[i] if i < len(self.expected_sequence) else 'END'
            detected = detected_sequence[i] if i < len(detected_sequence) else 'END'
            
            if expected == detected:
                matches += 1
                comparison['matches'].append({
                    'position': i,
                    'movement': expected,
                    'match': True
                })
            else:
                comparison['mismatches'].append({
                    'position': i,
                    'expected': expected,
                    'detected': detected,
                    'match': False
                })
        
        comparison['accuracy_score'] = matches / max_len if max_len > 0 else 0.0
        
        # Pattern analysis
        expected_counts = Counter(self.expected_sequence)
        detected_counts = Counter(detected_sequence)
        
        comparison['pattern_analysis'] = {
            'expected_movement_counts': dict(expected_counts),
            'detected_movement_counts': dict(detected_counts),
            'missing_movements': list(set(expected_counts.keys()) - set(detected_counts.keys())),
            'extra_movements': list(set(detected_counts.keys()) - set(expected_counts.keys())),
            'count_differences': {}
        }
        
        # Calculate count differences for each movement
        all_movements = set(expected_counts.keys()) | set(detected_counts.keys())
        for movement in all_movements:
            expected_count = expected_counts.get(movement, 0)
            detected_count = detected_counts.get(movement, 0)
            comparison['pattern_analysis']['count_differences'][movement] = {
                'expected': expected_count,
                'detected': detected_count,
                'difference': detected_count - expected_count
            }
        
        return comparison
    
    def create_detailed_sequence_report(self, detections, video_name, output_dir):
        """Create detailed sequence report for a video"""
        
        # Create individual video report directory
        video_report_dir = os.path.join(output_dir, f"{video_name}_analysis")
        os.makedirs(video_report_dir, exist_ok=True)
        
        # 1. Raw detections CSV (now includes ALL predictions)
        csv_file = os.path.join(video_report_dir, f"{video_name}_raw_detections.csv")
        with open(csv_file, 'w', newline='') as f:
            if detections:
                writer = csv.DictWriter(f, fieldnames=['frame', 'timestamp', 'movement', 'confidence', 'class_id', 'high_confidence'])
                writer.writeheader()
                writer.writerows(detections)
        
        # 2. Comprehensive sequence analysis with ALL movements
        sequence_analysis = self.extract_sequence_from_detections(detections, time_window=10.0)
        sequence_file = os.path.join(video_report_dir, f"{video_name}_sequence.txt")
        with open(sequence_file, 'w') as f:
            f.write("DETECTED MOVEMENT SEQUENCE (10-second windows)\n")
            f.write("=" * 50 + "\n")
            for i, movement in enumerate(sequence_analysis['sequence']):
                f.write(f"{i+1:2d}: {movement}\n")
            
            f.write(f"\nALL MOVEMENTS DETECTED (Total Count):\n")
            f.write("=" * 40 + "\n")
            for movement, count in sequence_analysis['all_movements'].most_common():
                percentage = (count / sum(sequence_analysis['all_movements'].values())) * 100
                f.write(f"{movement:15s}: {count:6d} ({percentage:.1f}%)\n")
            
            f.write(f"\nDETAILED TIME WINDOW ANALYSIS:\n")
            f.write("=" * 40 + "\n")
            for window in sequence_analysis['detailed_counts']:
                f.write(f"Time {window['time_start']:.0f}-{window['time_end']:.0f}s: {window['dominant_movement']} (dominant)\n")
                for mov, count in window['movement_counts'].items():
                    f.write(f"  - {mov}: {count} detections\n")
                f.write(f"  Total: {window['total_detections']} | High confidence: {window['high_confidence_count']}\n\n")
        
        # 3. Validation comparison
        comparison = self.compare_with_expected_sequence(sequence_analysis, video_name)
        validation_file = os.path.join(video_report_dir, f"{video_name}_validation.json")
        with open(validation_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # 4. Human-readable validation report with comprehensive movement analysis
        validation_report = os.path.join(video_report_dir, f"{video_name}_validation_report.txt")
        with open(validation_report, 'w') as f:
            f.write(f"COMPREHENSIVE VALIDATION REPORT: {video_name}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"ACCURACY SCORE: {comparison['accuracy_score']:.2%}\n")
            f.write(f"TOTAL DETECTIONS (ALL CONFIDENCE LEVELS): {comparison['total_all_detections']:,}\n\n")
            
            f.write("ALL MOVEMENTS DETECTED (Complete Count):\n")
            f.write("-" * 50 + "\n")
            for movement, count in comparison['all_movements_detected'].items():
                percentage = (count / comparison['total_all_detections']) * 100
                f.write(f"{movement:15s}: {count:6,d} detections ({percentage:.1f}%)\n")
            
            f.write(f"\nEXPECTED vs DETECTED SEQUENCE (10s windows):\n")
            f.write("-" * 40 + "\n")
            max_len = max(len(comparison['expected_sequence']), len(comparison['detected_sequence']))
            
            for i in range(max_len):
                expected = comparison['expected_sequence'][i] if i < len(comparison['expected_sequence']) else 'END'
                detected = comparison['detected_sequence'][i] if i < len(comparison['detected_sequence']) else 'END'
                match_symbol = "✓" if expected == detected else "✗"
                f.write(f"{i+1:2d}: {expected:15s} | {detected:15s} {match_symbol}\n")
            
            f.write(f"\nMOVEMENT COUNT ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            for movement, counts in comparison['pattern_analysis']['count_differences'].items():
                expected = counts['expected']
                detected = counts['detected']
                diff = counts['difference']
                status = "CORRECT" if diff == 0 else f"OFF BY {abs(diff)} ({'MORE' if diff > 0 else 'LESS'})"
                f.write(f"{movement:15s}: Expected {expected:2d}, Got {detected:2d} - {status}\n")
            
            if comparison['pattern_analysis']['missing_movements']:
                f.write(f"\nMISSING MOVEMENTS: {', '.join(comparison['pattern_analysis']['missing_movements'])}\n")
            
            if comparison['pattern_analysis']['extra_movements']:
                f.write(f"EXTRA MOVEMENTS: {', '.join(comparison['pattern_analysis']['extra_movements'])}\n")
        
        return comparison
    
    def process_all_vallejo_videos(self, vallejo_dir=r"C:\Users\Mike\Desktop\vallejo", 
                                  output_dir="vallejo_full_analysis"):
        """
        Process all vallejo videos completely and create comprehensive analysis
        """
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all video files
        video_extensions = ['*.mkv', '*.mp4', '*.avi', '*.mov']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(vallejo_dir, ext)))
        
        if not video_files:
            print(f"No video files found in {vallejo_dir}")
            return
        
        video_files.sort()  # Process in chronological order
        
        # Skip first video and start from second (index 1)
        video_files = video_files[1:]  # Skip first video
        print(f"Skipping first video, processing {len(video_files)} remaining vallejo videos (FULL LENGTH)")
        
        # Overall summary
        batch_summary = {
            'processing_date': datetime.now().isoformat(),
            'total_videos': len(video_files),
            'expected_sequence': self.expected_sequence,
            'confidence_threshold': self.generator.confidence_threshold,
            'video_results': []
        }
        
        successful = 0
        failed = 0
        all_comparisons = []
        
        for i, video_path in enumerate(video_files, 1):
            video_name = Path(video_path).stem
            annotated_filename = f"annotated_full_{video_name}.mp4"
            annotated_path = os.path.join(output_dir, annotated_filename)
            
            print(f"\n{'='*60}")
            print(f"Processing video {i}/{len(video_files)}: {video_name}")
            print(f"{'='*60}")
            
            try:
                start_time = time.time()
                
                # Process FULL video (no duration limit)
                print("🎬 Creating annotated video (full length)...")
                self.generator.generate_annotated_video(
                    input_video_path=video_path,
                    output_video_path=annotated_path,
                    max_duration=None,  # FULL VIDEO
                    start_time=0
                )
                
                # Process for detailed analysis (we need to extract detections)
                print("📊 Extracting detections for analysis...")
                detections = self._extract_detections_from_video(video_path)
                
                # Create detailed reports
                print("📝 Creating validation reports...")
                comparison = self.create_detailed_sequence_report(detections, video_name, output_dir)
                all_comparisons.append(comparison)
                
                process_time = time.time() - start_time
                
                video_result = {
                    'video_name': video_name,
                    'processing_time_minutes': round(process_time / 60, 1),
                    'total_detections': len(detections),
                    'total_all_detections': comparison['total_all_detections'],
                    'high_confidence_detections': len([d for d in detections if d.get('high_confidence', False)]),
                    'accuracy_score': comparison['accuracy_score'],
                    'detected_sequence_length': len(comparison['detected_sequence']),
                    'all_movements_detected': comparison['all_movements_detected'],
                    'success': True
                }
                
                batch_summary['video_results'].append(video_result)
                successful += 1
                
                print(f"✅ Success! Processed in {process_time/60:.1f} minutes")
                print(f"   Total detections (all confidence): {comparison['total_all_detections']:,}")
                print(f"   High confidence detections: {len([d for d in detections if d.get('high_confidence', False)])}")
                print(f"   Sequence accuracy: {comparison['accuracy_score']:.1%}")
                
                # Show top 3 detected movements
                top_movements = sorted(comparison['all_movements_detected'].items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"   Top movements: {', '.join([f'{mov}({count})' for mov, count in top_movements])}")
                
            except Exception as e:
                print(f"❌ Error processing {video_name}: {str(e)}")
                
                video_result = {
                    'video_name': video_name,
                    'processing_time_minutes': 0,
                    'total_detections': 0,
                    'accuracy_score': 0.0,
                    'detected_sequence_length': 0,
                    'success': False,
                    'error': str(e)
                }
                
                batch_summary['video_results'].append(video_result)
                failed += 1
        
        # Create overall batch analysis
        self._create_batch_analysis_report(batch_summary, all_comparisons, output_dir)
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📁 Output directory: {output_dir}")
        
        return batch_summary
    
    def _extract_detections_from_video(self, video_path):
        """Extract detections from video without creating annotated version"""
        import cv2
        import torch
        import numpy as np
        from collections import deque
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_buffer = deque(maxlen=self.generator.sequence_length)
        detections = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                timestamp = frame_count / fps if fps > 0 else frame_count
                
                # Process frame
                landmarks = self.generator.processor.extract_pose_landmarks_from_frame(frame)
                
                if landmarks is not None:
                    frame_buffer.append(landmarks)
                    
                    if len(frame_buffer) == self.generator.sequence_length:
                        sequence_tensor = torch.FloatTensor(list(frame_buffer)).unsqueeze(0)
                        
                        with torch.no_grad():
                            output = self.generator.model(sequence_tensor)
                            
                            if isinstance(output, tuple):
                                output = output[0]
                                
                            probabilities = torch.softmax(output, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)
                            
                            pred_class = int(predicted.item())
                            conf_value = confidence.item()
                            
                            # Record ALL predictions (not just high confidence ones)
                            movement_name = self.generator.movement_names.get(pred_class, f"unknown_{pred_class}")
                            
                            detections.append({
                                'frame': frame_count,
                                'timestamp': timestamp,
                                'movement': movement_name,
                                'confidence': conf_value,
                                'class_id': pred_class,
                                'high_confidence': conf_value >= self.generator.confidence_threshold
                            })
                
                # Progress update every 30 seconds
                if frame_count % (int(fps) * 30) == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        finally:
            cap.release()
        
        return detections
    
    def _create_batch_analysis_report(self, batch_summary, all_comparisons, output_dir):
        """Create overall batch analysis report"""
        
        # Save batch summary JSON
        summary_file = os.path.join(output_dir, 'batch_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        # Create batch analysis report
        report_file = os.path.join(output_dir, 'BATCH_ANALYSIS_REPORT.txt')
        with open(report_file, 'w') as f:
            f.write("VALLEJO VIDEO BATCH ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Processing Date: {batch_summary['processing_date']}\n")
            f.write(f"Total Videos Processed: {batch_summary['total_videos']}\n")
            f.write(f"Confidence Threshold: {batch_summary['confidence_threshold']}\n\n")
            
            # Overall accuracy statistics
            successful_comparisons = [c for c in all_comparisons if c]
            if successful_comparisons:
                avg_accuracy = sum(c['accuracy_score'] for c in successful_comparisons) / len(successful_comparisons)
                f.write(f"OVERALL SEQUENCE ACCURACY: {avg_accuracy:.2%}\n\n")
            
            # Individual video results
            f.write("INDIVIDUAL VIDEO RESULTS:\n")
            f.write("-" * 40 + "\n")
            for result in batch_summary['video_results']:
                if result['success']:
                    f.write(f"{result['video_name']}: {result['accuracy_score']:.1%} accuracy, "
                           f"{result['total_detections']} detections, "
                           f"{result['processing_time_minutes']}min\n")
                else:
                    f.write(f"{result['video_name']}: FAILED\n")
            
            # Movement pattern analysis across all videos
            if successful_comparisons:
                f.write(f"\nMOVEMENT PATTERN ANALYSIS (across all videos):\n")
                f.write("-" * 40 + "\n")
                
                # Aggregate movement counts
                total_expected = Counter()
                total_detected = Counter()
                
                for comparison in successful_comparisons:
                    for movement, counts in comparison['pattern_analysis']['count_differences'].items():
                        total_expected[movement] += counts['expected']
                        total_detected[movement] += counts['detected']
                
                f.write("Movement\t\tExpected\tDetected\tAccuracy\n")
                f.write("-" * 50 + "\n")
                for movement in sorted(total_expected.keys()):
                    expected = total_expected[movement]
                    detected = total_detected[movement]
                    accuracy = detected / expected if expected > 0 else 0
                    f.write(f"{movement:15s}\t{expected:8d}\t{detected:8d}\t{accuracy:.1%}\n")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process all vallejo videos with full analysis')
    parser.add_argument('--vallejo_dir', '-d', 
                        default=r"C:\Users\Mike\Desktop\vallejo",
                        help='Directory containing vallejo videos')
    parser.add_argument('--output_dir', '-o', 
                        default='vallejo_full_analysis',
                        help='Output directory for analysis')
    parser.add_argument('--model', '-m', 
                        default='models/pim_model_joint_bone.pth',
                        help='Path to model file')
    parser.add_argument('--confidence', '-c', type=float, default=0.6,
                        help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    processor = EnhancedVallejoProcessor(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    # Process all vallejo videos with full analysis
    results = processor.process_all_vallejo_videos(
        vallejo_dir=args.vallejo_dir,
        output_dir=args.output_dir
    )
    
    print(f"\n🎬 Complete analysis saved in: {args.output_dir}")
    print(f"📄 Check BATCH_ANALYSIS_REPORT.txt for overall results")

if __name__ == "__main__":
    # Default processing if run without arguments
    import sys
    if len(sys.argv) == 1:
        processor = EnhancedVallejoProcessor(confidence_threshold=0.6)
        processor.process_all_vallejo_videos()
    else:
        main()