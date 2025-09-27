#!/usr/bin/env python3
"""
Demo script for the new multi-view annotated video generation functionality.

This script demonstrates how to use the generate_annotated_videos_for_all_views method
to create separate annotated videos for each camera view in a multi-view PIM video.
"""

from annotated_video_generator import AnnotatedVideoGenerator
import os

def demo_multi_view_annotation():
    """
    Demonstrate the new multi-view annotation functionality.
    """

    # Initialize the generator
    generator = AnnotatedVideoGenerator(confidence_threshold=0.6)

    print("🎬 Multi-View PIM Video Annotation Demo")
    print("=" * 50)

    print("\n📋 New Method Available:")
    print("   generate_annotated_videos_for_all_views(input_video_path, output_dir=None, max_duration=None, start_time=0, num_views=None)")

    print("\n🔧 Method Parameters:")
    print("   - input_video_path: Path to your multi-view video file")
    print("   - output_dir: Directory to save individual view videos (auto-generated if None)")
    print("   - max_duration: Maximum seconds to process (None = full video)")
    print("   - start_time: Start processing from this time in seconds")
    print("   - num_views: Number of views to extract (None = auto-detect from video width)")

    print("\n🎯 What it does:")
    print("   - Auto-detects number of views from video width (supports 2, 3, or 4 views)")
    print("   - Splits video into individual camera views")
    print("   - Processes each view separately for accurate skeleton positioning")
    print("   - Generates annotated video for each view with PIM predictions")
    print("   - Creates separate output files with appropriate view names")

    print("\n💡 Usage Examples:")
    print("""
    # Auto-detect number of views
    output_dir = generator.generate_annotated_videos_for_all_views(
        input_video_path="your_video.mkv"
    )

    # Manually specify 3 views
    output_dir = generator.generate_annotated_videos_for_all_views(
        input_video_path="your_3view_video.mkv",
        num_views=3,
        max_duration=300
    )

    # Process 2-view video
    output_dir = generator.generate_annotated_videos_for_all_views(
        input_video_path="your_2view_video.mkv",
        num_views=2
    )
    """)

    print("\n✅ Benefits:")
    print("   - Robust handling of 2, 3, or 4 view configurations")
    print("   - Auto-detection prevents cropping errors")
    print("   - Fixes skeleton positioning issues in multi-view videos")
    print("   - Provides separate analysis for each camera perspective")
    print("   - Maintains spatial accuracy within each cropped view")

    # Check if method exists
    if hasattr(generator, 'generate_annotated_videos_for_all_views'):
        print("\n🎉 Method successfully implemented and ready to use!")
    else:
        print("\n❌ Method not found - check implementation")

if __name__ == "__main__":
    demo_multi_view_annotation()