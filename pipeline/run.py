"""
Entry point: video → BVH pipeline.

Usage:
    python pipeline/run.py <video_path> [output_bvh_path]

Example:
    python pipeline/run.py test_videos/walk.mp4 output/walk.bvh
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.extract_pose import extract_pose
from pipeline.smooth import smooth_landmarks
from pipeline.to_bvh import generate_bvh


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline/run.py <video_path> [output.bvh]")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: file not found: {video_path}")
        sys.exit(1)

    out_bvh = sys.argv[2] if len(sys.argv) > 2 else "output/animation.bvh"
    os.makedirs(os.path.dirname(out_bvh) or ".", exist_ok=True)

    print("=" * 50)
    print(f"Input:  {video_path}")
    print(f"Output: {out_bvh}")
    print("=" * 50)

    t0 = time.time()

    print("\n[1/3] Extracting pose landmarks...")
    frames, fps = extract_pose(video_path)

    print("\n[2/3] Smoothing landmarks...")
    smoothed = smooth_landmarks(frames, fps, cutoff_hz=6.0)

    print("\n[3/3] Writing BVH...")
    generate_bvh(smoothed, fps, out_bvh)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Open {out_bvh} in Blender, Maya, or any BVH viewer.")


if __name__ == "__main__":
    main()
