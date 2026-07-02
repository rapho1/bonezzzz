"""
Extracts 3D pose landmarks from a video using MediaPipe Pose.
Output: list of frames, each frame is a list of 33 landmarks {x, y, z, visibility}.
Coordinates are normalized (0-1 for x/y, z is depth relative to hip).
"""
import json
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose


def extract_pose(
    video_path: str,
    output_json: str | None = None,
    model_complexity: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> list[list[dict]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {fps:.1f} fps, {total_frames} frames")

    all_frames: list[list[dict]] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,   # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_world_landmarks:
                landmarks = [
                    {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    }
                    for lm in results.pose_world_landmarks.landmark
                ]
            else:
                # Fill with None if pose not detected in this frame
                landmarks = None

            all_frames.append(landmarks)

            if frame_idx % 30 == 0:
                detected = "OK" if landmarks else "MISSING"
                print(f"  Frame {frame_idx}/{total_frames} [{detected}]")
            frame_idx += 1

    cap.release()
    print(f"Extracted {len(all_frames)} frames, "
          f"{sum(1 for f in all_frames if f)} with pose detected")

    if output_json:
        with open(output_json, "w") as f:
            json.dump({"fps": fps, "frames": all_frames}, f)
        print(f"Saved keypoints -> {output_json}")

    return all_frames, fps


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_pose.py <video_path> [output.json]")
        sys.exit(1)
    video = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "output/keypoints.json"
    extract_pose(video, out)
