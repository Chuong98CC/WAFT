"""Video encoding helpers for optical flow inference on video files."""

import cv2
import numpy as np


def get_video_info(video_path):
    """Probe a video file for metadata.

    Args:
        video_path (str): Path to the video file.

    Returns:
        tuple: (total_frames, fps, width, height). total_frames is 0 if the
               codec cannot report the frame count.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Verify the codec is actually decodable by reading a test frame.
    ret, _ = cap.read()
    cap.release()

    if not ret:
        raise IOError(
            f"Cannot decode frames from: {video_path}\n"
            "The video codec may not be supported by your OpenCV/ffmpeg build. "
            "Try re-encoding to H.264:\n"
            "  ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 output.mp4"
        )

    if fps <= 0:
        fps = 30.0  # sensible fallback

    return total_frames, fps, width, height


def encode_flow_video(flow_frames, output_path, fps):
    """Encode a list of flow visualization images into an MP4 video.

    Args:
        flow_frames (list of np.ndarray): Flow visualization images (BGR, uint8).
        output_path (str): Output .mp4 file path.
        fps (float): Frames per second for the output video.
    """
    if not flow_frames:
        print("No flow frames to encode.")
        return

    h, w = flow_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in flow_frames:
        writer.write(frame)

    writer.release()
    print(f"Flow video saved to: {output_path}")


def encode_overlay_video(original_frames, flow_frames, output_path, fps, alpha=0.5):
    """Encode a video with flow visualization overlaid onto original frames.

    Args:
        original_frames (list of np.ndarray): Original BGR frames (uint8).
        flow_frames (list of np.ndarray): Flow visualization images (BGR, uint8).
        output_path (str): Output .mp4 file path.
        fps (float): Frames per second for the output video.
        alpha (float): Blend weight for the flow overlay (0 = pure original,
                       1 = pure flow).
    """
    if not original_frames or not flow_frames:
        print("No frames to encode.")
        return

    assert len(original_frames) == len(flow_frames), \
        f"Frame count mismatch: {len(original_frames)} originals vs {len(flow_frames)} flow frames"

    h, w = original_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for orig, flow_vis in zip(original_frames, flow_frames):
        blended = cv2.addWeighted(orig, 1.0 - alpha, flow_vis, alpha, 0)
        writer.write(blended)

    writer.release()
    print(f"Overlay video saved to: {output_path}")
