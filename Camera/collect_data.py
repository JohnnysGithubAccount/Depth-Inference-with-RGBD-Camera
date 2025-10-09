import pyrealsense2 as rs
import numpy as np
import cv2
import os


# === Initialize pipeline ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# === Start streaming ===
pipeline.start(config)

# === Create align object (align depth to color stream) ===
align = rs.align(rs.stream.color)

# === Create colorizer for depth visualization ===
colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 0)  # 0 = Jet

# === Create folders ===
scene = "scene_1"
rgb_folder = f"../self_collected_dataset/rgb/{scene}"
depth_folder = f"../self_collected_dataset/depth/{scene}"
os.makedirs(rgb_folder, exist_ok=True)
os.makedirs(depth_folder, exist_ok=True)

# === Font for letters ===
font = cv2.FONT_HERSHEY_SIMPLEX

def put_label(image, text):
    return cv2.putText(image.copy(), text, (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

scene_id = 1

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()

        # Align depth to color
        aligned_frames = align.process(frames)
        depth_frame = frames.get_depth_frame()           # original depth
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned depth
        color_frame = aligned_frames.get_color_frame()   # aligned color

        if not depth_frame or not aligned_depth_frame or not color_frame:
            continue

        # Convert to numpy
        color_image = np.asanyarray(color_frame.get_data())
        aligned_depth = np.asanyarray(aligned_depth_frame.get_data())
        original_depth = np.asanyarray(depth_frame.get_data())

        # Colorize for display
        aligned_depth_colored = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        original_depth_colored = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Add labels
        rgb_labeled = put_label(color_image, "Aligned RGB")
        aligned_depth_labeled = put_label(aligned_depth_colored, "Aligned Depth")
        original_depth_labeled = put_label(original_depth_colored, "Original Depth")

        # Stack horizontally
        display = np.hstack((aligned_depth_labeled, rgb_labeled, original_depth_labeled))

        cv2.imshow("RealSense Streams", display)
        key = cv2.waitKey(1) & 0xFF

        # Save on SPACE
        if key == 32:  # Space key
            rgb_path = f"image_{scene_id:04d}.png"
            rgb_path = os.path.join(rgb_folder, rgb_path)
            depth_path = f"image_{scene_id:04d}.png"
            depth_path = os.path.join(depth_folder, depth_path)

            cv2.imwrite(rgb_path, color_image)
            cv2.imwrite(depth_path, aligned_depth)  # 16-bit depth
            print(f"Saved scene {scene_id} → {rgb_path}, {depth_path}")
            scene_id += 1

        # Exit on ESC
        elif key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
