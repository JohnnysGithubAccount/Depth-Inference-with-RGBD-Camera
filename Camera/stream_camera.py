import pyrealsense2 as rs
import numpy as np
import cv2

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
colorizer.set_option(rs.option.color_scheme, 0)  # 0 = Jet (same as SDK Viewer)

try:
    while True:
        # === Wait for frames from the camera ===
        frames = pipeline.wait_for_frames()

        # === Align the depth frame to the color frame ===
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # === Convert frames to numpy arrays ===
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # === Stack color and depth images side by side ===
        images = np.hstack((color_image, depth_colormap))

        # === Display the result ===
        cv2.imshow("Color (left) | Depth Colormap (right)", images)

        # === Press ESC to exit ===
        if cv2.waitKey(1) == 27:
            break
finally:
    # === Stop streaming ===
    pipeline.stop()
