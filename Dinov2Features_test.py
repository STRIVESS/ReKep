import time
from datetime import datetime
import os
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt

# Define at top of your script
COLOR_PALETTE = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (0, 255, 128),  # Spring Green
    (128, 128, 128) # Gray
]

# Initialize RealSense camera
def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
    pipeline.start(config)
    return pipeline


# Get aligned RGB and Depth frames
def get_frames(pipeline):
    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return None, None

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    return color_image, depth_image


class Dinov2Features:
    # Initialize DINOv2 model
    def __init__(self, save_dir="feature_visualizations", log_dir="feature_logs", save_every=5):
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.save_every = save_every
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        # Single log file for all outputs
        self.log_file_path = os.path.join(self.log_dir, "feature_analysis.log")

        # Initialize log file with header
        with open(self.log_file_path, 'w') as f:
            f.write(f"DINOv2 Feature Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

        try:
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            print("Loaded DINOv2 with registers")
        except Exception as e:
            print(f"Failed to load with registers: {e}")
            # Fall back to version without registers
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            print("Loaded DINOv2 without registers")

        self.dinov2 = self.dinov2.eval()
        if torch.cuda.is_available():
            self.dinov2 = self.dinov2.cuda()

    def _log_message(self, message, print_to_console=True):
        """Helper method to log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        with open(self.log_file_path, 'a') as f:
            f.write(log_entry)

        if print_to_console:
            print(log_entry, end='')

    def _log_stats(self, stats, frame_count, log_type="raw"):
        """Log statistics to the single log file"""
        # Header
        self._log_message(f"Frame {frame_count} - {log_type.capitalize()} Features:", False)

        # Prepare stats text
        stats_text = []
        for k, v in stats.items():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                v = v.tolist() if hasattr(v, 'tolist') else str(v)
            stats_text.append(f"{k}: {v}")

        # Write all stats at once
        with open(self.log_file_path, 'a') as f:
            f.write("\n".join(stats_text) + "\n\n")

        # Print to console
        # print(f"\nFrame {frame_count} - {log_type.capitalize()} Features:")
        # print("\n".join(stats_text) + "\n")

    def _preprocess(self, image, visualize=False):
        """
        Preprocess image for DINOv2 feature extraction

        Args:
            image: Input RGB image (numpy array, HxWx3)
            visualize: Whether to show preprocessing results

        Returns:
            image_tensor: Preprocessed tensor (1,3,H,W)
            shape_info: Dictionary with original and processed shapes
            resized_image: The resized image (for visualization)
        """
        patch_size = 14
        h, w = image.shape[:2]

        # Resize to nearest multiple of patch size
        new_h = (h // patch_size) * patch_size
        new_w = (w // patch_size) * patch_size

        if new_h != h or new_w != w:
            resized_image = cv2.resize(image, (new_w, new_h))
            if visualize:
                cv2.imshow('Resized Input', resized_image)
        else:
            resized_image = image

        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).unsqueeze(0).float()
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        # ImageNet normalization
        image_tensor = image_tensor / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image_tensor.device)
        image_tensor = (image_tensor - mean) / std

        shape_info = {
            'orig_h': h,
            'orig_w': w,
            'new_h': new_h,
            'new_w': new_w,
            'patch_size': patch_size,
            'patch_h': new_h // patch_size,
            'patch_w': new_w // patch_size
        }

        return image_tensor, shape_info, resized_image


    def _print_feature_stats(self, patch_tokens, frame_count):
        """Optimized feature statistics printing"""
        stats = {
            "Type": str(type(patch_tokens)),
            "Shape": tuple(patch_tokens.shape),
            "Device": str(patch_tokens.device),
            "Data type": str(patch_tokens.dtype),
            "Sample values (first 10)": patch_tokens[0, 0, :10].cpu().numpy().tolist(),
            "Global mean": torch.mean(patch_tokens).item(),
            "Global std": torch.std(patch_tokens).item()
        }

        if frame_count % self.save_every == 0:
            self._log_stats(stats, frame_count, "raw")


    def _analyze_features(self, features, frame_count):
        """Optimized feature analysis"""

        features_np = features.cpu().numpy() if torch.is_tensor(features) else features
        stats = {
            "Final shape": features_np.shape,
            "Min (first 5 dims)": np.min(features_np, axis=(0, 1))[:5].tolist(),
            "Max (first 5 dims)": np.max(features_np, axis=(0, 1))[:5].tolist(),
            "Mean (first 5 dims)": np.mean(features_np, axis=(0, 1))[:5].tolist(),
            "Std (first 5 dims)": np.std(features_np, axis=(0, 1))[:5].tolist(),
            "Global min": np.min(features_np).item(),
            "Global max": np.max(features_np).item(),
            "Global mean": np.mean(features_np).item(),
            "Global std": np.std(features_np).item()
        }

        # Only save every 5 frames
        if frame_count % self.save_every == 0:
        # Vectorized statistics calculation
            # Log processed features
            self._log_stats(stats, frame_count, "processed")

            # Visualize feature distributions
            plt.figure(figsize=(12, 4))
            plt.subplot(121)
            plt.hist(features[:, :, 0].flatten(), bins=50)
            plt.title("First Feature Channel Distribution")

            plt.subplot(122)
            plt.imshow(features[:, :, 0], cmap='viridis')
            plt.title("First Feature Channel Spatial Map")
            plt.colorbar()
            # plt.show()

            # Save the figure
            save_path = os.path.join(self.save_dir, f"frame_{frame_count:04d}_features.jpg")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved feature visualizations to {save_path}")


    def _get_features(self, dinov2, frame_count, image_tensor, shape_info, visualize=False, print_features=False):
        """
        Optimized feature extraction and processing
        """
        with torch.no_grad():
            features = dinov2.forward_features(image_tensor)
            patch_tokens = features['x_norm_patchtokens']

            if print_features:
                self._print_feature_stats(patch_tokens, frame_count)

        # Vectorized feature processing
        feature_grid = patch_tokens.reshape(1, shape_info['patch_h'], shape_info['patch_w'], -1)
        feature_grid = feature_grid.permute(0, 3, 1, 2)

        feature_grid = F.interpolate(
            feature_grid,
            size=(shape_info['orig_h'], shape_info['orig_w']),
            mode='bilinear',
            align_corners=False
        )

        features = feature_grid.squeeze(0).permute(1, 2, 0).cpu().numpy()

        feature_vis = None
        if visualize:
            # First 3 channels
            vis1 = features[:, :, :3]  # Take first 3 feature channels
            vis1 = (vis1 - vis1.min()) / (vis1.max() - vis1.min())  # Normalize to [0,1]
            vis1 = (vis1 * 255).astype(np.uint8)  # Convert to 8-bit image

            # Mean activation
            vis2 = np.mean(features, axis=2)  # Average across all feature channels
            vis2 = (vis2 - vis2.min()) / (vis2.max() - vis2.min())  # Normalize
            vis2 = (vis2 * 255).astype(np.uint8)  # Convert to 8-bit

            feature_vis = {'first_3_channels': vis1, 'mean_activation': vis2}

        return features, feature_vis

    def extract_features(self, dinov2, image, frame_count, visualize=False, print_features=False):
        """Optimized feature extraction pipeline"""
        image_tensor, shape_info, _ = self._preprocess(image,visualize)
        features, feature_vis = self._get_features(dinov2, frame_count, image_tensor, shape_info, visualize, print_features)

        if print_features and frame_count % self.save_every == 0:
            self._analyze_features(features, frame_count)

        if visualize and feature_vis is not None:
            cv2.imshow('Feature Map (First 3 Channels)', feature_vis['first_3_channels'])
            cv2.imshow('Mean Feature Map', feature_vis['mean_activation'])

        return features

def main():
    """Optimized main processing loop"""
    # Initialize
    feature_extractor = Dinov2Features()
    pipeline = init_realsense()
    # sam_predictor = init_sam()

    # # Timing variables
    # frame_times = deque(maxlen=10)
    # last_log_time = time.time()
    frame_count = 0

    try:
        while True:
            # start_time = time.perf_counter()

            # Process frame
            color_image, depth_image = get_frames(pipeline)
            if color_image is not None:
                # Use the instance to call methods
                features = feature_extractor.extract_features(
                    feature_extractor.dinov2,  # Pass the model
                    color_image,
                    frame_count=frame_count,
                    visualize=True,
                    print_features=True
                )
                frame_count += 1
            # # FPS calculation
            # frame_times.append(time.perf_counter() - start_time)

            # # Periodic logging
            # if time.time() - last_log_time > 1.0:
            #     avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            #     print(f"Current FPS: {avg_fps:.1f}")
            #     last_log_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
