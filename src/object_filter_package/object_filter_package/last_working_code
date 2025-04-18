#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import argparse
import numpy as np

# List of class names (adjust as per your model's classes)
class_names = [
    'AllenKey', 'Axis2', 'Bearing', 'Bearing2', 'Bearing_box', 'Bearing_box_ax16',
    'Distance_tube', 'Drill', 'Em_01', 'Em_02', 'F20_20_B', 'F20_20_G', 'Housing',
    'BLue COntainer box', 'M20_100', 'M30', 'Motor2', 'R20', 'S40_40_B', 'S40_40_G', 'Screwdriver',
    'Spacer', 'Wrench', 'm20', 'container_box_red'
]

class YoloZedNode(Node):
    def __init__(self, onnx_model_path, imgsz=640):
        super().__init__('yolo_zed_node')
        self.get_logger().info("Initializing YOLO ZED node")
        self.subscription = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()
        self.imgsz = imgsz

        # Load the ONNX model using OpenCV's DNN module
        self.net = cv2.dnn.readNetFromONNX(onnx_model_path)
        self.get_logger().info(f"Loaded ONNX model from: {onnx_model_path}")

        # Uncomment to use CUDA acceleration if available
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def preprocess(self, image):
        """Resize and normalize the image for ONNX model input."""
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True, crop=False)
        return blob

    def postprocess(self, outputs, original_image, conf_threshold=0.3, nms_threshold=0.4):
        out = outputs[0]
        self.get_logger().info(f"Model output shape: {out.shape}")

        # Handle various output shapes
        if len(out.shape) == 3:
            if out.shape[0] == 1 and out.shape[1] == 29:
                predictions = out.transpose(0, 2, 1)[0]
            elif out.shape[0] == 1 and out.shape[2] == 29:
                predictions = out[0]
            else:
                self.get_logger().error(f"Unexpected 3D output shape: {out.shape}")
                return [], [], []
        elif len(out.shape) == 2:
            if out.shape[1] == 29:
                predictions = out
            elif out.shape[0] == 29 and out.shape[1] == 8400:
                predictions = out.T  # Transpose to (8400, 29)
            else:
                self.get_logger().error(f"Unhandled 2D output shape: {out.shape}")
                return [], [], []
        else:
            self.get_logger().error(f"Unexpected output shape: {out.shape}")
            return [], [], []

        # Apply sigmoid activation
        predictions[:, 4:] = 1 / (1 + np.exp(-predictions[:, 4:]))

        # Confidence and class ID calculation
        scores = predictions[:, 4] * np.max(predictions[:, 5:], axis=1)
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Filter by confidence
        mask = scores > conf_threshold
        predictions = predictions[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        boxes = []
        for pred in predictions:
            cx, cy, w, h = pred[:4]
            x = cx - w / 2
            y = cy - h / 2
            boxes.append([x, y, w, h])

        # Scale boxes back to original image size
        h_orig, w_orig = original_image.shape[:2]
        scale_x = w_orig / self.imgsz
        scale_y = h_orig / self.imgsz
        scaled_boxes = []
        for box in boxes:
            x, y, w_box, h_box = box
            scaled_boxes.append([
                int(x * scale_x), int(y * scale_y),
                int(w_box * scale_x), int(h_box * scale_y)
            ])

        # NMS to filter overlapping boxes
        indices = cv2.dnn.NMSBoxes(scaled_boxes, scores.tolist(), conf_threshold, nms_threshold)

        final_boxes, final_scores, final_class_ids = [], [], []
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(scaled_boxes[i])
                final_scores.append(scores[i])
                final_class_ids.append(class_ids[i])
        return final_boxes, final_scores, final_class_ids

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if cv_image is None or cv_image.size == 0:
                self.get_logger().error("Converted image is empty")
                return
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        if len(cv_image.shape) == 3 and cv_image.shape[2] == 4:
            cv_image = cv_image[:, :, :3]

        cv_image = np.ascontiguousarray(cv_image)  # <- this ensures OpenCV compatibility


        blob = self.preprocess(cv_image)
        self.get_logger().debug(f"Input blob shape: {blob.shape}")  # Optional debug
        self.net.setInput(blob)

        try:
            outputs = self.net.forward()
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        boxes, confidences, class_ids = self.postprocess(outputs, cv_image)

        for box, score, class_id in zip(boxes, confidences, class_ids):
            x, y, w, h = box
            label = f"{class_names[class_id]}: {score:.2f}" if class_id < len(class_names) else f"ID {class_id}: {score:.2f}"
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(cv_image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 ONNX Detection", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description="ROS2 Node for YOLO ONNX inference on ZED camera feed")
    parser.add_argument('--onnx_model', type=str, required=True, help="Path to the YOLO ONNX model file")
    parser.add_argument('--imgsz', type=int, default=640, help="Input image size (default: 640)")
    parsed_args, unknown = parser.parse_known_args()

    node = YoloZedNode(parsed_args.onnx_model, imgsz=parsed_args.imgsz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down YOLO node...")
    finally:
        node.destroy_node()
        if rclpy.ok():  # Prevent double shutdown
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
