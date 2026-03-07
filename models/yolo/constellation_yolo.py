
import numpy as np
import cv2
import onnxruntime as ort
import sys
import os

def preprocess_image(image_path, input_shape=(640, 640)):
    """Preprocesses the image for ONNX model input."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]
    img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
    img = img.transpose((2, 0, 1)) # HWC to CHW
    img = np.ascontiguousarray(img) # Ensure contiguous memory
    img = img.astype(np.float32) / 255.0 # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0) # Add batch dimension
    return img, img_height, img_width

def postprocess_results(outputs, original_img_shape, input_shape=(640, 640), conf_threshold=0.25, iou_threshold=0.45):
    """Post-processes ONNX model outputs to get bounding boxes, scores, and class IDs."""
    predictions = np.array(outputs[0])

    # Transpose and reshape to [boxes, x1y1x2y2, conf, classes]
    predictions = predictions.transpose(1, 0)

    # Filter out low confidence predictions
    scores = predictions[:, 4]
    valid_predictions = predictions[scores > conf_threshold]

    boxes = valid_predictions[:, :4]
    scores = valid_predictions[:, 4]
    class_ids = np.argmax(valid_predictions[:, 5:], axis=1)

    # Scale bounding boxes to original image size
    img_height, img_width = original_img_shape
    ratio_w, ratio_h = img_width / input_shape[0], img_height / input_shape[1]

    # Convert box format from xywh to x1y1x2y2 and scale
    boxes_xyxy = np.copy(boxes)
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * ratio_w # x1
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * ratio_h # y1
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * ratio_w # x2
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * ratio_h # y2

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        return boxes_xyxy[indices], scores[indices], class_ids[indices]
    return np.array([]), np.array([]), np.array([])

def draw_boxes(image, boxes, scores, class_ids, class_names):
    """Draws bounding boxes and labels on the image."""
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f'{class_names[class_id]} {score:.2f}'
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict_onnx.py <image_path>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_image_path = os.path.join(os.path.dirname(input_image_path), 'predicted_' + os.path.basename(input_image_path))

    # Define class names (must match your data.yaml)
    class_names = ['aquila', 'bootes', 'canis_major', 'canis_minor', 'cassiopeia', 'cygnus', 'gemini', 'leo', 'lyra', 'moon', 'orion', 'pleiades', 'sagittarius', 'scorpius', 'taurus', 'ursa_major']

    session = ort.InferenceSession("/content/constellation_stars_yolov8n.onnx", providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    try:
        preprocessed_img, original_h, original_w = preprocess_image(input_image_path)
        outputs = session.run([output_name], {input_name: preprocessed_img})
        
        boxes, scores, class_ids = postprocess_results(outputs, (original_h, original_w))
        
        original_image = cv2.imread(input_image_path)
        annotated_image = draw_boxes(original_image, boxes, scores, class_ids, class_names)
        
        cv2.imwrite(output_image_path, annotated_image)
        print(f"Annotated image saved to {output_image_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred during inference: {e}")
