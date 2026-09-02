import os
import cv2
import numpy as np

SEVERITY = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

class TFLiteRetinalPredictor:
    """
    Lightweight, offline Diabetic Retinopathy inference engine.
    Designed to run on low-cost hardware without PyTorch dependencies,
    fulfilling the Pitch Deck's Offline-First & Edge Deployment claims.
    """

    def __init__(self, model_path="dr_model_quantized.tflite"):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.ort_session = None

        if not os.path.exists(model_path):
            alt_paths = ["dr_model.tflite", "dr_model_quantized.onnx", "dr_model.onnx"]
            for alt in alt_paths:
                if os.path.exists(alt):
                    self.model_path = alt
                    break

        self._load_model()

    def _load_model(self):
        # 1. Try ai_edge_litert (official Google LiteRT for Python 3.13+)
        try:
            import ai_edge_litert.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.runtime_type = "LiteRT (ai-edge-litert)"
            print(f"[TFLite Predictor] Loaded via {self.runtime_type}: {self.model_path}")
            return
        except Exception as e1:
            pass

        # 2. Try tflite_runtime
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.runtime_type = "TFLite Runtime"
            print(f"[TFLite Predictor] Loaded via {self.runtime_type}: {self.model_path}")
            return
        except Exception as e2:
            pass

        # 3. Try standard tensorflow.lite
        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.runtime_type = "TensorFlow Lite"
            print(f"[TFLite Predictor] Loaded via {self.runtime_type}: {self.model_path}")
            return
        except Exception as e3:
            pass

        # 4. Fallback to ONNX Runtime Edge engine
        try:
            import onnxruntime as ort
            onnx_path = "dr_model_quantized.onnx" if os.path.exists("dr_model_quantized.onnx") else "dr_model.onnx"
            self.ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.runtime_type = "ONNX Runtime Edge"
            print(f"[TFLite Predictor] Fallback loaded via {self.runtime_type}: {onnx_path}")
            return
        except Exception as e4:
            raise RuntimeError(
                f"[Error] Failed to initialize offline edge inference runtime: {e4}"
            )

    def preprocess(self, image_rgb: np.ndarray, target_size=(224, 224)) -> np.ndarray:
        """
        Pure NumPy / OpenCV preprocessing:
        Resize, convert to float32 [0, 1], normalize with ImageNet mean/std, (1, 3, H, W).
        """
        resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        img_float = resized.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (img_float - mean) / std

        # Transpose from (H, W, C) to (C, H, W)
        transposed = np.transpose(normalized, (2, 0, 1))
        # Add batch dimension -> (1, C, H, W)
        batch = np.expand_dims(transposed, axis=0).astype(np.float32)
        return batch

    def softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def predict(self, image_rgb: np.ndarray, return_probs: bool = False):
        """
        Runs offline edge inference on input RGB image.
        """
        input_data = self.preprocess(image_rgb)

        if self.interpreter is not None:
            # Handle NCHW vs NHWC input shapes in TFLite
            expected_shape = self.input_details[0]["shape"]
            expected_dtype = self.input_details[0]["dtype"]

            if len(expected_shape) == 4 and expected_shape[-1] == 3:
                # NHWC format
                input_tensor = np.transpose(input_data, (0, 2, 3, 1))
            else:
                input_tensor = input_data

            # Cast input to expected model tensor type (e.g. float16 / float32 / uint8)
            input_tensor = input_tensor.astype(expected_dtype)

            self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        else:
            # ONNX Runtime Edge fallback
            input_name = self.ort_session.get_inputs()[0].name
            output_name = self.ort_session.get_outputs()[0].name
            output_data = self.ort_session.run([output_name], {input_name: input_data})[0]

        logits = output_data.astype(np.float32).flatten()
        probs = self.softmax(logits)
        pred_idx = int(np.argmax(probs))
        severity_label = SEVERITY[pred_idx]
        confidence = float(probs[pred_idx])

        if return_probs:
            prob_dict = {SEVERITY[i]: float(probs[i]) for i in range(len(SEVERITY))}
            return severity_label, confidence, prob_dict

        return severity_label, confidence


def predict_stage_tflite(image_rgb: np.ndarray, model_path="dr_model_quantized.tflite", return_probs: bool = False):
    """Convenience wrapper for offline edge prediction."""
    predictor = TFLiteRetinalPredictor(model_path=model_path)
    return predictor.predict(image_rgb, return_probs=return_probs)
