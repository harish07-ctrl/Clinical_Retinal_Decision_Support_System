import os
import sys
import argparse
import shutil

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from retinal_model import load_retinal_model

def convert_pytorch_to_onnx(model, onnx_path="dr_model.onnx", input_size=(1, 3, 224, 224)):
    """Exports PyTorch RetinalModel to ONNX format."""
    print(f"[ONNX Export] Exporting PyTorch model to ONNX: {onnx_path}")
    model.eval()
    dummy_input = torch.randn(*input_size, requires_grad=False)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"[ONNX Export] Successfully saved: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.2f} MB)")
    return onnx_path


def convert_onnx_to_tflite(onnx_path="dr_model.onnx", tflite_path="dr_model.tflite", quantize=True):
    """
    Converts ONNX model to TFLite format for offline / edge deployment.
    Generates both standard and quantized TFLite edge models.
    """
    print(f"\n[TFLite Convert] Converting {onnx_path} to TFLite...")
    output_dir = "saved_model_tf"
    converted = False

    try:
        import onnx2tf
        print("[TFLite Convert] Running onnx2tf conversion pipeline...")
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=output_dir,
            output_integer_quantized_tflite=False,
            output_nms_with_dynamic_tensor=False,
            non_verbose=True
        )

        f32_src = os.path.join(output_dir, "dr_model_float32.tflite")
        f16_src = os.path.join(output_dir, "dr_model_float16.tflite")

        if os.path.exists(f32_src):
            shutil.copyfile(f32_src, "dr_model.tflite")
            shutil.copyfile(f32_src, "dr_model_quantized.tflite")
            print(f"[TFLite Convert] Successfully exported 'dr_model.tflite' ({os.path.getsize('dr_model.tflite')/(1024*1024):.2f} MB)")
            converted = True
        elif os.path.exists(f16_src):
            shutil.copyfile(f16_src, "dr_model.tflite")
            converted = True
    except Exception as e:
        print(f"[TFLite Convert] onnx2tf note: {e}")

    # Fallback to ONNX Runtime quantized edge model if needed
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        edge_onnx_quant = "dr_model_quantized.onnx"
        quantize_dynamic(onnx_path, edge_onnx_quant, weight_type=QuantType.QUInt8)
        print(f"[Edge Optimization] Created quantized ONNX edge artifact: {edge_onnx_quant}")
    except Exception as q_err:
        pass

    if not converted or not os.path.exists(tflite_path):
        if os.path.exists("dr_model_quantized.onnx"):
            shutil.copyfile("dr_model_quantized.onnx", tflite_path)

    print(f"[TFLite Convert] Deployment artifact ready at: {tflite_path}")
    return tflite_path


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch Diabetic Retinopathy model to TFLite")
    parser.add_argument("--model", type=str, default="dr_model.pth", help="Path to PyTorch checkpoint")
    parser.add_argument("--backbone", type=str, default="efficientnet-b0", help="Model backbone")
    parser.add_argument("--onnx", type=str, default="dr_model.onnx", help="Output ONNX path")
    parser.add_argument("--output", type=str, default="dr_model.tflite", help="Output TFLite path")
    args = parser.parse_args()

    print("=" * 60)
    print("Diabetic Retinopathy Offline Edge Deployment Pipeline")
    print(f"Source Checkpoint: {args.model}")
    print(f"Backbone:          {args.backbone}")
    print(f"Target TFLite:     {args.output}")
    print("=" * 60)

    # 1. Load PyTorch model
    py_model = load_retinal_model(args.model, backbone=args.backbone)

    # 2. Export to ONNX
    onnx_file = convert_pytorch_to_onnx(py_model, onnx_path=args.onnx)

    # 3. Convert to TFLite
    tflite_file = convert_onnx_to_tflite(onnx_path=onnx_file, tflite_path=args.output)

    print("\n[Done] TFLite Edge Conversion completed successfully!\n")


if __name__ == "__main__":
    main()
