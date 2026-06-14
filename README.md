# MMDeploy (Modified Version for RTMDet Instance Segmentation)

This repository is based on **MMDeploy** and includes several modifications specifically for **RTMDet Instance Segmentation** deployment with **TensorRT** and **ONNX Runtime**.

## Changes in This Fork

### 1. RTMDet Instance Segmentation TensorRT / ONNX Performance Fix

A common issue when exporting **RTMDet Instance Segmentation** models to **TensorRT** or **ONNX** is that GPU inference performance does not improve significantly compared to the original PyTorch model, resulting in little or no FPS gain.

This fork includes modifications in the RTMDet instance segmentation deployment head to address this issue and improve GPU utilization and inference throughput after export.

### 2. FP16 Conversion Fix for ONNX Runtime

The script:

```bash
tools/convert2fp16.py
```

has been modified to improve FP16 conversion compatibility with ONNX Runtime.

The implementation skips operators from the section:

```python
# --- Common list of ops that often cause issues with FP16 conversion ---
```

which helps avoid conversion failures and runtime issues commonly encountered when converting RTMDet Instance Segmentation ONNX models to FP16.

### 3. ONNX Runtime Inference Example

An example ONNX Runtime inference script has been added:

```bash
onnx.py
```

The script demonstrates how to:

- Load an exported RTMDet Instance Segmentation ONNX model
- Run inference with ONNX Runtime
- Use the CUDA Execution Provider
- Measure inference performance
- Perform mask and bounding box inference

---

## Tested Use Case

These modifications were developed and tested for:

- RTMDet Instance Segmentation
- ONNX Runtime (CUDA)
- TensorRT
- FP16 deployment

Other architectures may also benefit from the changes, but they were primarily designed for RTMDet Instance Segmentation.

## About MMDeploy

MMDeploy is the OpenMMLab deployment framework for exporting and deploying models to multiple inference backends, including:

- ONNX Runtime
- TensorRT
- NCNN
- OpenVINO
- TorchScript
- PPLNN

## Documentation

Official documentation:

https://mmdeploy.readthedocs.io/en/latest/

Useful references:

- `docs/en/02-how-to-run/convert_model.md`
- `docs/en/02-how-to-run/write_config.md`
- `docs/en/02-how-to-run/profile_model.md`
- `docs/en/02-how-to-run/quantize_model.md`

## License

This project follows the original MMDeploy license.

Apache License 2.0