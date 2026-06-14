# MMDeploy (Modified Version)

This repository is based on **MMDeploy** and includes several modifications focused on improving deployment performance and ONNX Runtime compatibility.

## Changes in This Fork

### 1. TensorRT / ONNX GPU Performance Fix

A common issue when exporting models to **TensorRT** or **ONNX** is that GPU inference performance does not improve significantly compared to the original PyTorch model, resulting in little or no FPS gain.

This fork includes modifications in the deployment head to address this issue and improve GPU utilization and inference throughput after export.

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

which helps avoid conversion failures and runtime issues commonly encountered when converting ONNX models to FP16.

### 3. ONNX Runtime Inference Example

An example ONNX Runtime inference script has been added:

```bash
onnx.py
```

The script demonstrates how to:

- Load an exported ONNX model
- Run inference with ONNX Runtime
- Use CUDA Execution Provider
- Measure inference performance

---

## About MMDeploy

MMDeploy is the OpenMMLab deployment framework for exporting and deploying models to multiple inference backends, including:

- ONNX Runtime
- TensorRT
- NCNN
- OpenVINO
- TorchScript
- PPLNN

### Supported OpenMMLab Projects

- MMDetection
- MMYOLO
- MMSegmentation
- MMPose
- MMOCR
- MMPretrain
- MMRotate
- MMAction2
- MMDetection3D

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