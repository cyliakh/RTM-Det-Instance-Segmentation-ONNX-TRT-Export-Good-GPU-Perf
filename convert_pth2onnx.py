from mmdeploy.apis import torch2onnx
#from mmdeploy.backend.tensorrt.onnx2tensorrt import onnx2tensorrt
from mmdeploy.backend.sdk.export_info import export2SDK

img = '../mmdetection/demo/demo.jpg' #or any image
work_dir = 'mmdeploy_models/mmdet/onnx/1280'
onnx_file = 'onnx_model_name.onnx'  # ONNX first
deploy_cfg = 'configs/mmdet/instance-seg/instance-seg_rtmdet-ins_onnxruntime_static-640x640.py' #cofigs inside mmdeploy

model_cfg = 'rtmdet-ins_s_8xb32-300e_coco.py' #insert your path of the model config 
model_checkpoint = '/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth' #your pth model you want to convert
device = 'cuda:0'

#rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth
#rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth
#rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth
#rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth
#rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth

# 1️⃣ PyTorch -> ONNX
torch2onnx(img, work_dir, onnx_file, deploy_cfg, model_cfg,
           model_checkpoint, device)

# 2️⃣ ONNX -> TensorRT
#onnx2tensorrt(work_dir + '/' + onnx_file, work_dir, deploy_cfg, engine_file, device=device)

# 3️⃣ Optional: export SDK info for inference
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint,
           device=device)
