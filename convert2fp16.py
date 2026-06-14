import onnx
from onnx import TensorProto
from onnxconverter_common import float16

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = "mmdeploy_models/mmdet/onnx/1280/onnx_model_name.onnx"
FP16_MODEL_PATH = "mmdeploy_models/mmdet/onnx/1280/fp16_model_io_fp32.onnx"
FP16_IO_MODEL_PATH = "mmdeploy_models/mmdet/onnx/1280/fp16_model_io_fp16.onnx"

# --- Common list of ops that often cause issues with FP16 conversion ---
OP_BLOCK_LIST = [
    'Resize', 'NonMaxSuppression', 'TopK', 'Gather', 
    'RandomNormalLike', 'RandomNormal'
]

# --- Data types to leave untouched ---
INT_TYPES = (
    TensorProto.INT64, TensorProto.INT32, 
    TensorProto.INT8, TensorProto.UINT8,
    TensorProto.BOOL
)

# ==========================================================
# UTILITY: Patch Cast nodes to match their corresponding model output type
# ==========================================================
def patch_cast_nodes_to_match_outputs(model):
    """
    Finds all Cast nodes that feed directly into a model output.
    Updates the 'to' attribute of that Cast node to match the
    data type defined in model.graph.output.
    """
    output_types = {o.name: o.type.tensor_type.elem_type for o in model.graph.output}
    output_names = set(output_types.keys())
    
    print("...  patching Cast nodes feeding model outputs")
    patched_count = 0

    for node in model.graph.node:
        if node.op_type == "Cast":
            node_outputs_are_model_outputs = output_names.intersection(node.output)
            
            if node_outputs_are_model_outputs:
                output_name = list(node_outputs_are_model_outputs)[0]
                target_type = output_types[output_name]
                
                for attr in node.attribute:
                    if attr.name == "to":
                        if attr.i != target_type:
                            print(f"    - Patching Cast node '{node.name}' (feeds '{output_name}') from type {attr.i} to {target_type}")
                            attr.i = target_type
                            patched_count += 1

    if patched_count == 0:
        print("    - No Cast nodes needed patching.")

# ==========================================================
# LOAD ORIGINAL MODEL
# ==========================================================
print(f"🔹 Loading FP32 model: {MODEL_PATH}")
model = onnx.load(MODEL_PATH)
print(f"    - Original Inputs:  {[i.name + ':' + str(i.type.tensor_type.elem_type) for i in model.graph.input]}")
print(f"    - Original Outputs: {[o.name + ':' + str(o.type.tensor_type.elem_type) for o in model.graph.output]}")


# ==========================================================
# MODEL 1: CONVERT TO FP16 (internal ops only, keep FP32 I/O)
# ==========================================================
print("\n⚙️ Converting internal tensors to FP16 (keeping original FP32/INT I/O types)...")
model_fp16 = float16.convert_float_to_float16(
    model,
    keep_io_types=True,
    disable_shape_infer=False,
    op_block_list=OP_BLOCK_LIST
)

# --- Patch Inputs to FP32 ---
# Manually force all non-INT inputs to FP32
print("... re-verifying model inputs are FP32 (and INTs are untouched)")
for inp in model_fp16.graph.input:
    if inp.type.tensor_type.elem_type not in INT_TYPES:
        inp.type.tensor_type.elem_type = TensorProto.FLOAT

# --- Patch Outputs to FP32 ---
# Manually force all non-INT outputs to FP32
print("... re-verifying model outputs are FP32 (and INTs are untouched)")
for output in model_fp16.graph.output:
    if output.type.tensor_type.elem_type not in INT_TYPES:
        output.type.tensor_type.elem_type = TensorProto.FLOAT

# Patch any Cast nodes feeding these outputs to match
patch_cast_nodes_to_match_outputs(model_fp16)

#onnx.save(model_fp16, FP16_MODEL_PATH)
print(f"✅ Saved FP16-internal, FP32/INT-I/O model: {FP16_MODEL_PATH}")
print(f"    - Final Inputs:  {[i.name + ':' + str(i.type.tensor_type.elem_type) for i in model_fp16.graph.input]}")
print(f"    - Final Outputs: {[o.name + ':' + str(o.type.tensor_type.elem_type) for o in model_fp16.graph.output]}")


# ==========================================================
# MODEL 2: CONVERT TO FP16 (full model including I/O)
# ==========================================================
print("\n⚙️ Converting full model to FP16 (including I/O)...")
# Reload the original FP32 model to start fresh
model_fp32 = onnx.load(MODEL_PATH)

model_fp16_io = float16.convert_float_to_float16(
    model_fp32,
    keep_io_types=False,  # <-- Convert I/O to FP16
    disable_shape_infer=False,
    op_block_list=OP_BLOCK_LIST
)

# --- Patch Inputs to FP16 ---
# Manually force all non-INT inputs to FP16
print("... setting model inputs to FP16 (and INTs are untouched)")
for inp in model_fp16_io.graph.input:
    if inp.type.tensor_type.elem_type not in INT_TYPES:
        inp.type.tensor_type.elem_type = TensorProto.FLOAT16
    else:
        print(f"    - Preserving INT input: '{inp.name}' as {inp.type.tensor_type.elem_type}")

# --- Patch Outputs to FP16 ---
# Manually force all non-INT outputs to FP16
print("... setting model outputs to FP16 (and INTs are untouched)")
for output in model_fp16_io.graph.output:
    if output.type.tensor_type.elem_type not in INT_TYPES:
        output.type.tensor_type.elem_type = TensorProto.FLOAT16
    else:
        print(f"    - Preserving INT output: '{output.name}' as {output.type.tensor_type.elem_type}")

# Patch any Cast nodes feeding these outputs to match
patch_cast_nodes_to_match_outputs(model_fp16_io)

onnx.save(model_fp16_io, FP16_IO_MODEL_PATH)
print(f"✅ Saved full FP16 model: {FP16_IO_MODEL_PATH}")
print(f"    - Final Inputs:  {[i.name + ':' + str(i.type.tensor_type.elem_type) for i in model_fp16_io.graph.input]}")
print(f"    - Final Outputs: {[o.name + ':' + str(o.type.tensor_type.elem_type) for o in model_fp16_io.graph.output]}")

print("\n🎉 Conversion complete!")