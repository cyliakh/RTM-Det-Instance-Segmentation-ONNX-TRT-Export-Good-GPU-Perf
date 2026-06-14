import onnxruntime as ort
import numpy as np
import cv2
import time
import os

# =========================
# 1️⃣ Load ONNX model
# =========================
onnx_model = ""
providers = ['TensorrtExecutionProvider']  # or ['CPUExecutionProvider']

session = ort.InferenceSession(onnx_model, providers=providers)
print("Execution providers used:", session.get_providers())

# =========================
# 2️⃣ Open video
# =========================
video_path = ""
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"❌ Failed to open video: {video_path}")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


model_input_w = 1280
model_input_h = 1280

# Calculate scaling factors
scale_x = width / model_input_w
scale_y = height / model_input_h


#os.makedirs("outputs", exist_ok=True)
#out_path = "outputs/output_video.mp4"
#fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

print(f"🎥 Processing video: {video_path}")
print(f"Frames: {frame_count}, Resolution: {width}x{height}, FPS: {fps}")

# =========================
# 3️⃣ Inference loop
# =========================
input_name = session.get_inputs()[0].name
frame_idx = 0
total_infer_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Preprocess
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1280, 1280))
    img_norm = img_resized.astype(np.float32) / 255.0   #replace by letterbox
    img_input = np.transpose(img_norm, (2, 0, 1))[None, :, :, :]  
    #img_input = img_input.astype(np.float16)  # uncomment this line if you need fp16 input for fp16 io models
    
    # Run inference
    start = time.time()
    dets, labels, masks = session.run(None, {input_name: img_input})
    infer_time = time.time() - start
    dets, labels, masks = dets.astype(np.float32), labels.astype(np.float32), masks.astype(np.float32)

    print("Inference time for frame {}: {:.4f}s".format(frame_idx, infer_time))
    print("masks shape:", masks.shape)

    # Process outputs
    dets = np.squeeze(dets, axis=0)
    labels = np.squeeze(labels, axis=0)
    masks = np.squeeze(masks, axis=0)

    # Visualize
    vis = frame.copy()
    for i in range(dets.shape[0]):
        if dets.shape[1] >= 5:
            x1, y1, x2, y2, score = dets[i]
        else:
            x1, y1, x2, y2 = dets[i]
            score = 1.0

        if score < 0.2:
            continue

        # --- ADD THIS: Scale coordinates ---
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)
        # --- END ADD ---

        # --- MODIFY THIS: Use scaled coordinates ---
        cv2.rectangle(vis, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)
        
        label_id = int(labels[i]) if labels.ndim == 1 else int(np.argmax(labels[i]))
        text = f"ID:{label_id} {score:.2f}"
        
        # --- MODIFY THIS: Use scaled coordinates ---
        cv2.putText(vis, text, (x1_scaled, y1_scaled - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 1)

        if masks.ndim == 3 and i < masks.shape[0]:
            mask = masks[i].astype(np.float32)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_CUBIC)  # smoother
            #mask = cv2.GaussianBlur(mask, (3, 3), 0)  # optional smoothing
            mask = (mask > 0.5).astype(np.uint8)

            colored_mask = np.zeros_like(vis)
            colored_mask[:, :, 1] = mask * 255
            vis = cv2.addWeighted(vis, 1.0, colored_mask, 0.4, 0)
            cv2.imwrite(f"outputvid.png", vis)

    # Write to video
    #writer.write(vis)


cap.release()
#writer.release()
print(f"✅ Saved processed video to: {out_path}")

