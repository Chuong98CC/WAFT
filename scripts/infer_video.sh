VIDEO="data/astribot_stereo_lrb/videos/observation.images.cam_head_stereo_left/chunk-000/file_convertx264.mp4"

# ── PyTorch checkpoint inference ─────────────────────────────────────────────
# python infer_video.py \
#         --input $VIDEO \
#         --cfg config/a2/dav2/chairs-things.json \
#         --ckpt waftv2-ckpts/dav2/zero-shot.pth \
#         --output-dir ./output \
#         --output-mode all \
#         --start 0 \
#         --stride 4 \
#         --max-frames 150

# ── ONNX inference ───────────────────────────────────────────────────────────
# python infer_video_onnx.py \
#         --input $VIDEO \
#         --onnx weights/waftv2_dinov3_i5_480x640.onnx \
#         --output-dir ./outputs/dino3_onnx \
#         --output-mode all \
#         --start 210 \
#         --stride 4 \
#         --max-frames 30

# ── TensorRT inference ───────────────────────────────────────────────────────
python infer_video_onnx.py \
        --input $VIDEO \
        --trt weights/waftv2/waftv2_dinov3_i5_640x480.engine \
        --output-dir ./outputs/dino3_trt \
        --output-mode flow \
        --start 210 \
        --stride 4 \
        --max-frames 30
