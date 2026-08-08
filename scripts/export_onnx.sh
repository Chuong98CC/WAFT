# python export_onnx.py \
#         --cfg config/a2/dav2/chairs-things.json \
#         --ckpt waftv2-ckpts/dav2/zero-shot.pth \
#         --height 448 --width 672 \
#         --iters 5 \
#         --output weights/waftv2_dav2_i5_448x672.onnx

python export_onnx.py \
        --cfg config/a2/dinov3/chairs-things.json \
        --ckpt waftv2-ckpts/dinov3/zero-shot.pth \
        --height 480 --width 640 \
        --iters 5 \
        --output weights/waftv2_dinov3_i5_640x480.onnx