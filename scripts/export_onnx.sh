python export_onnx.py \
        --cfg config/a2/dav2/chairs-things.json \
        --ckpt waftv2-ckpts/dav2/zero-shot.pth \
        --height 448 --width 672 \
        --iters 5 \
        --output weights/waftv2_dav2_i5_448x672.onnx