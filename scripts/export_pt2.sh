# Export WAFTv2 (dinov3 backbone) to a torch.export .pt2 artifact (bf16).
# Uses the same config / checkpoint / resolution as export_onnx.sh.
# Activate the waftv2 conda env first: conda activate waftv2

python infer_pt2/export_pt2.py \
        --cfg config/a2/dinov3/chairs-things.json \
        --ckpt waftv2-ckpts/dinov3/zero-shot.pth \
        --height 480 --width 640 \
        --iters 5 \
        --output weights/waftv2/waftv2_bf16.pt2
