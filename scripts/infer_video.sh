VIDEO="data/astribot_stereo_lrb/videos/observation.images.cam_head_stereo_left/chunk-000/file_convertx264.mp4"
# python infer_video.py \
#         --input $VIDEO \
#         --cfg config/a2/dav2/chairs-things.json \
#         --ckpt waftv2-ckpts/dav2/zero-shot.pth \
#         --output-dir ./output \
#         --output-mode all \
#         --start 0 \
#         --stride 4 \
#         --max-frames 150
python infer_video_onnx.py \
        --input $VIDEO \
        --onnx weights/waftv2_dav2_i5_448x672.onnx \
        --output-dir ./output_onnx \
        --output-mode all \
        --start 0 \
        --stride 4 \
        --max-frames 150