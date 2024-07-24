POSE_CONFIG="/Users/derek/Desktop/ViTPose/ViTPose_base_coco_256x192.py"
POSE_CHECKPOINT="/Users/derek/Desktop/ViTPose/vitpose_base_coco_aic_mpii.pth"
IMG_PATH="/Users/derek/Desktop/Detection/detection/output_frame_001.png"
# IMG_PATH="/Users/derek/Documents/full_pic.png"

# Run the Python script
python3 demo/top_down_img_demo.py \
    "$POSE_CONFIG" \
    "$POSE_CHECKPOINT" \
    --img-path "$IMG_PATH" \
    --device mps \
    --show;

# python demo/bottom_up_img_demo.py \
#     configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py \
#     https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth \
#     --img-path "$IMG_PATH" \
#     --device mps \
#     --show;