    POSE_CONFIG="/Users/derek/Desktop/ViTPose/ViTPose_base_coco_256x192.py"
    POSE_CHECKPOINT="/Users/derek/Desktop/ViTPose/vitpose_base_coco_aic_mpii.pth"
    IMG_ROOT="/Users/derek/Desktop/detection/test_images"

    python run_inference.py \
        --pose_config "$POSE_CONFIG" \
        --pose_checkpoint "$POSE_CHECKPOINT" \
        --img_root "$IMG_ROOT" \