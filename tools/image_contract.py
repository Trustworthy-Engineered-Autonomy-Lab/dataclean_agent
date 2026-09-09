"""Canonical IROS2026 image tensor contract shared by training/deployment."""

# The camera records 224x224 RGB frames.  IROS2026 removes the 80-pixel
# background band above the track before feeding a model, leaving 144x224.
RAW_IMAGE_HEIGHT = 224
IMAGE_CROP_TOP = 80
IMAGE_HEIGHT = RAW_IMAGE_HEIGHT - IMAGE_CROP_TOP
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
INPUT_CONTRACT_VERSION = "iros2026-rgb-crop80-144x224-v2"

TRAINING_INPUT_SHAPE = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)  # NCHW tail
DEPLOYMENT_INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)  # NHWC tail
IMAGE_SIZE_LABEL = f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}"
