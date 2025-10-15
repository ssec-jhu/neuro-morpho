from neuro_morpho.model.unet import UNet
from neuro_morpho.util import get_device
import cv2
import numpy as np
import torch
from pathlib import Path

def main() -> None:
    
    model = UNet()
    device = get_device()
    model_save_dir = "/Users/vkluzner/OneDrive/NeuralMorphology//SimulationsNew_Tif3334/neuro-morpho/models"  if device == "mps" \
        else "/home/idies/workspace/ssec_neural_morphology/SimulationsNew_Tif3334/neuro-morpho/models" if device == "cuda" or device == "cpu" \
        else None
    if model_save_dir is None:
        raise ValueError("Unsupported device")
    checkpoint_dir = Path(model_save_dir) / "304f9c01319a4899a80f4514c413213d" / "checkpoints"
    model.load_checkpoint(checkpoint_dir)
    model.model.eval()

    print("Device:", model.device)
    print("TF32 matmul:", torch.backends.cuda.matmul.allow_tf32)
    print("TF32 cuDNN:", torch.backends.cudnn.allow_tf32)
    print("Model dtype:", next(model.model.parameters()).dtype)
    print("Eval mode:", not model.model.training)

    image_name = "Realistic-SBR-1-Sample-3-time-100.00.tif"
    image_path = Path(model_save_dir.replace("models", "val/imgs")) / image_name
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)  # (height, width) or (height, width, channels)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    tile = image[1000:1512, 1000:1512]  # Example tile
    np.save("tile.npy", tile)
    cv2.imwrite("tile.png", tile)
    tile = cv2.convertScaleAbs(tile, alpha=255.0 / tile.max()) / 255.0
    
    # # Start the inferring process
    # tile_flip_0 = cv2.flip(tile, 0)  # Vertical flip
    # tile_flip_1 = cv2.flip(tile, 1)  # Horizontal flip
    # tile_flip__1 = cv2.flip(tile, -1)  # Both axes
    # tile_stack = np.stack([tile, tile_flip_0, tile_flip_1, tile_flip__1])
    # tile_torch = torch.tensor(tile_stack).unsqueeze(1).to(torch.float32).to(model.device)
    # with torch.no_grad():
    #     pred, _, _, _ = model.model(tile_torch)
    #     pred = torch.sigmoid(pred)
    #     pred_ori, pred_flip_0, pred_flip_1, pred_flip__1 = pred
    # pred_ori = pred_ori.cpu().numpy().squeeze()
    # pred_flip_0 = cv2.flip(pred_flip_0.cpu().numpy().squeeze(), 0)
    # pred_flip_1 = cv2.flip(pred_flip_1.cpu().numpy().squeeze(), 1)
    # pred_flip__1 = cv2.flip(pred_flip__1.cpu().numpy().squeeze(), -1)
    # tile_pred = np.mean([pred_ori, pred_flip_0, pred_flip_1, pred_flip__1], axis=0)

     # Start the inferring process
    tile_stack = np.stack(tile)[np.newaxis, np.newaxis, :, :]
    tile_torch = torch.tensor(tile_stack).to(torch.float32).to(model.device)
    with torch.no_grad():
        pred, _, _, _ = model.model(tile_torch)
        pred = torch.sigmoid(pred)
    tile_pred = pred.cpu().numpy().squeeze()
    
    np.save("tile_pred.npy", tile_pred)
    cv2.imwrite("tile_pred.png", (tile_pred / tile_pred.max() * 255).astype(np.uint8))
    print("Done")
    
if __name__ == "__main__":
    main()