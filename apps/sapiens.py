import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from huggingface_hub import snapshot_download


class Config:
    ASSETS_DIR = os.path.join(
        snapshot_download(repo_id="facebook/sapiens-normal", repo_type="space"), 'assets'
    )
    CHECKPOINTS_DIR = os.path.join(ASSETS_DIR, "checkpoints")
    CHECKPOINTS = {
        "0.3b": "sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2",
        "0.6b": "sapiens_0.6b_normal_render_people_epoch_200_torchscript.pt2",
        "1b": "sapiens_1b_normal_render_people_epoch_115_torchscript.pt2",
        "2b": "sapiens_2b_normal_render_people_epoch_70_torchscript.pt2",
    }
    SEG_CHECKPOINTS = {
        "fg-bg-1b": "sapiens_1b_seg_foreground_epoch_8_torchscript.pt2",
        "no-bg-removal": None,
        "part-seg-1b": "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
    }


class ModelManager:
    @staticmethod
    def load_model(checkpoint_name, device):
        if checkpoint_name is None:
            return None
        checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, checkpoint_name)
        model = torch.jit.load(checkpoint_path)
        model.eval()
        model.to(device)
        return model

    @staticmethod
    @torch.inference_mode()
    def run_model(model, input_tensor, height, width):
        output = model(input_tensor)
        return F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)


class ImageProcessor:
    def __init__(self, device):

        self.mean = [123.5 / 255, 116.5 / 255, 103.5 / 255]
        self.std = [58.5 / 255, 57.0 / 255, 57.5 / 255]

        self.transform_fn = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        self.device = device

    def process_image(self, image: Image.Image, normal_model_name: str, seg_model_name: str):

        # Load models here instead of storing them as class attributes
        normal_model = ModelManager.load_model(Config.CHECKPOINTS[normal_model_name], self.device)
        input_tensor = self.transform_fn(image).unsqueeze(0).to(self.device)

        # Run normal estimation
        normal_map = ModelManager.run_model(normal_model, input_tensor, image.height, image.width)

        # Run segmentation
        if seg_model_name != "no-bg-removal":
            seg_model = ModelManager.load_model(Config.SEG_CHECKPOINTS[seg_model_name], self.device)
            seg_output = ModelManager.run_model(seg_model, input_tensor, image.height, image.width)
            seg_mask = (seg_output.argmax(dim=1) > 0).unsqueeze(0).repeat(1, 3, 1, 1)

        # Normalize and visualize normal map
        normal_map_norm = torch.linalg.norm(normal_map, dim=1, keepdim=True)
        normal_map_normalized = normal_map / (normal_map_norm + 1e-5)
        normal_map_normalized[seg_mask == 0] = 0.0
        normal_map_normalized = normal_map_normalized.to(self.device)

        return normal_map_normalized
