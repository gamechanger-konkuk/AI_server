#from diffusers import StableDiffusionControlNetPipeline
#from diffusers import ControlNetModel
from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel
from diffusers import UniPCMultistepScheduler

from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image


class ImageGenerator():
    #def __init__(self, sd_path, cn_path, custom_unet_path):
    def __init__(self, sd_path, custom_unet_path):
        self.sd_path = sd_path
        self.custom_unet_path = custom_unet_path
        #self.load_image_prompt()
        self.load_pipeline()
        self.load_custom_unet()
        #self.pipe.enable_model_cpu_offload()
        self.generator = torch.manual_seed(10)

    def load_image_prompt(self):
        path = "https://hf.co/datasets/huggingface/"\
                + "documentation-images/resolve/main/"\
                + "diffusers/input_image_vermeer.png"
        image = load_image(path)
        image = np.array(image)

        # get canny image
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        self.image_prompt = Image.fromarray(image)

        return

    def load_pipeline(self):
        # controlnet = ControlNetModel.from_pretrained(
        #     self.cn_path, torch_dtype=torch.float16
        #     )
        # self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
        #     self.sd_path, controlnet=controlnet, torch_dtype=torch.float16
        #     )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.sd_path, torch_dtype=torch.float16
            )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
            )
        return

    def load_custom_unet(self):
        self.pipe.unet = UNet2DConditionModel.from_single_file(
            self.custom_unet_path, torch_dtype=torch.float16
            )

    def image_generation(self, text_prompt):
        image = self.pipe(
            text_prompt,
            num_inference_steps=20,
            generator=self.generator
            #image=self.image_prompt
            ).images[0]

        return image
