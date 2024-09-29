# flake8: noqa

from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import torch

class Image_generation_pipeline():
    def __init__(self, config, model):
        self.model = model
        self.num_inference_step = config['inference_config']['num_inference_step']
        model_path = config['path'][self.model]

        if self.model == 'sdxl':
            self.pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
            self.pipe.enable_model_cpu_offload()
        elif self.model == 'sd1.5':
            self.pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
            self.pipe.enable_model_cpu_offload()
        elif self.model == 'sdxl_turbo':
            self.pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
            self.pipe.enable_model_cpu_offload()

    def generate_image(self, prompt_list):
        return self.pipe(
            self.style2prompt(prompt_list), 
            num_inference_steps=self.num_inference_step
        )

    def style2prompt(self, prompt_list):
        # Here, you'd implement logic to modify the prompt based on the style
        prompt_str_list = [prompt.text_prompt + " in the style of " + prompt.style for prompt in prompt_list] 
        return prompt_str_list