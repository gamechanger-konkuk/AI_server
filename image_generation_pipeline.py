from diffusers import StableDiffusionXLPipeline
import torch

class Image_generation_pipeline():
    def __init__(self, sdxl_model_path, inference_step, textual_inversion_paths, seed):
        self.inference_step = inference_step
        self.seed = seed

        self.pipe = StableDiffusionXLPipeline.from_pretrained(sdxl_model_path, torch_dtype=torch.float16)

        for path in textual_inversion_paths:
            self.pipe.load_textual_inversion(path)

    def generate_image(self):
        return self.pipe(
            "futuristic woman in style of Claude Monet's painting", 
            num_inference_steps=20, 
            generator=generator,
            
        )

    def style2prompt(self, prompt):
        # Here, you'd implement logic to modify the prompt based on the style
        return prompt.text + " in the style of " + prompt.style
        
# Load the Stable Diffusion model pipeline
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")

# Load different pre-trained Textual Inversion embeddings from Hugging Face
#pipe.load_textual_inversion("sd-concepts-library/cat-toy")
#pipe.load_textual_inversion("sd-concepts-library/nebula")

# List of prompts using the textual inversion tokens (e.g., <low-poly>, <arcane>, <moonrise>)
prompts = [
    "A portrait of a <cat-toy> warrior",
    "A woman in <nebula> style"
]

generator = torch.manual_seed(3)
# Generate images using different embeddings for each prompt
images = pipe("futuristic woman in style of Claude Monet's painting", num_inference_steps=20, generator=generator)

image1 = images.images[0]