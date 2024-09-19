from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
from sdwithcontrolnet import ImageGenerator
from PIL import Image
from pydantic import BaseModel

app = FastAPI()
image_generator = None
curr_prompt = None

class Prompt(BaseModel):
    text: str

@app.post("/submit-text")
async def post_prompt(prompt: Prompt):
    global curr_prompt
    curr_prompt = prompt.text
    return {"message": "Text received", "img": curr_prompt}

@app.get("/get-image")
async def get_image():
    # save_path = "./GradProj_Test/server_images/generated.jpg"
    # generated_image = image_generator.image_generation(curr_prompt)

    # img = Image.open(generated_image)
    # if img.mode in ("RGBA", "P"):
    #     img = img.convert("RGB")
    # img.save(save_path, "JPEG")
    save_path = "./GradProj_Test/server_images/hiphop_capybara.jpg"

    print(f'Image successfully generated: {save_path}')

    if os.path.exists(save_path):
        return FileResponse(path=save_path, filename="hiphop_capybara.jpg", media_type='image/jpeg')
    else:
        return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    sd = "runwayml/stable-diffusion-v1-5"
    custom = "C:/Users/gygy9/AI_models/StableDiffusion_v1.5/custom_unet/diffusion_pytorch_model.safetensors"
    image_generator = ImageGenerator(sd, custom)

    uvicorn.run(app, host="127.0.0.1", port=8000)