# flake8: noqa

import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from queue import Queue, Empty
import threading
import asyncio
import uuid
import json
from PIL import Image
from io import BytesIO
import torch
from diffusers import StableDiffusionXLPipeline
from rembg import remove, new_session
import uvicorn
from logging.handlers import RotatingFileHandler

# FastAPI app and controller setup
class Prompt(BaseModel):
    text_prompt: str
    style: str  # You can choose to handle different styles later if needed

class Controller():
    def __init__(self, config):
        # Configure logger for the Controller instance
        self.logger = logging.getLogger("Controller")
        
        # Set up rotating logs with a maximum file size and backup count
        log_handler = RotatingFileHandler(
            'controller.log',  # Log file path
            maxBytes=5 * 1024 * 1024,  # 5 MB per log file
            backupCount=5  # Keep up to 5 backup log files
        )
        
        # Set the logging level and formatter for this logger
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(formatter)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(log_handler)

        # Initialize controller attributes
        self.logger.info("[Controller] Initializing the pipeline...")
        self.events = {}
        self.image_generation_queue = Queue()
        self.background_removal_queue = Queue()
        self.bck_rmv_session = new_session()

        self.img_gen_results = {}
        self.bck_rmv_results = {}
        self.max_batch_size = 4

        self.batch_processing_size = config['inference_config']['batch_processing_size']
        self.inference_step = config['inference_config']['inference_step']
        
        model_path = config['models_config']['path']
        self.sdxl_model_path = model_path['sdxl_model']
        #self.background_remove_model_path = model_path['background_remove_model']
        #self.inpainting_fill_model_path = model_path['inpainting_fill_model']

        try:
            self.img_gen_pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                variant="fp16",
                torch_dtype=torch.float16
            )
            self.img_gen_pipe.enable_model_cpu_offload()
            self.logger.info("[Controller] SDXL pipeline initialized.")
        except Exception as e:
            self.logger.error(f"[Controller] Error initializing SDXL pipeline: {e}")

        self.background_thread = None

    def batch_image_generation(self):
        self.logger.info("[Batch] Background thread started for image generation...")
        while True:
            temp_list = []
            try:
                # Collect all available requests, up to the batch size
                for _ in range(self.max_batch_size):
                    temp_list.append(self.image_generation_queue.get(block=False))
                    self.logger.info(f"[Batch] Got a request. Queue size: {self.image_generation_queue.qsize()}")
            except Empty:
                pass

            if not temp_list:
                continue

            # Process the prompts
            request_id_list = [id_prompt_tuple[0] for id_prompt_tuple in temp_list]
            prompt_list = [id_prompt_tuple[1] for id_prompt_tuple in temp_list]

            self.logger.info(f"[Batch] Processing batch of size {len(temp_list)}...")

            try:
                # Reduce inference steps to speed up generation
                generated_images = self.img_gen_pipe(prompt_list, num_inference_steps=20).images
                self.logger.info("[Batch] Images generated successfully with reduced steps.")
            except Exception as e:
                self.logger.error(f"[Batch] Error during image generation: {e}")
                # Handle the error and set the events to avoid deadlock
                for request_id in request_id_list:
                    self.img_gen_results[request_id] = None
                    if request_id in self.events:
                        self.events[request_id].set()  # Avoid deadlock
                continue

            # Store the results and set events
            for request_id, img in zip(request_id_list, generated_images):
                self.img_gen_results[request_id] = img
                if request_id in self.events:
                    self.events[request_id].set()  # Signal the corresponding thread
                    self.logger.info(f"[Batch] Image ready for request {request_id}")

    async def image_generation_request(self, request_id, prompt):
        self.logger.info(f"[Request] Received image generation request {request_id} with prompt: {prompt}")
        event = threading.Event()
        self.events[request_id] = event

        # Add the prompt to the queue
        self.image_generation_queue.put((request_id, prompt))
        self.logger.info(f"[Request] Added request {request_id} to the queue.")

        # Wait until the image is generated
        await asyncio.to_thread(event.wait)
        self.logger.info(f"[Request] Image generated for request {request_id}")

        # Once the image is ready, retrieve it
        image = self.img_gen_results.get(request_id)

        # Ensure we have an image result
        if image is None:
            raise HTTPException(status_code=503, detail=f"Image generation failed. Please try again later.")

        # Convert the image to an in-memory file (BytesIO)
        img_io = BytesIO()
        image.save(img_io, format="JPEG")
        img_io.seek(0)  # Move the cursor to the start of the file

        # Clean up
        del self.img_gen_results[request_id]
        del self.events[request_id]
        self.logger.info(f"[Request] Image ready for request {request_id} as in-memory stream")
        return img_io

    def run(self):
        # Start the background thread for image generation.
        if self.background_thread is None:
            self.logger.info("[Controller] Starting background thread for image generation...")
            self.background_thread = threading.Thread(target=self.batch_image_generation, daemon=True)
            self.background_thread.start()
        else:
            self.logger.info("[Controller] Background thread is already running.")

app = FastAPI()
with open('./config.json', 'r') as f:
    config = json.load(f)
controller = Controller(config)

@app.post("/generate-image")
async def generate_image(prompt: Prompt):
    # Generate a unique identifier for each request
    request_id = str(uuid.uuid4())
    controller.logger.info(f"[API] Received /generate-image request {request_id}")
    img_io = await controller.image_generation_request(request_id, prompt.text_prompt)

    # Return the image directly from memory using StreamingResponse
    return StreamingResponse(img_io, media_type="image/jpeg")

@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    # Log the start of the background removal process
    controller.logger.info(f"[API] Received /remove-background request with file: {file.filename}")
    
    # Ensure the uploaded file is an image
    if not file.content_type.startswith("image/"):
        controller.logger.error(f"[API] Invalid file type for background removal: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")

    try:
        # Read the uploaded file as bytes
        input_image_bytes = await file.read()

        # Open the image as a PIL image object
        input_image = Image.open(BytesIO(input_image_bytes))
        controller.logger.info(f"[API] Image {file.filename} loaded successfully for background removal.")

        # Remove the background (returns a new image)
        output_image = remove(input_image, session=controller.bck_rmv_session)

        # Convert RGBA to RGB if needed
        if output_image.mode == 'RGBA':
            output_image = output_image.convert('RGB')
            controller.logger.info(f"[API] Converted RGBA image {file.filename} to RGB.")

        # Save the result to a BytesIO object
        output_buffer = BytesIO()
        output_image.save(output_buffer, format='JPEG')
        output_buffer.seek(0)  # Reset buffer pointer to the beginning

        controller.logger.info(f"[API] Successfully removed background from {file.filename}.")

        return StreamingResponse(output_buffer, media_type="image/jpeg")

    except Exception as e:
        # Log the error and raise an HTTPException
        controller.logger.error(f"[API] Error occurred during background removal for {file.filename}: {e}")
        raise HTTPException(status_code=503, detail="Background removal failed. Please try again later.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    controller.run()