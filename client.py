import requests
import os
from PIL import Image
from io import BytesIO

# Post the text prompt
post_url = 'http://127.0.0.1:8000/submit-text'
get_url = 'http://127.0.0.1:8000/get-image'
save_path = './GradProj/client_images/hiphop_capybara.jpg'

text_prompt = "futuristic woman"
payload = {"text": text_prompt}

try:
    # Send the POST request
    post_response = requests.post(post_url, json=payload)

    # Check if the request was successful (status code 200)
    if post_response.status_code == 200:
        # Open the image from the response content using BytesIO
        image = Image.open(BytesIO(post_response.content))

        # Check the image format (e.g., PNG, JPEG, etc.)
        print(f"Image format: {image.format}")

        # Save the image based on its format
        if image.format in ['PNG', 'JPEG', 'JPG']:
            image.save(f"downloaded_image.{image.format.lower()}")
            print("Image downloaded and saved successfully!")
        else:
            print(f"Unsupported image format: {image.format}")

    else:
        print(f"Failed to download image. Status code: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred while making the POST request: {e}")
except Exception as e:
    print(f"An error occurred: {e}")