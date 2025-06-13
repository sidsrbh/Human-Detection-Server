from PIL import Image
import numpy as np

def image_to_ascii(image_path):
    # Open the image
    img = Image.open(image_path).convert("RGBA")
    
    # Create an output image with white background
    output_img = Image.new("RGBA", img.size, "white")
    output_img.paste(img, (0, 0), img)
    
    # Convert the image to grayscale
    grayscale_img = output_img.convert("L")
    
    # Resize the image
    width, height = grayscale_img.size
    aspect_ratio = height / width
    new_width = 100  # You can change this value for a larger or smaller image
    new_height = int(aspect_ratio * new_width * 0.55)  # 0.55 is a correction factor
    resized_img = grayscale_img.resize((new_width, new_height))
    
    # Convert the image to a NumPy array
    img_array = np.array(resized_img)
    
    # Define the characters to replace pixel values
    chars = ["*", "*", "*", " ", " ", " ", " "]
    
    # Generate the ASCII art
    for i in range(img_array.shape[0]):
        ascii_str = ""
        for j in range(img_array.shape[1]):
            pixel_value = img_array[i, j]
            index = pixel_value * (len(chars) - 1) // 255
            ascii_str += chars[index]
        print(ascii_str)

if __name__ == "__main__":
    image_path = "/your/image/path/Black Logo_No BG.png"  # Replace with the path to your image
    image_to_ascii(image_path)
