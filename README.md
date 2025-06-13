# Human Detection Server with YOLOv5

-----

This project provides a robust Python-based server for real-time human detection using the **YOLOv5** deep learning model. It's designed to receive image data over a socket, perform object detection, and return JSON-formatted results, along with saving the annotated image. Additionally, it includes a simple client-side script to convert an image to ASCII art, and a Kubernetes Service configuration for deployment.

## Table of Contents

  * [Features](https://www.google.com/search?q=%23features)
  * [How it Works](https://www.google.com/search?q=%23how-it-works)
  * [Project Structure](https://www.google.com/search?q=%23project-structure)
  * [Setup and Installation](https://www.google.com/search?q=%23setup-and-installation)
      * [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      * [Setting up the Server](https://www.google.com/search?q=%23setting-up-the-server)
      * [Setting up the Client (ASCII Art Converter)](https://www.google.com/search?q=%23setting-up-the-client-ascii-art-converter)
  * [Usage](https://www.google.com/search?q=%23usage)
      * [Running the Server](https://www.google.com/search?q=%23running-the-server)
      * [Sending Images to the Server](https://www.google.com/search?q=%23sending-images-to-the-server)
      * [Using the ASCII Art Converter](https://www.google.com/search?q=%23using-the-ascii-art-converter)
  * [Kubernetes Deployment](https://www.google.com/search?q=%23kubernetes-deployment)
  * [Troubleshooting](https://www.google.com/search?q=%23troubleshooting)
  * [Contributing](https://www.google.com/search?q=%23contributing)
  * [License](https://www.google.com/search?q=%23license)

-----

## Features

  * **Real-time Object Detection:** Leverages the powerful YOLOv5 model for efficient and accurate human detection.
  * **Socket Communication:** Utilizes TCP sockets for robust client-server image data transfer.
  * **JSON Response:** Returns detection results (number of objects, labels, confidences, bounding boxes) in a structured JSON format.
  * **Annotated Image Output:** Saves processed images with bounding boxes and labels to an `output/` directory.
  * **Multi-threaded Server:** Handles multiple client connections concurrently for improved performance.
  * **Dynamic Port Configuration:** Server port can be configured via environment variable (`PORT`).
  * **ASCII Art Conversion (Client-side):** Includes a utility to convert any image into ASCII art.
  * **Kubernetes Ready:** Provides a `Service` manifest for easy deployment to a Kubernetes cluster.

-----

## How it Works

1.  **Server (`server.py`):**

      * Listens for incoming TCP connections on a specified port (default 8080).
      * Upon connection, it receives raw image bytes from the client. The image data is expected to be terminated by a `END_OF_IMAGE` delimiter.
      * Converts the received bytes into a NumPy array and then into an OpenCV image.
      * Loads the YOLOv5 model (using `torch.hub.load`).
      * Performs inference on the image to detect objects.
      * Extracts detection details (labels, confidences, bounding boxes).
      * Constructs a JSON response containing the detection information.
      * Renders the detected objects on the image and saves it to the `output/` directory.
      * Sends the JSON response back to the client.
      * Each client connection is handled in a separate thread to avoid blocking.

2.  **Client (Implicit from provided code):**

      * While a dedicated client script for sending images wasn't provided, the `image_to_ascii` function implies a client-side component.
      * A typical client would read an image, convert it to bytes, append the `END_OF_IMAGE` delimiter, and send it to the server.
      * The ASCII art converter is a separate utility that processes local images.

3.  **YOLOv5 Integration:**

      * The server dynamically loads the YOLOv5s model from `ultralytics/yolov5` via PyTorch Hub. This requires an internet connection for the first run to download the model weights.

-----

## Project Structure

```
.
├── output/                   # Directory for saving processed images (created by server)
├── server.py                 # Main server application with YOLOv5 detection logic
├── requirements.txt          # (Implicit, but needed for Python dependencies)
├── Dockerfile                # Docker build instructions for the server
├── image-recognition-service.yaml # Kubernetes Service manifest
└── example/                  # (Suggested) Directory for client-side scripts, example images, etc.
    └── main.go               # Example Go code (from previous context, not directly part of server/client)
    └── example_image.jpg     # An example image for testing (if added)
    └── client.py             # (Suggested) A Python client script to send images
    └── ascii_converter.py    # (Suggested) Separate script for image_to_ascii functionality
```

-----

## Setup and Installation

### Prerequisites

  * **Python 3.8+**: The server is built with Python.
  * **pip**: Python package installer.
  * **Git**: For cloning the repository and handling submodules.
  * **Docker** (Optional, for containerization): To build and run the server in a Docker container.
  * **Kubernetes Cluster** (Optional, for deployment): To deploy the server to Kubernetes.
  * **CUDA-compatible GPU** (Optional, highly recommended for performance): For faster YOLOv5 inference. If not available, it will run on CPU.

### Setting up the Server

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/sidsrbh/Human-Detection-Server.git
    cd "Human Detection Python Server" # Navigate into the project directory
    ```

2.  **Initialize and Update YOLOv5 Submodule:**
    The project uses YOLOv5 as a Git submodule.

    ```bash
    git submodule update --init --recursive
    ```

    This will clone the YOLOv5 repository into the `yolov5/` directory within your project.

3.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Python dependencies:**
    You'll need `torch`, `opencv-python-headless`, `numpy`, `ultralytics` (which YOLOv5 depends on), and `Pillow`.

    Create a `requirements.txt` file at the root of your project with these contents:

    ```
    torch
    opencv-python-headless
    numpy
    ultralytics
    Pillow
    ```

    Then, install them:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: Installing `torch` can be tricky depending on your CUDA setup. Refer to the [PyTorch website](https://pytorch.org/get-started/locally/) for specific installation commands if you encounter issues, especially for GPU support.*

### Setting up the Client (ASCII Art Converter)

The `image_to_ascii` function is currently bundled within `server.py` in the provided code snippet. For better separation, it's recommended to move it to a dedicated client or utility script (e.g., `ascii_converter.py`).

If you run `server.py` as `__main__`, it will execute the ASCII conversion part.

To run the ASCII art converter as a standalone utility, ensure you have `Pillow` installed:

```bash
pip install Pillow
```

Then, you can run the relevant code block with your desired `image_path`.

-----

## Usage

### Running the Server

1.  **Navigate to the project directory:**

    ```bash
    cd "Human Detection Python Server"
    ```

2.  **Activate your virtual environment (if you created one):**

    ```bash
    source venv/bin/activate
    ```

3.  **Run the server:**

    ```bash
    python server.py
    ```

    The server will start listening on port 8080 by default.
    You can specify a different port using the `PORT` environment variable:

    ```bash
    PORT=12345 python server.py
    ```

    *The first time `process_image` is called, `torch.hub.load` will download the YOLOv5 model weights. This might take a few moments depending on your internet connection.*

### Sending Images to the Server

You'll need a separate client script to send images. Here's an example Python client script (`client.py`) you can create in the `example/` directory:

```python
# example/client.py
import socket
import cv2
import numpy as np
import time

def send_image(host, port, image_path):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host, port))
        print(f"Connected to server at {host}:{port}")

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return

        # Encode image to JPEG bytes
        _, img_encoded = cv2.imencode('.jpg', image)
        image_data = img_encoded.tobytes()

        delimiter = b'END_OF_IMAGE'
        full_data = image_data + delimiter

        # Send data in chunks
        chunk_size = 4096
        for i in range(0, len(full_data), chunk_size):
            chunk = full_data[i:i + chunk_size]
            client_socket.sendall(chunk)
            print(f"Sent {len(chunk)} bytes. Total sent: {i + len(chunk)}")
            time.sleep(0.001) # Small delay to avoid overwhelming the server

        print("Image data sent successfully.")

        # Receive response
        response_bytes = b''
        while True:
            packet = client_socket.recv(4096)
            if not packet:
                break
            response_bytes += packet
            if len(packet) < 4096: # Assuming the response is relatively small and fits in few packets
                break

        print("\nServer Response:")
        print(response_bytes.decode('utf-8'))

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client_socket.close()
        print("Connection closed.")

if __name__ == "__main__":
    SERVER_HOST = '127.0.0.1' # Change to your server's IP if not local
    SERVER_PORT = 8080       # Match the server's listening port
    IMAGE_TO_SEND = 'path/to/your/image.jpg' # <<<<<< IMPORTANT: Replace with your image path

    send_image(SERVER_HOST, SERVER_PORT, IMAGE_TO_SEND)
```

To run this client:

1.  Save the above code as `client.py` inside an `example/` directory.
2.  **Crucially, change `IMAGE_TO_SEND` to the actual path of an image file on your system.**
3.  Ensure your server is running.
4.  Run the client:
    ```bash
    python example/client.py
    ```
    Check the `output/` directory in your server's location for the processed image.

### Using the ASCII Art Converter

The ASCII art converter is a standalone function. If you move it to `example/ascii_converter.py`:

```python
# example/ascii_converter.py
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
    chars = ["*", "*", "*", " ", " ", " ", " "] # You can experiment with different characters here for varied density
    
    print("\n--- ASCII Art ---")
    # Generate the ASCII art
    for i in range(img_array.shape[0]):
        ascii_str = ""
        for j in range(img_array.shape[1]):
            pixel_value = img_array[i, j]
            index = pixel_value * (len(chars) - 1) // 255
            ascii_str += chars[index]
        print(ascii_str)
    print("-----------------\n")

if __name__ == "__main__":
    image_path = "/your/image/path/Black Logo_No BG.png"  # <<<<<< IMPORTANT: Replace with the path to your image
    image_to_ascii(image_path)
```

To run this:

1.  Save the code as `ascii_converter.py` in `example/` (or wherever you prefer).
2.  **Crucially, change `image_path` to the actual path of the image you want to convert.**
3.  Run the script:
    ```bash
    python example/ascii_converter.py
    ```
    The ASCII art will be printed to your console.

-----

## Kubernetes Deployment

The project includes a `Dockerfile` for building a Docker image of the server and a `image-recognition-service.yaml` for deploying it to Kubernetes.

### Building the Docker Image

1.  **Navigate to the project root:**

    ```bash
    cd "Human Detection Python Server"
    ```

2.  **Build the Docker image:**

    ```bash
    docker build -t human-detection-server:latest .
    ```

    This command builds the Docker image named `human-detection-server` with the tag `latest`.

### Deploying to Kubernetes

The `image-recognition-service.yaml` file defines a Kubernetes `Service` of type `LoadBalancer` which exposes the server. You'll also need a `Deployment` manifest to actually run the server pods.

**1. Create a Deployment (e.g., `human-detection-deployment.yaml`):**

```yaml
# human-detection-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: human-detection-app
  labels:
    app: human-detection-app
spec:
  replicas: 1 # You can increase this for more instances
  selector:
    matchLabels:
      app: human-detection-app
  template:
    metadata:
      labels:
        app: human-detection-app
    spec:
      containers:
      - name: human-detection-server
        image: human-detection-server:latest # Use the image you just built
        ports:
        - containerPort: 8080
        env:
        - name: PORT
          value: "8080" # Ensure this matches the server's listening port
        # Add resource limits if needed, especially for GPU usage
        # resources:
        #   limits:
        #     nvidia.com/gpu: 1 # If using a GPU-enabled cluster and base image
```

**2. Apply the Deployment and Service manifests:**

```bash
kubectl apply -f human-detection-deployment.yaml
kubectl apply -f image-recognition-service.yaml
```

**3. Check the status:**

```bash
kubectl get deployments
kubectl get services
kubectl get pods
```

Once the `LoadBalancer` service gets an external IP, you can use that IP to connect to your server.

**Important Considerations for Kubernetes:**

  * **YOLOv5 Model Download:** The model will be downloaded inside the container the first time `process_image` is called. For production, consider pre-baking the model into your Docker image or using a persistent volume.
  * **GPU Support:** For GPU inference, your Dockerfile would need to use a CUDA-enabled base image (e.g., `nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04`) and you'd need a Kubernetes cluster with NVIDIA GPU operators installed. Installing `torch` with CUDA support in the Dockerfile would also be necessary.
  * **Resource Limits:** Define appropriate CPU and memory limits in your Deployment manifest.

-----

## Troubleshooting

  * **`ssh: Could not resolve hostname github.com-sidsrbh: nodename nor servname provided, or not known`**: This is a Git/SSH configuration issue. Ensure your `~/.ssh/config` file has the correct `Hostname github.com` entry for your `github-sidsrbh` alias, and your remote URL uses `git@github-sidsrbh:`.
  * **`ERROR: Permission to sidsrbh/set.git denied to indicarena`**: This indicates Git is using the wrong identity for a push. Verify your `git remote -v` output points to the correct GitHub account, and clear any conflicting cached credentials if necessary.
  * **`index out of range` panic in Go graph code**: This is a Go programming error, likely due to incorrect slice/array indexing. Review the logic for resizing and accessing your adjacency matrix in `AddNodeAdjMatrix` and `AddEdgeAdjMatrix`.
  * **`torch.hub.load` download issues**: Ensure your server has active internet access to download the YOLOv5 model weights the first time.
  * **"No data received" / Delimiter Issues**: Double-check that your client is sending the `END_OF_IMAGE` delimiter correctly and the server is expecting the exact same bytes.
  * **"Directory not empty" warnings**: Harmless when adding submodules; it just means Git won't delete existing local files.

-----

## Contributing

Feel free to open issues or submit pull requests for any improvements, bug fixes, or new features.

-----

## License

MIT License

-----