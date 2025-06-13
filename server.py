import socket
import time  # Import the time module
import cv2
import numpy as np
import threading
import os  # Import the os module


import torch

# Create the output folder if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

import json

class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]
def process_image(image_data):
    print("Data Processing Started!")

    # Convert the byte array to a NumPy array and reshape it to an image
    np_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Perform inference
    results = model(image)

    # Extract labels, confidences, and bounding boxes from the results
    labels = results.pred[0][:, -1].cpu().numpy()
    confidences = results.pred[0][:, -2].cpu().numpy()
    boxes = results.pred[0][:, :-2].cpu().numpy()

    # Prepare the response dictionary
    response_dict = {
        "number": len(labels),
        "Objects": {}
    }
    labelName = ""
    for i, (label, confidence, box) in enumerate(zip(labels, confidences, boxes)):
        label_name = class_names[int(label)]  # Convert label number to label name
        x, y, w, h = box  # Extract bounding box coordinates
        labelName = label_name
        response_dict["Objects"][f"Object{i+1}"] = {
            "label": label_name,
            "confidence": float(confidence),
            "bounding_box": {
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h),
            }
        }
    # Convert the dictionary to a JSON string
    response_json = json.dumps(response_dict)
    rendered_img = results.render()[0]  # This returns a list of numpy arrays, one for each image
    timestamp = int(time.time())
    if(labelName.replace(" ", "") != ""):
        cv2.imwrite(f"output/processed_image.jpg", rendered_img)
    
    return response_json.encode('utf-8')


def handle_client(client_socket, address):
    try:
        print(f"Connection from {address} has been established.")
        image_data = b''
        total_bytes = 0  # Variable to keep track of the total bytes received
        delimiter = b'END_OF_IMAGE'

        while True:
            packet = client_socket.recv(4096)
            packet_size = len(packet)  # Get the size of the received packet
            total_bytes += packet_size  # Update the total bytes received

            print(f"Received {packet_size} bytes. Total bytes received: {total_bytes}")

            image_data += packet

            # Check for delimiter
            if image_data[-len(delimiter):] == delimiter:
                print("Delimiter found. Breaking loop.")
                image_data = image_data[:-len(delimiter)]  # Remove delimiter
                break

        if len(image_data) == 0:
            print(f"No data received from {address}.")
            return

        result = process_image(image_data)
        print("Data Processed!")
        client_socket.sendall(result)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print(f"Closing connection from {address}.")
        client_socket.close()



if __name__ == "__main__":
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow reusing the address
    
    port = int(os.environ.get('PORT', 8080))  # Use the PORT environment variable if available
    server_socket.bind(('0.0.0.0', port))
    
    server_socket.listen(5)
    print(f"Server is listening on port {port}...")
    
    while True:
        client_socket, address = server_socket.accept()
        client_thread = threading.Thread(target=handle_client, args=(client_socket, address))
        client_thread.start()
