# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org opencv-python-headless numpy

# Make port 80 available to the world outside this container
EXPOSE 12346

# Run your Python script when the container launches
CMD ["python", "server.py"]
