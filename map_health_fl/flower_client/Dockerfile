# Use a PyTorch-compatible base image
FROM docker.arvancloud.ir/python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --trusted-host https://mirror-pypi.runflare.com -i https://mirror-pypi.runflare.com/simple/ --no-cache-dir -r requirements.txt --timeout=100

# Copy the source code
COPY ./src .

# Command to run the client
# The CLIENT_ID will be passed from docker-compose
CMD ["python", "-u", "client.py"]