FROM docker.arvancloud.ir/python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --trusted-host https://mirror-pypi.runflare.com -i https://mirror-pypi.runflare.com/simple/ --no-cache-dir -r requirements.txt --timeout=100
COPY ./src .
CMD ["python", "server.py"]