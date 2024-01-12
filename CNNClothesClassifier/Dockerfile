FROM python:3.10-slim
COPY requirements.txt .
WORKDIR /app
COPY . /app
RUN python -m pip install -r requirements.txt
CMD ["python", "server.py"]