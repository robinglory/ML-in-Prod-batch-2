FROM python:3.10

WORKDIR /app

COPY client/client.py /app/client.py

COPY ../shared /app/shared

RUN pip install tensorflow flwr

CMD ["python", "client.py"]
