FROM python:3.10


WORKDIR /app

COPY server/server.py /app/server.py


COPY ../shared /app/shared

RUN pip install tensorflow flwr

CMD ["python", "server.py"]
