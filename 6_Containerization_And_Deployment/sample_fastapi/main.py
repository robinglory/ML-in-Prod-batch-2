from fastapi import FastAPI, Request,Body
from schemas import studentRequestModel
import uvicorn

app = FastAPI()


@app.get("/")
def home():
    return "hello world"


@app.post("/get_student")
def get_student(request : Request,
                body : studentRequestModel = Body(...)):
    return "OK"

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8888, reload=True)