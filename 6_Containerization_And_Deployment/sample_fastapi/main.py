from fastapi import FastAPI, Request,Body
from schemas import studentRequestModel,textRequestModel,textResponseModel
import time
import uvicorn
import asyncio


from fastapi import status,Response
from fastapi.responses import StreamingResponse


from contextlib import asynccontextmanager
from model_work import textModel,audioModel



ml_models ={}

@asynccontextmanager
async def liefspan(app : FastAPI):
    text_m_obj = textModel()
    text_m_obj.load_pipeline()
    ml_models["text_m_obj"] = text_m_obj


    audio_m_obj = audioModel()
    audio_m_obj.load_audio_model()
    ml_models["audio_m_obj"] = audio_m_obj

    yield
    ml_models.clear()




    



app = FastAPI(lifespan=liefspan)


@app.get("/")
def home():
    return "hello world"


@app.post("/get_student")
def get_student(request : Request,
                body : studentRequestModel = Body(...)) -> textResponseModel:
    
    start_time = time.time()
    result = "OK"
    return textResponseModel(
        execution_time=int(time.time()-start_time),
        result=result
    )


@app.post("/sync")
def sync_prediction(prompt: str) -> textResponseModel:
    start_time = time.time()
    time.sleep(5)

    
    result = "OK"
    return textResponseModel(
        execution_time=int(time.time()-start_time),
        result=result
    )



@app.post("/async")
async def async_prediction() ->textResponseModel:
    start_time = time.time()
    await asyncio.sleep(5)

    result = "OK"
    return textResponseModel(
        execution_time=int(time.time()-start_time),
        result=result
    )


@app.post("/text_gen")
def serve_text_gen(request : Request,
                body : textRequestModel = Body(...)) -> textResponseModel:
    start_time = time.time()
    generated_text = ml_models["text_m_obj"].predict(user_message = body.prompt)

    return textResponseModel(
            execution_time=int(time.time()-start_time),
            result=generated_text
        )

@app.post("/audio_gen",
          responses={status.HTTP_200_OK:{"content" : {"audio/wav":{}}}},
          response_class=StreamingResponse,)

async def serve_audio_gen(request : Request,
                body : textRequestModel = Body(...)) -> StreamingResponse:
    
    output_audio_array = ml_models["audio_m_obj"].generate_audio(body.prompt)
    
    return StreamingResponse(output_audio_array, media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8888, reload=True)