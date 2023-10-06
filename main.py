from typing import Union

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from inference import imageClassPrediction
from inference_comment_tox import textClassifier
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    text: str


origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    contents = await file.read()
    result = imageClassPrediction(contents)
    return {"Name": result}

@app.post("/uploadtext")
async def create_upload_text(item: Item):
    print(item.text)
    result = textClassifier(item.text)
    return {"result": result}