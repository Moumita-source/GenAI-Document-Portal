from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
from multi_doc_chat.src.data_ingestion import DataIngestion, IngestionArtifact

app = FastAPI()

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    
    data_ingestion = DataIngestion(files= files)
    result: IngestionArtifact  = await data_ingestion.initiate_data_ingestion()
    return {
        "session_id": result.session_id,
        "message": result.message
    }
    
    
@app.post("/chat")
async def chat_message(session_id: str = Form(...), message: str = Form(...)):
    if not session_id:
        return {"bot_reply": "No active session. Please upload a file first."}

    return {
        "session_id": session_id,
        "user_message": message,
        "bot_reply": f"This is your active session: {session_id}"
    }
