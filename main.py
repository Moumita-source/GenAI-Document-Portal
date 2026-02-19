from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List

app = FastAPI()

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

import uuid

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        contents = await file.read()
        results.append({
            "filename": file.filename,
            "size": len(contents)
        })

    if results:
        # Generate a new session ID whenever files are uploaded
        session_id = str(uuid.uuid4())
        return {
            "session_id": session_id,
            "uploaded": results
        }
    else:
        return {
            "session_id": None,
            "uploaded": [],
            "error": "No files uploaded"
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
