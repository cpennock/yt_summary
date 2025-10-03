from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from yt_summary_agent import fetch_youtube_transcript, summarize_video, summarize_text_with_openai


app = FastAPI(title="YouTube Transcript/Summary Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class TranscriptResponse(BaseModel):
    video_id: str
    transcript: str


class SummaryResponse(BaseModel):
    video_id: str
    summary: str


class SummaryTextRequest(BaseModel):
    text: str
    model: str = "gpt-4o-mini"

class SummaryTextResponse(BaseModel):
    text: str
    summary: str

@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/transcript", response_model=TranscriptResponse)
async def get_transcript(video_id: str = Query(..., min_length=5)) -> TranscriptResponse:
    print("in /transcript")
    try:
        transcript = fetch_youtube_transcript(video_id)
        print("transcript", transcript)
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found or empty")
        return TranscriptResponse(video_id=video_id, transcript=transcript)
    except Exception as exc:  # YouTubeTranscriptApi raises various exceptions; map to 400
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/summarize_text", response_model=SummaryTextResponse)
async def summarize_text(request: SummaryTextRequest) -> SummaryTextResponse:
    try:
        if len(request.text) < 5:
            raise HTTPException(status_code=400, detail="Text must be at least 5 characters long")
        summary = summarize_text_with_openai(request.text, model=request.model)
        if not summary:
            raise HTTPException(status_code=404, detail="Summary could not be generated")
        return SummaryTextResponse(text=request.text, summary=summary)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/summary", response_model=SummaryResponse)
async def get_summary(
    video_id: str = Query(..., min_length=5),
    model: str = Query("gpt-4o-mini"),
) -> SummaryResponse:
    try:
        summary = summarize_video(video_id, model=model)
        if not summary:
            raise HTTPException(status_code=404, detail="Summary could not be generated")
        return SummaryResponse(video_id=video_id, summary=summary)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# To run locally: `uvicorn web_app:app --host 0.0.0.0 --port 8000`

