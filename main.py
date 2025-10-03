from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from yt_summary_agent import fetch_youtube_transcript, summarize_video, summarize_text_with_openai


app = FastAPI(title="YouTube Transcript/Summary Service")


class TranscriptResponse(BaseModel):
    video_id: str
    transcript: str


class SummaryResponse(BaseModel):
    video_id: str
    summary: str


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


@app.get("/summarize_text", response_model=SummaryTextResponse)
async def summarize_text(
    text: str = Query(..., min_length=5),
    model: str = Query("gpt-4o-mini"),
) -> SummaryTextResponse:
    try:
        summary = summarize_text_with_openai(text, model=model)
        if not summary:
            raise HTTPException(status_code=404, detail="Summary could not be generated")
        return SummaryTextResponse(text=text, summary=summary)
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

