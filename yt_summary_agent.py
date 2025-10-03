from youtube_transcript_api import YouTubeTranscriptApi
import os
import openai
    
def fetch_youtube_transcript(video_id: str) -> str:
    """Fetch transcript text for a given YouTube video id as a single string."""
    transcript_api = YouTubeTranscriptApi()
    segments = transcript_api.fetch(video_id)

    full_text = " ".join(segment.text for segment in segments).strip()
    return full_text


def summarize_text_with_openai(text: str, model: str = "gpt-4o-mini") -> str:
    """Summarize provided text using OpenAI's chat completion models.

    Expects OPENAI_API_KEY in the environment (used automatically by the SDK).
    """
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = openai.OpenAI()
    prompt = (
        "You are a helpful assistant that summarizes YouTube transcripts. "
        "Produce a clear, concise summary capturing the main points, structure, and key takeaways. Return this as a bulletted list."
    )

    # Light truncation safeguard to avoid excessively long inputs
    max_chars = 20_000
    input_text = text if len(text) <= max_chars else text[:max_chars]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Summarize the following transcript:\n\n{input_text}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def summarize_video(video_id: str, model: str = "gpt-4o-mini") -> str:
    """Fetch the transcript for a video and return an OpenAI-generated summary."""
    transcript = fetch_youtube_transcript(video_id)
    if not transcript:
        return ""
    return summarize_text_with_openai(transcript, model=model)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python yt_summary_agent.py <youtube_video_id>")
        raise SystemExit(1)

    video_id = sys.argv[1]
    print(summarize_video(video_id))
