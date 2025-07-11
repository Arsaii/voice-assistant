import os
import json
import uvicorn
import asyncio
import time
import google.generativeai as genai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PORT = int(os.getenv("PORT", "8080"))

# Robust domain detection: Railway, ngrok or localhost fallback
DOMAIN = os.getenv("RAILWAY_STATIC_URL") or os.getenv("NGROK_URL")
if not DOMAIN:
    DOMAIN = f"localhost:{PORT}"
if DOMAIN.startswith("https://"):
    DOMAIN = DOMAIN.replace("https://", "")

WS_URL = (
    f"wss://{DOMAIN}/ws"
    if "localhost" not in DOMAIN
    else f"ws://{DOMAIN}/ws"
)

WELCOME_GREETING = (
    "Hello, how can I help?"
)

SYSTEM_PROMPT = """You are a helpful and friendly voice assistant. This conversation is happening over a phone call, so your responses will be spoken aloud. 
Please adhere to the following rules:
1. Provide clear, concise, and direct answers.
2. Spell out all numbers (e.g., say 'one thousand two hundred' instead of 1200).
3. Do not use any special characters like asterisks, bullet points, or emojis.
4. Keep the conversation natural and engaging."""

# --- Gemini API Initialization ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT
)

# Store active chat sessions
sessions: dict[str, any] = {}

async def gemini_response(chat_session, user_prompt):
    """
    Call the Gemini API with timeout handling and return both
    the response text and the elapsed time for the API call.
    """
    start_api = time.time()
    try:
        response = await asyncio.wait_for(
            chat_session.send_message_async(user_prompt),
            timeout=15.0
        )
    except asyncio.TimeoutError:
        return "I'm sorry, I'm having trouble processing that right now. Could you try again?", time.time() - start_api
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return "I encountered an error. Please try again.", time.time() - start_api

    api_elapsed = time.time() - start_api
    return response.text, api_elapsed

# Create FastAPI app
app = FastAPI()

@app.post("/twiml")
async def twiml_endpoint():
    """Return TwiML to connect Twilio to our WebSocket."""
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <ConversationRelay
      url="{WS_URL}"
      welcomeGreeting="{WELCOME_GREETING}"
      ttsProvider="ElevenLabs"
      voice="FGY2WhTYpPnrIDTdsKH5" />
  </Connect>
</Response>"""
    return Response(content=xml_response, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint with detailed profiling."""
    await websocket.accept()
    call_sid = None

    try:
        while True:
            t0 = time.time()
            raw = await websocket.receive_text()
            t1 = time.time()

            # 1) Parse & preparation
            parse_start = t1
            message = json.loads(raw)

            if message.get("type") == "setup":
                call_sid = message["callSid"]
                print(f"Setup for call: {call_sid}")
                sessions[call_sid] = model.start_chat(history=[])
                continue

            if message.get("type") != "prompt" or not call_sid:
                continue

            prompt_ready = time.time()
            user_prompt = message["voicePrompt"]
            print(f"Received prompt: {user_prompt}")

            # 2) Gemini API call
            api_response, api_time = await gemini_response(sessions[call_sid], user_prompt)

            # 3) Send & serialization
            send_start = time.time()
            await websocket.send_text(json.dumps({
                "type": "text",
                "token": api_response,
                "last": True
            }))
            send_time = time.time() - send_start

            # 4) Total turnaround
            total_time = time.time() - t0

            # Profiling log
            print(
                f"[PROFILE] parse/prep: {(prompt_ready - parse_start):.2f}s | "
                f"api: {api_time:.2f}s | "
                f"send: {send_time:.2f}s | "
                f"total: {total_time:.2f}s"
            )

    except WebSocketDisconnect:
        print(f"WebSocket connection closed for call {call_sid}")
        sessions.pop(call_sid, None)
        print(f"Cleared session for call {call_sid}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "domain": DOMAIN}

if __name__ == "__main__":
    print(f"Starting server on port {PORT}")
    print(f"WebSocket URL for Twilio: {WS_URL}")
    print(f"Detected platform domain: {DOMAIN}")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=2)