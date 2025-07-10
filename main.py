import os
import json
import uvicorn
import asyncio
import time                                   # ← neu
import google.generativeai as genai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PORT = int(os.getenv("PORT", "8080"))

# Dynamische Domain-Erkennung
DOMAIN = os.getenv("RAILWAY_STATIC_URL") or os.getenv("NGROK_URL")
if not DOMAIN:
    DOMAIN = f"localhost:{PORT}"
if DOMAIN.startswith("https://"):
    DOMAIN = DOMAIN.replace("https://", "")

WS_URL = f"wss://{DOMAIN}/ws" if "localhost" not in DOMAIN else f"ws://{DOMAIN}/ws"

WELCOME_GREETING = "Hi! I am a voice assistant powered by Twilio and Google Gemini. Ask me anything!"

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
    model_name='gemini-2.5-flash',
    system_instruction=SYSTEM_PROMPT
)

sessions = {}

async def gemini_response(chat_session, user_prompt):
    """Get a response from the Gemini API with timeout handling and profiling."""
    try:
        start_api = time.time()
        response = await asyncio.wait_for(
            chat_session.send_message_async(user_prompt),
            timeout=15.0
        )
        api_elapsed = time.time() - start_api
        print(f"[PROFILE] Gemini API call took {api_elapsed:.2f}s")
        return response.text
    except asyncio.TimeoutError:
        return "I'm sorry, I'm having trouble processing that right now. Could you try again?"
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return "I encountered an error. Please try again."

app = FastAPI()

@app.post("/twiml")
async def twiml_endpoint():
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
    <Connect>
    <ConversationRelay url="{WS_URL}" welcomeGreeting="{WELCOME_GREETING}" ttsProvider="ElevenLabs" voice="FGY2WhTYpPnrIDTdsKH5" />
    </Connect>
    </Response>"""
    return Response(content=xml_response, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    call_sid = None
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "setup":
                call_sid = message["callSid"]
                print(f"Setup for call: {call_sid}")
                sessions[call_sid] = model.start_chat(history=[])

            elif message["type"] == "prompt":
                if not call_sid or call_sid not in sessions:
                    print(f"Error: Received prompt for unknown call_sid {call_sid}")
                    continue

                user_prompt = message["voicePrompt"]
                print(f"Received prompt: {user_prompt}")

                # Gesamt-Profiling starten
                start_total = time.time()

                chat_session = sessions[call_sid]
                response_text = await gemini_response(chat_session, user_prompt)

                total_elapsed = time.time() - start_total
                print(f"[PROFILE] Total turnaround for prompt took {total_elapsed:.2f}s")

                await websocket.send_text(
                    json.dumps({
                        "type": "text",
                        "token": response_text,
                        "last": True
                    })
                )
                print(f"Sent response: {response_text}")

            elif message["type"] == "interrupt":
                print(f"Handling interruption for call {call_sid}.")

            else:
                print(f"Unknown message type received: {message['type']}")

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