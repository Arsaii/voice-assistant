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

async def gemini_response_streaming(chat_session, user_prompt, websocket):
    """
    Call the Gemini API with streaming, sending chunks to the WebSocket
    as they are received.
    Returns the full response text and the elapsed time for the API call.
    """
    start_api = time.time()
    full_response_text = ""
    try:
        # Use stream=True to get an async iterator of chunks
        response_stream = await asyncio.wait_for(
            chat_session.send_message_async(user_prompt, stream=True),
            timeout=15.0 # Timeout for the initial response from Gemini
        )
        
        async for chunk in response_stream:
            if chunk.text: # Ensure chunk has text content
                full_response_text += chunk.text
                # Send each chunk as a partial message
                await websocket.send_text(json.dumps({
                    "type": "text",
                    "token": chunk.text,
                    "last": False # Indicate that more content is coming
                }))
                # print(f"Sent partial token: '{chunk.text}'") # Uncomment for verbose debugging

        # After all chunks are received, send a final message to signal completion
        # An empty token with last: True ensures the client knows the message is done.
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": "", # No new token, just a signal
            "last": True # Signal end of message
        }))

    except asyncio.TimeoutError:
        error_message = "I'm sorry, I'm having trouble processing that right now. Could you try again?"
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": error_message,
            "last": True # This is the final message if timeout occurs
        }))
        return error_message, time.time() - start_api
    except Exception as e:
        print(f"Error with Gemini API streaming: {e}")
        error_message = "I encountered an error. Please try again."
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": error_message,
            "last": True # This is the final message if an error occurs
        }))
        return error_message, time.time() - start_api

    api_elapsed = time.time() - start_api
    return full_response_text, api_elapsed

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
            t0 = time.time() # Start of total turnaround time
            raw = await websocket.receive_text()
            
            # 1) Parse & preparation
            parse_start = time.time() # Start measuring parse/prep
            message = json.loads(raw)

            if message.get("type") == "setup":
                call_sid = message["callSid"]
                print(f"Setup for call: {call_sid}")
                sessions[call_sid] = model.start_chat(history=[])
                continue

            if message.get("type") != "prompt" or not call_sid:
                continue

            user_prompt = message["voicePrompt"]
            print(f"Received prompt: {user_prompt}")
            parse_prep_time = time.time() - parse_start # End measuring parse/prep

            # 2) Gemini API call (now with streaming)
            # This function now sends responses directly to the websocket
            api_response_full_text, api_time = await gemini_response_streaming(
                sessions[call_sid], user_prompt, websocket
            )

            # 3) Send & serialization (This step is now integrated into gemini_response_streaming,
            #    so we're measuring the time until all chunks are sent, rather than one big send)
            # The 'send_time' here will be minimal as the actual sending happens within the streaming function.
            # We can re-evaluate what 'send_time' means in a streaming context.
            # For this profiling, 'api_time' now implicitly includes the time taken to send all chunks.

            # 4) Total turnaround
            total_time = time.time() - t0

            # Profiling log
            # Note: parse/prep is measured until user_prompt is extracted.
            # api_time includes the time to receive all chunks AND send them over the websocket.
            # The 'send' part of the original profile is now mostly absorbed into 'api_time'
            # because the websocket sends happen during the API streaming.
            print(
                f"[PROFILE] parse/prep: {parse_prep_time:.4f}s | "
                f"api_and_sending_chunks: {api_time:.4f}s | " # Renamed for clarity
                f"total_turnaround: {total_time:.4f}s"
            )

    except WebSocketDisconnect:
        print(f"WebSocket connection closed for call {call_sid}")
        sessions.pop(call_sid, None)
        print(f"Cleared session for call {call_sid}")
    except Exception as e:
        print(f"Unexpected error in websocket_endpoint: {e}")
        if call_sid and call_sid in sessions:
            sessions.pop(call_sid, None) # Clean up session on unexpected error

@app.get("/health")
async def health_check():
    return {"status": "healthy", "domain": DOMAIN}

if __name__ == "__main__":
    print(f"Starting server on port {PORT}")
    print(f"WebSocket URL for Twilio: {WS_URL}")
    print(f"Detected platform domain: {DOMAIN}")
    # Ensure workers is appropriate for your server's CPU cores and expected load.
    # For a typical voice assistant, 2 workers is a reasonable starting point.
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=2)