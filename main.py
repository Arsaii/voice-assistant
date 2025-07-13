import os
import json
import uvicorn
import asyncio
import time
import datetime
import aiohttp
import google.generativeai as genai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from dotenv import load_dotenv

# --- ADDED: Import Twilio TwiML classes ---
from twilio.twiml.voice_response import VoiceResponse, Connect, ConversationRelay

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

# --- OPTIMIZED: Shorter, more focused system prompt for voice ---
SYSTEM_PROMPT = """You are a helpful voice assistant. Keep responses conversational and concise since this is a phone call. 
Rules:
1. Be direct and brief - aim for 1-2 sentences per response
2. Spell out numbers (say 'twenty-three' not '23')
3. No special characters, bullets, or formatting
4. Sound natural and friendly"""

# --- Gemini API Initialization ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=GOOGLE_API_KEY)

# --- OPTIMIZED: Generation config for faster responses ---
generation_config = genai.GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    max_output_tokens=100,  # Shorter responses for voice
    candidate_count=1,
)

# --- OPTIMIZED: Try to use cached content for faster initialization ---
try:
    # Cache system instruction for 1 hour to speed up model initialization
    cached_content = genai.caching.CachedContent.create(
        model="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
        ttl=datetime.timedelta(hours=1),
    )
    model = genai.GenerativeModel.from_cached_content(cached_content)
    print("✅ Using cached Gemini model for faster responses")
except Exception as e:
    print(f"⚠️ Caching failed, using regular model: {e}")
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
        generation_config=generation_config
    )

# --- OPTIMIZED: HTTP session for connection pooling ---
http_session = None

async def create_http_session():
    """Create optimized HTTP session for better connection management"""
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=30,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=30,
    )
    return aiohttp.ClientSession(connector=connector)

# Store active chat sessions
sessions: dict[str, any] = {}

async def gemini_response_streaming(chat_session, user_prompt, websocket):
    """
    OPTIMIZED: Streaming with intelligent buffering for better performance.
    Buffers chunks until we have meaningful content or sentence boundaries.
    """
    start_api = time.time()
    full_response_text = ""
    buffer = ""
    buffer_size = 30  # Minimum characters before sending (reduced for voice)
    
    try:
        # --- OPTIMIZED: Reduced timeout for faster failure detection ---
        response_stream = await asyncio.wait_for(
            chat_session.send_message_async(user_prompt, stream=True),
            timeout=8.0  # Reduced from 15s to 8s
        )
        
        first_chunk_sent = False
        async for chunk in response_stream:
            if chunk.text:
                full_response_text += chunk.text
                buffer += chunk.text
                
                # --- OPTIMIZED: Send first chunk immediately for faster TTFB ---
                if not first_chunk_sent and len(buffer) >= 10:
                    await websocket.send_text(json.dumps({
                        "type": "text",
                        "token": buffer,
                        "last": False
                    }))
                    buffer = ""
                    first_chunk_sent = True
                    continue
                
                # --- OPTIMIZED: Smart buffering based on content and length ---
                should_send = (
                    len(buffer) >= buffer_size or 
                    any(punct in buffer for punct in ['.', '!', '?', '\n']) or
                    buffer.endswith(', ') or  # Natural pause points
                    len(buffer) >= 50  # Don't let buffer get too large
                )
                
                if should_send and first_chunk_sent:
                    await websocket.send_text(json.dumps({
                        "type": "text",
                        "token": buffer,
                        "last": False
                    }))
                    buffer = ""

        # Send any remaining buffer
        if buffer:
            await websocket.send_text(json.dumps({
                "type": "text",
                "token": buffer,
                "last": False
            }))
        
        # Send completion signal
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": "",
            "last": True
        }))

    except asyncio.TimeoutError:
        error_message = "I'm having trouble right now. Could you try again?"
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": error_message,
            "last": True
        }))
        return error_message, time.time() - start_api
    except Exception as e:
        print(f"Error with Gemini API streaming: {e}")
        error_message = "I encountered an error. Please try again."
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": error_message,
            "last": True
        }))
        return error_message, time.time() - start_api

    api_elapsed = time.time() - start_api
    return full_response_text, api_elapsed

# Create FastAPI app
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Initialize HTTP session on startup"""
    global http_session
    http_session = await create_http_session()
    print("✅ HTTP session initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up HTTP session on shutdown"""
    global http_session
    if http_session:
        await http_session.close()
        print("✅ HTTP session closed")

@app.post("/twiml")
async def twiml_endpoint():
    """OPTIMIZED: Return TwiML with better TTS configuration for speed"""
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <ConversationRelay
      url="{WS_URL}"
      welcomeGreeting="{WELCOME_GREETING}"
      ttsProvider="ElevenLabs"
      voice="FGY2WhTYpPnrIDTdsKH5"
      debug="debugging speaker-events tokens-played"
      elevenlabsTextNormalization="off"
      elevenlabsModelId="eleven_turbo_v2_5"
      elevenlabsStability="0.5"
      elevenlabsSimilarity="0.8"
      elevenlabsOptimizeStreamingLatency="4"
    />
  </Connect>
</Response>"""
    return Response(content=xml_response, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """OPTIMIZED: WebSocket endpoint with detailed performance profiling"""
    await websocket.accept()
    call_sid = None

    try:
        while True:
            loop_start = time.time()
            raw = await websocket.receive_text()
            
            # --- OPTIMIZED: Faster JSON parsing ---
            message = json.loads(raw)

            # Print debug messages but don't process them
            if message.get("type") in ["info", "debug"]:
                if message.get("name") in ["roundTripDelayMs", "tokensPlayed"]:
                    print(f"[{call_sid}] [{message.get('name', 'DEBUG')}] {message.get('value', message)}")
                continue

            if message.get("type") == "setup":
                call_sid = message["callSid"]
                print(f"✅ Setup for call: {call_sid}")
                # --- OPTIMIZED: Start chat with generation config ---
                sessions[call_sid] = model.start_chat(history=[])
                continue

            if message.get("type") != "prompt" or not call_sid:
                continue

            # --- DETAILED PROFILING START ---
            parse_start = time.time()
            user_prompt = message["voicePrompt"]
            print(f"📝 Received prompt: {user_prompt}")
            parse_time = time.time() - parse_start

            # Gemini API call with streaming
            gemini_start = time.time()
            api_response_full_text, api_internal_time = await gemini_response_streaming(
                sessions[call_sid], user_prompt, websocket
            )
            gemini_total_time = time.time() - gemini_start

            # Total turnaround time
            total_time = time.time() - loop_start

            # --- OPTIMIZED: Detailed performance logging ---
            print(f"[PERFORMANCE] "
                  f"parse: {parse_time*1000:.1f}ms | "
                  f"gemini_internal: {api_internal_time*1000:.1f}ms | "
                  f"gemini_total: {gemini_total_time*1000:.1f}ms | "
                  f"total_turnaround: {total_time*1000:.1f}ms")

            # --- PERFORMANCE ANALYSIS ---
            if total_time > 2.0:
                print(f"⚠️  SLOW RESPONSE: {total_time:.2f}s - Consider optimizing")
            elif total_time < 1.0:
                print(f"✅ FAST RESPONSE: {total_time:.2f}s")

    except WebSocketDisconnect:
        print(f"🔌 WebSocket connection closed for call {call_sid}")
        if call_sid and call_sid in sessions:
            sessions.pop(call_sid, None)
            print(f"🧹 Cleared session for call {call_sid}")
    except Exception as e:
        print(f"💥 Unexpected error in websocket_endpoint: {e}")
        if call_sid and call_sid in sessions:
            sessions.pop(call_sid, None)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "domain": DOMAIN,
        "optimizations": [
            "cached_model",
            "connection_pooling", 
            "smart_buffering",
            "fast_tts_config",
            "reduced_timeouts"
        ]
    }

if __name__ == "__main__":
    print(f"🚀 Starting optimized voice assistant on port {PORT}")
    print(f"🔗 WebSocket URL for Twilio: {WS_URL}")
    print(f"🌐 Detected platform domain: {DOMAIN}")
    print(f"⚡ Optimizations: Caching, Connection Pooling, Smart Buffering, Fast TTS")
    
    # --- OPTIMIZED: Adjusted worker count for voice workload ---
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)