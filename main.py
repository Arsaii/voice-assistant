import os
import json
import uvicorn
import asyncio
import time
import aiohttp
import google.generativeai as genai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from contextlib import asynccontextmanager
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

# --- ADDED: Safety settings to prevent false positives ---
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

# --- OPTIMIZED: Regular model with generation config and safety settings ---
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",  # More stable than 2.5-flash
    system_instruction=SYSTEM_PROMPT,
    generation_config=generation_config,
    safety_settings=safety_settings
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
    session = aiohttp.ClientSession(connector=connector)
    return session

# Store active chat sessions
sessions: dict[str, any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler (replaces deprecated startup/shutdown events)"""
    # Startup
    global http_session
    http_session = await create_http_session()
    print("✅ HTTP session initialized")
    
    yield
    
    # Shutdown
    if http_session:
        await http_session.close()
    print("✅ HTTP session closed")

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

async def gemini_response_streaming(chat_session, user_prompt, websocket):
    """
    OPTIMIZED: Streaming with intelligent buffering for better performance.
    Buffers chunks until we have meaningful content or sentence boundaries.
    """
    start_api = time.time()
    full_response_text = ""
    buffer = ""
    buffer_size = 20  # Minimum characters before sending
    
    try:
        # Send message to Gemini API
        response_stream = await asyncio.wait_for(
            chat_session.send_message_async(user_prompt, stream=True),
            timeout=6.0  # Reduced timeout for faster failure detection
        )
        
        first_chunk_sent = False
        first_chunk_time = None
        
        async for chunk in response_stream:
            if chunk.text:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    ttfb = first_chunk_time - start_api
                    print(f"📡 TTFB: {ttfb*1000:.0f}ms")
                
                full_response_text += chunk.text
                buffer += chunk.text
                
                # Send first chunk immediately for faster TTFB
                if not first_chunk_sent and len(buffer) >= 5:
                    await websocket.send_text(json.dumps({
                        "type": "text",
                        "token": buffer,
                        "last": False
                    }))
                    buffer = ""
                    first_chunk_sent = True
                    continue
                
                # Smart buffering based on content and length
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
                    
            elif chunk.candidates and chunk.candidates[0].finish_reason:
                # Handle safety blocks and other finish reasons
                finish_reason = chunk.candidates[0].finish_reason
                
                if finish_reason == 2:  # SAFETY
                    error_message = "I'm sorry, I can't respond to that right now. Could you try rephrasing your question?"
                elif finish_reason == 3:  # RECITATION
                    error_message = "I apologize, but I need to avoid that response. Could you ask something else?"
                elif finish_reason == 4:  # OTHER
                    error_message = "I'm having trouble with that request. Could you try again?"
                else:
                    error_message = "I encountered an issue. Please try again."
                
                await websocket.send_text(json.dumps({
                    "type": "text",
                    "token": error_message,
                    "last": True
                }))
                
                return error_message, time.time() - start_api

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
        print("⚠️ Gemini API timeout (6s)")
        error_message = "I'm having trouble right now. Could you try again?"
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": error_message,
            "last": True
        }))
        return error_message, time.time() - start_api
        
    except Exception as e:
        print(f"💥 Gemini API error: {e}")
        error_message = "I encountered an error. Please try again."
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": error_message,
            "last": True
        }))
        return error_message, time.time() - start_api

    api_elapsed = time.time() - start_api
    print(f"⚡ API total: {api_elapsed*1000:.0f}ms")
    
    return full_response_text, api_elapsed

@app.post("/twiml")
async def twiml_endpoint():
    """Return TwiML with shorter VAD settings for more responsive detection"""
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
      elevenlabsOptimizeStreamingLatency="5"
      elevenlabsRequestTimeoutMs="3000"
      vadSilenceMs="200"
      vadThreshold="0.2"
      vadMode="aggressive"
      vadDebounceMs="25"
      vadPreambleMs="100"
      vadPostambleMs="50"
      vadMinSpeechMs="150"
      vadMaxSpeechMs="10000"
    />
  </Connect>
</Response>"""
    
    return Response(content=xml_response, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint with focused performance logging"""
    await websocket.accept()
    call_sid = None

    try:
        while True:
            # Receive and parse message
            raw = await websocket.receive_text()
            message = json.loads(raw)

            # Print debug messages but don't process them
            if message.get("type") in ["info", "debug"]:
                if message.get("name") in ["roundTripDelayMs", "tokensPlayed"]:
                    print(f"[{call_sid}] [{message.get('name', 'DEBUG')}] {message.get('value', message)}")
                continue

            if message.get("type") == "setup":
                call_sid = message["callSid"]
                sessions[call_sid] = model.start_chat(history=[])
                print(f"✅ Setup for call: {call_sid}")
                continue

            if message.get("type") != "prompt" or not call_sid:
                continue

            # Process user prompt
            user_prompt = message["voicePrompt"]
            print(f"🎤 User: {user_prompt}")
            
            # Start timing the full turnaround
            turnaround_start = time.time()

            # Gemini API call with streaming
            api_response_full_text, api_time = await gemini_response_streaming(
                sessions[call_sid], user_prompt, websocket
            )

            # Calculate total turnaround time
            total_time = time.time() - turnaround_start
            
            # Only log performance summary
            print(f"🚀 Total: {total_time*1000:.0f}ms | API: {api_time*1000:.0f}ms")

            # Performance analysis
            if total_time > 2.0:
                print(f"⚠️ SLOW: {total_time:.1f}s")
            elif total_time < 0.8:
                print(f"⚡ FAST: {total_time:.1f}s")

    except WebSocketDisconnect:
        if call_sid and call_sid in sessions:
            sessions.pop(call_sid, None)
            print(f"🔌 Disconnected: {call_sid}")
        
    except Exception as e:
        print(f"💥 WebSocket error: {e}")
        if call_sid and call_sid in sessions:
            sessions.pop(call_sid, None)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "domain": DOMAIN,
        "optimizations": [
            "connection_pooling", 
            "smart_buffering",
            "fast_tts_config",
            "reduced_timeouts",
            "aggressive_vad"
        ]
    }

if __name__ == "__main__":
    print(f"🚀 Starting voice assistant on port {PORT}")
    print(f"🔗 WebSocket URL: {WS_URL}")
    print(f"🌐 Domain: {DOMAIN}")
    
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)