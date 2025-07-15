import os
import json
import uvicorn
import asyncio
import time
import aiohttp
from groq import Groq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# --- REMOVED: Import Twilio TwiML classes (ConversationRelay not available in Python library) ---
# We'll generate the XML manually since ConversationRelay is newer than the Python library

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
    "Hallo, wie kann ich Ihnen helfen?"
)

# --- OPTIMIZED: German system prompt for voice ---
SYSTEM_PROMPT = """Du bist ein hilfsreicher Sprachassistent auf Deutsch. Halte Antworten gesprächig und prägnant, da dies ein Telefonanruf ist.
Regeln:
1. Sei direkt und kurz - strebe 1-2 Sätze pro Antwort an
2. Buchstabiere Zahlen aus (sage 'dreiundzwanzig' nicht '23')
3. Keine Sonderzeichen, Aufzählungspunkte oder Formatierungen
4. Klinge natürlich und freundlich
5. Antworte immer auf Deutsch"""

# --- Groq API Initialization ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

groq_client = Groq(api_key=GROQ_API_KEY)

# --- Chat configuration ---
CHAT_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.1-70b-versatile")  # Use environment variable with fallback
MAX_TOKENS = 100
TEMPERATURE = 0.7

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

# Store active chat sessions with conversation history
sessions: dict[str, list] = {}

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

async def groq_response_streaming(chat_history, user_prompt, websocket):
    """
    OPTIMIZED: Streaming with Groq Llama for ultra-fast response times.
    """
    start_api = time.time()
    full_response_text = ""
    buffer = ""
    buffer_size = 20  # Minimum characters before sending
    
    try:
        # Add user message to chat history
        chat_history.append({"role": "user", "content": user_prompt})
        
        # Create streaming completion with Groq
        stream = groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *chat_history
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
        )
        
        first_chunk_sent = False
        first_chunk_time = None
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    ttfb = first_chunk_time - start_api
                    print(f"📡 TTFB: {ttfb*1000:.0f}ms")
                
                full_response_text += content
                buffer += content
                
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
        
        # Add assistant response to chat history
        chat_history.append({"role": "assistant", "content": full_response_text})
        
        # Keep chat history manageable (last 10 messages)
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

    except Exception as e:
        print(f"💥 Groq API error: {e}")
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
    try:
        print(f"🔗 TwiML endpoint called")
        print(f"🌐 Domain: {DOMAIN}")
        print(f"🔌 WebSocket URL: {WS_URL}")
        print(f"🤖 Model: {CHAT_MODEL}")
        
        xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <ConversationRelay
      url="{WS_URL}"
      welcomeGreeting="{WELCOME_GREETING}"
      ttsProvider="ElevenLabs"
      voice="FGY2WhTYpPnrIDTdsKH5"
      language="de"
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
        
        print(f"✅ TwiML response generated successfully")
        return Response(content=xml_response, media_type="text/xml")
        
    except Exception as e:
        print(f"💥 TwiML endpoint error: {e}")
        # Return a simple fallback TwiML
        fallback_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Sorry, there was an error setting up the call. Please try again.</Say>
</Response>"""
        return Response(content=fallback_xml, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint with focused performance logging"""
    print(f"🔌 WebSocket connection attempt from {websocket.client}")
    
    try:
        await websocket.accept()
        print(f"✅ WebSocket connected successfully")
        call_sid = None

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
                sessions[call_sid] = []  # Initialize empty chat history
                print(f"✅ Setup for call: {call_sid}")
                continue

            if message.get("type") != "prompt" or not call_sid:
                continue

            # Process user prompt
            user_prompt = message["voicePrompt"]
            print(f"🎤 User: {user_prompt}")
            
            # Start timing the full turnaround
            turnaround_start = time.time()

            # Groq API call with streaming
            api_response_full_text, api_time = await groq_response_streaming(
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
        print(f"🔌 WebSocket disconnected normally")
        if call_sid and call_sid in sessions:
            sessions.pop(call_sid, None)
            print(f"🔌 Disconnected: {call_sid}")
        
    except Exception as e:
        print(f"💥 WebSocket error: {e}")
        if call_sid and call_sid in sessions:
            sessions.pop(call_sid, None)

@app.get("/")
async def root():
    """Simple root endpoint for testing"""
    return {
        "message": "Voice Assistant API is running",
        "endpoints": {
            "twiml": "/twiml",
            "websocket": "/ws",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "domain": DOMAIN,
        "model": CHAT_MODEL,
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
    print(f"🤖 Model: {CHAT_MODEL}")
    
    # Verify environment variables
    print(f"✅ Environment check:")
    print(f"  - GROQ_API_KEY: {'Set' if GROQ_API_KEY else 'NOT SET'}")
    print(f"  - GROQ_MODEL_NAME: {os.getenv('GROQ_MODEL_NAME', 'Not set (using default)')}")
    print(f"  - RAILWAY_STATIC_URL: {os.getenv('RAILWAY_STATIC_URL', 'Not set')}")
    print(f"  - NGROK_URL: {os.getenv('NGROK_URL', 'Not set')}")
    
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)