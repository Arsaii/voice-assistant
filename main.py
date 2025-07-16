import os
import json
import uvicorn
import asyncio
import time
import aiohttp
import base64
import websockets
from groq import Groq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PORT = int(os.getenv("PORT", "8080"))

# Robust domain detection: Railway or localhost fallback
DOMAIN = os.getenv("RAILWAY_STATIC_URL")
if not DOMAIN:
    DOMAIN = f"localhost:{PORT}"
if DOMAIN.startswith("https://"):
    DOMAIN = DOMAIN.replace("https://", "")

WS_URL = (
    f"wss://{DOMAIN}/ws"
    if "localhost" not in DOMAIN
    else f"ws://{DOMAIN}/ws"
)

MEDIA_WS_URL = (
    f"wss://{DOMAIN}/media-ws"
    if "localhost" not in DOMAIN
    else f"ws://{DOMAIN}/media-ws"
)

WELCOME_GREETING = "Hello, how can I help?"

# --- OPTIMIZED: Shorter, more focused system prompt for voice ---
SYSTEM_PROMPT = """You are a helpful voice assistant. Keep responses conversational and concise since this is a phone call. 
Rules:
1. Be direct and brief - aim for 1-2 sentences per response
2. Spell out numbers (say 'twenty-three' not '23')
3. No special characters, bullets, or formatting
4. Sound natural and friendly"""

# --- API Initializations ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY environment variable not set.")

ELEVENLABS_API_KEY = os.getenv("EL_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "FGY2WhTYpPnrIDTdsKH5")
if not ELEVENLABS_API_KEY:
    raise ValueError("EL_API_KEY environment variable not set.")

groq_client = Groq(api_key=GROQ_API_KEY)

# --- Chat configuration ---
CHAT_MODEL = os.getenv("GROQ_MODEL_NAME")
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

# Store active chat sessions with conversation history and timing data
sessions: dict[str, dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler"""
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

async def groq_response_streaming(chat_history, user_prompt):
    """Generate AI response using Groq"""
    start_api = time.time()
    full_response_text = ""
    
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
        
        # Collect the full response
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response_text += chunk.choices[0].delta.content
        
        # Add assistant response to chat history
        chat_history.append({"role": "assistant", "content": full_response_text})
        
        # Keep chat history manageable (last 10 messages)
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

    except Exception as e:
        print(f"💥 Groq API error: {e}")
        full_response_text = "I encountered an error. Please try again."

    api_elapsed = time.time() - start_api
    print(f"⚡ Groq API: {api_elapsed*1000:.0f}ms")
    
    return full_response_text, api_elapsed

async def elevenlabs_tts(text):
    """Convert text to speech using ElevenLabs"""
    start_tts = time.time()
    
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        data = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True
            },
            "optimize_streaming_latency": 4
        }
        
        async with http_session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                audio_data = await response.read()
                tts_elapsed = time.time() - start_tts
                print(f"🎵 ElevenLabs TTS: {tts_elapsed*1000:.0f}ms")
                return audio_data
            else:
                print(f"💥 ElevenLabs error: {response.status}")
                return None
                
    except Exception as e:
        print(f"💥 ElevenLabs TTS error: {e}")
        return None

async def connect_to_deepgram(call_sid):
    """Connect to Deepgram for real-time STT"""
    try:
        deepgram_url = f"wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&channels=1&model=nova-2&language=en&smart_format=true&interim_results=false&endpointing=200&vad_turnoff=100"
        
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}"
        }
        
        print(f"🎙️ Connecting to Deepgram for call {call_sid}")
        deepgram_ws = await websockets.connect(deepgram_url, extra_headers=headers)
        print(f"✅ Deepgram connected for call {call_sid}")
        
        return deepgram_ws
    except Exception as e:
        print(f"💥 Deepgram connection error: {e}")
        return None

async def process_deepgram_response(deepgram_ws, call_sid, twilio_ws):
    """Process Deepgram STT responses and generate TTS"""
    try:
        async for message in deepgram_ws:
            data = json.loads(message)
            
            if data.get("type") == "Results":
                transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                
                if transcript.strip():
                    print(f"🎙️ Deepgram STT: {transcript}")
                    
                    # Process with our AI
                    if call_sid in sessions:
                        prompt_received_time = time.time()
                        
                        # Calculate STT timing
                        stt_time = 0
                        if sessions[call_sid]["timing"]["last_response_complete"]:
                            stt_time = prompt_received_time - sessions[call_sid]["timing"]["last_response_complete"]
                        else:
                            stt_time = prompt_received_time - sessions[call_sid]["timing"]["setup_time"]
                            
                        print(f"🚀 Deepgram STT time: {stt_time*1000:.0f}ms")
                        
                        sessions[call_sid]["timing"]["last_prompt_received"] = prompt_received_time
                        
                        # Generate AI response
                        response_text, api_time = await groq_response_streaming(
                            sessions[call_sid]["chat_history"], transcript
                        )
                        
                        print(f"🤖 AI Response: {response_text}")
                        
                        # Convert to speech with ElevenLabs
                        audio_data = await elevenlabs_tts(response_text)
                        
                        if audio_data:
                            # Convert MP3 to base64 for Twilio
                            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                            
                            # Send audio to Twilio Media Stream
                            media_message = {
                                "event": "media",
                                "streamSid": sessions[call_sid].get("stream_sid"),
                                "media": {
                                    "payload": audio_b64
                                }
                            }
                            await twilio_ws.send_text(json.dumps(media_message))
                            print(f"🎵 Sent ElevenLabs audio to Twilio")
                        
                        # Mark response completion
                        response_complete_time = time.time()
                        sessions[call_sid]["timing"]["last_response_complete"] = response_complete_time
                        
                        # Calculate timing
                        total_server_time = response_complete_time - prompt_received_time
                        
                        print(f"📊 COMPLETE TIMING BREAKDOWN:")
                        print(f"  🎙️ Deepgram STT: {stt_time*1000:.0f}ms")
                        print(f"  🧠 Groq AI: {api_time*1000:.0f}ms")
                        print(f"  🎵 ElevenLabs TTS: included in total")
                        print(f"  🎯 Total Server: {total_server_time*1000:.0f}ms")
                        print(f"  🚀 Expected end-to-end: ~{total_server_time:.1f}s")
                        
    except Exception as e:
        print(f"💥 Deepgram processing error: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")

@app.post("/twiml")
async def twiml_endpoint():
    """Return TwiML for media streaming with Deepgram STT + ElevenLabs TTS"""
    try:
        print(f"🔗 TwiML endpoint called")
        print(f"🌐 Domain: {DOMAIN}")
        print(f"🔌 Media WebSocket URL: {MEDIA_WS_URL}")
        print(f"🤖 Model: {CHAT_MODEL}")
        
        xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>{WELCOME_GREETING}</Say>
    <Start>
        <Stream url="{MEDIA_WS_URL}" />
    </Start>
    <Pause length="30"/>
</Response>"""
        
        print(f"✅ TwiML response generated successfully")
        print(f"🚀 Using Deepgram STT + ElevenLabs TTS + Groq LLM")
        return Response(content=xml_response, media_type="text/xml")
        
    except Exception as e:
        print(f"💥 TwiML endpoint error: {e}")
        import traceback
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        
        fallback_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Sorry, there was an error setting up the call. Please try again.</Say>
</Response>"""
        return Response(content=fallback_xml, media_type="text/xml")

@app.websocket("/media-ws")
async def media_websocket_endpoint(websocket: WebSocket):
    """Handle Twilio Media Streams with Deepgram STT + ElevenLabs TTS"""
    print(f"🔌 Media WebSocket connection attempt")
    
    call_sid = None
    deepgram_ws = None
    
    try:
        await websocket.accept()
        print(f"✅ Media WebSocket connected successfully")

        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("event") == "start":
                    call_sid = data.get("start", {}).get("callSid")
                    stream_sid = data.get("start", {}).get("streamSid")
                    print(f"🎬 Media stream started for call: {call_sid}")
                    
                    # Initialize session
                    sessions[call_sid] = {
                        "chat_history": [],
                        "stream_sid": stream_sid,
                        "timing": {
                            "setup_time": time.time(),
                            "last_response_complete": None,
                            "last_prompt_received": None
                        }
                    }
                    
                    # Connect to Deepgram
                    deepgram_ws = await connect_to_deepgram(call_sid)
                    
                    if deepgram_ws:
                        # Start processing Deepgram responses
                        asyncio.create_task(process_deepgram_response(deepgram_ws, call_sid, websocket))
                
                elif data.get("event") == "media" and deepgram_ws:
                    # Forward audio to Deepgram
                    audio_data = base64.b64decode(data["media"]["payload"])
                    await deepgram_ws.send(audio_data)
                
                elif data.get("event") == "stop":
                    print(f"🛑 Media stream stopped for call: {call_sid}")
                    break
                    
            except WebSocketDisconnect:
                print(f"🔌 Media WebSocket disconnect detected")
                break
                
            except Exception as e:
                if "disconnect" in str(e).lower() or "receive" in str(e).lower():
                    print(f"🔌 Media connection closed during processing")
                    break
                else:
                    print(f"💥 Error in media processing: {e}")
                    continue

    except WebSocketDisconnect:
        print(f"🔌 Media WebSocket disconnected normally")
        
    except Exception as e:
        print(f"💥 Media WebSocket error: {e}")
        
    finally:
        if deepgram_ws:
            await deepgram_ws.close()
            print(f"🔌 Deepgram connection closed")
            
        if call_sid and call_sid in sessions:
            timing = sessions[call_sid]["timing"]
            if timing["last_response_complete"] and timing["setup_time"]:
                total_call_time = timing["last_response_complete"] - timing["setup_time"]
                print(f"📊 CALL SUMMARY for {call_sid}:")
                print(f"  🕒 Total call duration: {total_call_time:.1f}s")
            
            sessions.pop(call_sid, None)
            print(f"🔌 Disconnected: {call_sid}")

@app.get("/")
async def root():
    """Simple root endpoint for testing"""
    return {
        "message": "Voice Assistant API with Deepgram STT + ElevenLabs TTS + Groq LLM",
        "endpoints": {
            "twiml": "/twiml",
            "media_websocket": "/media-ws",
            "health": "/health"
        },
        "features": [
            "Deepgram real-time STT",
            "Groq LLM processing", 
            "ElevenLabs TTS",
            "Optimized for speed"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "domain": DOMAIN,
        "media_websocket_url": MEDIA_WS_URL,
        "model": CHAT_MODEL,
        "stt_provider": "Deepgram",
        "tts_provider": "ElevenLabs",
        "llm_provider": "Groq",
        "voice_id": ELEVENLABS_VOICE_ID,
        "optimizations": [
            "deepgram_stt", 
            "groq_llm",
            "elevenlabs_tts",
            "media_streaming",
            "comprehensive_timing"
        ]
    }

if __name__ == "__main__":
    print(f"🚀 Starting voice assistant with Deepgram STT + ElevenLabs TTS + Groq LLM")
    print(f"🔗 Media WebSocket URL: {MEDIA_WS_URL}")
    print(f"🌐 Domain: {DOMAIN}")
    print(f"🤖 Groq Model: {CHAT_MODEL}")
    print(f"🎵 ElevenLabs Voice: {ELEVENLABS_VOICE_ID}")
    
    # Verify environment variables
    print(f"✅ Environment check:")
    print(f"  - GROQ_API_KEY: {'Set' if GROQ_API_KEY else 'NOT SET'}")
    print(f"  - DEEPGRAM_API_KEY: {'Set' if DEEPGRAM_API_KEY else 'NOT SET'}")
    print(f"  - EL_API_KEY: {'Set' if ELEVENLABS_API_KEY else 'NOT SET'}")
    print(f"  - GROQ_MODEL_NAME: {CHAT_MODEL}")
    print(f"  - ELEVENLABS_VOICE_ID: {ELEVENLABS_VOICE_ID}")
    
    print(f"🚀 Complete Pipeline:")
    print(f"  Your voice → Deepgram STT → Groq LLM → ElevenLabs TTS → Your ears")
    print(f"  Expected performance: <1.5s total response time")
    
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)