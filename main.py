import os
import json
import uvicorn
import asyncio
import time
import aiohttp
from groq import Groq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
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

ELEVENLABS_API_KEY = os.getenv("EL_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "FGY2WhTYpPnrIDTdsKH5")
if not ELEVENLABS_API_KEY:
    raise ValueError("EL_API_KEY environment variable not set.")

TELNYX_API_KEY = os.getenv("TELNYX_API_KEY")
if not TELNYX_API_KEY:
    raise ValueError("TELNYX_API_KEY environment variable not set.")

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

# FIXED: Accept both GET and POST for /texml endpoint
@app.get("/texml")
@app.post("/texml")
async def texml_endpoint(request: Request):
    """Return TeXML using Telnyx's built-in STT + ElevenLabs TTS"""
    try:
        print(f"🔗 TeXML endpoint called with {request.method}")
        print(f"🌐 Domain: {DOMAIN}")
        print(f"🤖 Model: {CHAT_MODEL}")
        
        # Log incoming parameters for debugging
        if request.method == "GET":
            params = dict(request.query_params)
            print(f"📥 GET params: {params}")
        else:
            try:
                form = await request.form()
                params = dict(form)
                print(f"📥 POST params: {params}")
            except:
                params = {}
        
        # Fixed TeXML - separate greeting from gathering, no nested Say in Gather
        xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">{WELCOME_GREETING}</Say>
    <Gather action="/texml-response" method="POST" timeout="15" finishOnKey="" speechTimeout="5"></Gather>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">I didn't hear anything. Goodbye!</Say>
    <Hangup/>
</Response>"""
        
        print(f"✅ TeXML response generated successfully")
        print(f"🚀 Using Telnyx Frankfurt STT + ElevenLabs TTS (Premium)")
        print(f"🎵 ElevenLabs Voice ID: {ELEVENLABS_VOICE_ID}")
        return Response(content=xml_response, media_type="text/xml")
        
    except Exception as e:
        print(f"💥 TeXML endpoint error: {e}")
        import traceback
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        
        fallback_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Sorry, there was an error setting up the call. Please try again.</Say>
</Response>"""
        return Response(content=fallback_xml, media_type="text/xml")

@app.post("/texml-response")
@app.get("/texml-response")  # Add GET method for debugging
async def texml_response_endpoint(request: Request):
    """Handle Telnyx STT results and generate AI response with comprehensive timing"""
    try:
        print(f"🎯 /texml-response endpoint called with {request.method}!")
        
        # TIMING: Mark when we received the user's prompt (after STT)
        prompt_received_time = time.time()
        
        # Enhanced form data parsing with debugging for both GET and POST
        if request.method == "GET":
            form_data = dict(request.query_params)
            print(f"📋 GET query params: {form_data}")
        else:
            form = await request.form()
            form_data = dict(form)
            print(f"📋 POST form data: {form_data}")
        
        # Try multiple possible field names for speech recognition
        transcript = (
            form_data.get("SpeechResult", "") or 
            form_data.get("UnstableSpeechResult", "") or
            form_data.get("speech_result", "") or
            form_data.get("Digits", "") or
            form_data.get("RecognitionResult", "") or
            form_data.get("Body", "")  # Sometimes used for speech results
        )
        
        call_control_id = (
            form_data.get("CallControlId", "") or
            form_data.get("CallSid", "") or
            form_data.get("call_control_id", "")
        )
        
        print(f"🎤 Extracted transcript: '{transcript}'")
        print(f"📞 Call Control ID: '{call_control_id}'")
        print(f"📝 All form keys: {list(form_data.keys())}")
        
        if not transcript:
            print(f"⚠️ No speech detected - trying again")
            # No speech detected, try again
            xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather action="/texml-response" method="POST" timeout="15">
        <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">I didn't catch that. Could you repeat?</Say>
    </Gather>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">I'm having trouble hearing you. Goodbye!</Say>
    <Hangup/>
</Response>"""
            return Response(content=xml_response, media_type="text/xml")
        
        # Initialize session if needed
        if call_control_id not in sessions:
            sessions[call_control_id] = {
                "chat_history": [],
                "timing": {
                    "setup_time": time.time(),
                    "last_response_complete": None,
                    "last_prompt_received": None
                }
            }
            print(f"🆕 New session created for call: {call_control_id}")
        
        # Calculate STT + VAD time
        stt_vad_time = 0
        if sessions[call_control_id]["timing"]["last_response_complete"]:
            stt_vad_time = prompt_received_time - sessions[call_control_id]["timing"]["last_response_complete"]
            print(f"🔊 Telnyx STT + VAD time: {stt_vad_time*1000:.0f}ms")
        else:
            stt_vad_time = prompt_received_time - sessions[call_control_id]["timing"]["setup_time"]
            print(f"🔊 Initial Telnyx STT + VAD time: {stt_vad_time*1000:.0f}ms")
        
        sessions[call_control_id]["timing"]["last_prompt_received"] = prompt_received_time
        
        print(f"🤖 Generating AI response for: '{transcript}'")
        
        # Generate AI response
        response_text, api_time = await groq_response_streaming(
            sessions[call_control_id]["chat_history"], transcript
        )
        
        # TIMING: Mark response completion
        response_complete_time = time.time()
        sessions[call_control_id]["timing"]["last_response_complete"] = response_complete_time
        
        # Calculate comprehensive timing breakdown
        total_server_time = response_complete_time - prompt_received_time
        network_overhead = total_server_time - api_time
        
        print(f"🤖 AI Response: '{response_text}'")
        print(f"📊 TELNYX TIMING BREAKDOWN:")
        print(f"  🔊 Telnyx STT + VAD: {stt_vad_time*1000:.0f}ms")
        print(f"  🎯 Server Total: {total_server_time*1000:.0f}ms")
        print(f"  🧠 AI Processing: {api_time*1000:.0f}ms")
        print(f"  📡 Network/Overhead: {network_overhead*1000:.0f}ms")
        print(f"  ⚡ Subtotal (visible): {(stt_vad_time + total_server_time)*1000:.0f}ms")
        print(f"  ⚠️ + ElevenLabs TTS time (~400-800ms) = ~{((stt_vad_time + total_server_time)*1000 + 600):.0f}ms total")
        print(f"  🇩🇪 Frankfurt routing active!")

        # Performance analysis
        total_visible_time = stt_vad_time + total_server_time
        estimated_total = total_visible_time + 0.6  # Add estimated TTS time
        
        if estimated_total > 2.5:
            print(f"🐌 SLOW OVERALL: {estimated_total:.1f}s")
        elif estimated_total < 1.5:
            print(f"🚀 FAST OVERALL: {estimated_total:.1f}s")
        else:
            print(f"✅ NORMAL OVERALL: {estimated_total:.1f}s")
        
        # Return TeXML with AI response using ElevenLabs
        xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">{response_text}</Say>
    <Gather action="/texml-response" method="POST" timeout="15">
        <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">Anything else I can help with?</Say>
    </Gather>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">Thanks for calling!</Say>
    <Hangup/>
</Response>"""
        
        print(f"✅ Returning AI response XML")
        return Response(content=xml_response, media_type="text/xml")
        
    except Exception as e:
        print(f"💥 TeXML response error: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        
        xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">Sorry, I had an error. Please try again.</Say>
    <Hangup/>
</Response>"""
        return Response(content=xml_response, media_type="text/xml")

@app.get("/")
async def root():
    """Simple root endpoint for testing"""
    return {
        "message": "Telnyx Voice Assistant API with Frankfurt routing",
        "endpoints": {
            "texml": "/texml (GET/POST)",
            "texml_response": "/texml-response (POST)",
            "health": "/health"
        },
        "features": [
            "Telnyx Frankfurt servers (low latency)",
            "Telnyx built-in STT",
            "Groq LLM processing", 
            "ElevenLabs TTS",
            "Comprehensive timing analysis"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "domain": DOMAIN,
        "model": CHAT_MODEL,
        "stt_provider": "Telnyx Frankfurt",
        "tts_provider": "ElevenLabs",
        "llm_provider": "Groq",
        "voice_id": ELEVENLABS_VOICE_ID,
        "optimizations": [
            "telnyx_frankfurt_routing", 
            "groq_llm",
            "elevenlabs_tts",
            "comprehensive_timing"
        ]
    }

if __name__ == "__main__":
    print(f"🚀 Starting Telnyx voice assistant with Frankfurt routing")
    print(f"🌐 Domain: {DOMAIN}")
    print(f"🤖 Groq Model: {CHAT_MODEL}")
    print(f"🎵 ElevenLabs Voice: {ELEVENLABS_VOICE_ID}")
    
    # Verify environment variables
    print(f"✅ Environment check:")
    print(f"  - GROQ_API_KEY: {'Set' if GROQ_API_KEY else 'NOT SET'}")
    print(f"  - EL_API_KEY: {'Set' if ELEVENLABS_API_KEY else 'NOT SET'}")
    print(f"  - TELNYX_API_KEY: {'Set' if TELNYX_API_KEY else 'NOT SET'}")
    print(f"  - GROQ_MODEL_NAME: {CHAT_MODEL}")
    print(f"  - ELEVENLABS_VOICE_ID: {ELEVENLABS_VOICE_ID}")
    
    print(f"🇩🇪 Clean Telnyx Pipeline:")
    print(f"  Your voice → Telnyx Frankfurt STT → Groq LLM → ElevenLabs TTS → Your ears")
    print(f"  Expected performance with Frankfurt routing: <2s total response time")
    
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)