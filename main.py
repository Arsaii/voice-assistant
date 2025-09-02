import os
import json
import uvicorn
import time
import aiohttp
from groq import Groq
from fastapi import FastAPI, Request
from fastapi.responses import Response
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PORT = int(os.getenv("PORT", "8080"))

# Domain detection
DOMAIN = os.getenv("RAILWAY_STATIC_URL")
if not DOMAIN:
    DOMAIN = f"localhost:{PORT}"
if DOMAIN.startswith("https://"):
    DOMAIN = DOMAIN.replace("https://", "")

WELCOME_GREETING = "Hello, how can I help?"

# System prompt for voice assistant
SYSTEM_PROMPT = """You are a helpful voice assistant. Keep responses conversational and concise since this is a phone call. 
Rules:
1. Be direct and brief - aim for 1-2 sentences per response
2. Spell out numbers (say 'twenty-three' not '23')
3. No special characters, bullets, or formatting
4. Sound natural and friendly"""

# API Keys
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

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
CHAT_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
MAX_TOKENS = 100
TEMPERATURE = 0.7

# Global session for HTTP requests
http_session = None

async def create_http_session():
    """Create HTTP session for API calls"""
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=30,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=30,
    )
    return aiohttp.ClientSession(connector=connector)

# Store conversation sessions
sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global http_session
    http_session = await create_http_session()
    print("HTTP session initialized")
    
    yield
    
    if http_session:
        await http_session.close()
    print("HTTP session closed")

app = FastAPI(lifespan=lifespan)

async def generate_ai_response(chat_history, user_prompt):
    """Generate AI response using Groq"""
    start_time = time.time()
    
    try:
        # Add user message to history
        chat_history.append({"role": "user", "content": user_prompt})
        
        # Generate response with Groq
        completion = groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *chat_history
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        
        response_text = completion.choices[0].message.content
        
        # Add response to history
        chat_history.append({"role": "assistant", "content": response_text})
        
        # Keep history manageable
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        elapsed = time.time() - start_time
        print(f"AI response generated in {elapsed*1000:.0f}ms: '{response_text}'")
        
        return response_text, elapsed
        
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return "I encountered an error. Please try again.", 0

async def speak_to_call(call_sid, text):
    """Send text to be spoken on the call using Telnyx Voice API"""
    try:
        print(f"Speaking to call {call_sid}: '{text[:50]}...'")
        
        url = f"https://api.telnyx.com/v2/calls/{call_sid}/actions/speak"
        
        headers = {
            "Authorization": f"Bearer {TELNYX_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "payload": text,
            "voice": f"ElevenLabs.Default.{ELEVENLABS_VOICE_ID}",
            "voice_settings": {
                "api_key_ref": "el_api_key"
            }
        }
        
        async with http_session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                print(f"Speak request successful: {result}")
                return True
            else:
                error_text = await response.text()
                print(f"Speak request failed ({response.status}): {error_text}")
                return False
                
    except Exception as e:
        print(f"Error speaking to call: {e}")
        return False

@app.get("/texml")
@app.post("/texml")
async def texml_endpoint(request: Request):
    """Main TeXML endpoint for call handling"""
    try:
        print(f"TeXML endpoint called with {request.method}")
        
        # Parse request parameters
        if request.method == "GET":
            params = dict(request.query_params)
        else:
            try:
                form = await request.form()
                params = dict(form)
            except:
                params = {}
        
        print(f"TeXML params: {params}")
        
        # Generate TeXML response with shorter pause and transcription-only mode
        xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">{WELCOME_GREETING}</Say>
    <Start>
        <Transcription language="en" transcriptionCallback="/transcription" transcriptionEngine="B" />
    </Start>
    <Pause length="5"/>
    <Stop>
        <Transcription/>
    </Stop>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">Thanks for calling!</Say>
    <Hangup/>
</Response>"""
        
        print("TeXML response generated successfully")
        return Response(content=xml_response, media_type="text/xml")
        
    except Exception as e:
        print(f"TeXML endpoint error: {e}")
        
        fallback_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Sorry, there was an error setting up the call. Please try again.</Say>
</Response>"""
        return Response(content=fallback_xml, media_type="text/xml")

@app.post("/transcription")
async def transcription_endpoint(request: Request):
    """Handle transcription results from Telnyx"""
    try:
        print("Transcription endpoint called")
        
        # Parse form data
        form = await request.form()
        form_data = dict(form)
        print(f"Transcription data: {form_data}")
        
        # Extract transcript information
        transcript = form_data.get("Transcript", "").strip()
        is_final = form_data.get("IsFinal", "false").lower() == "true"
        confidence_str = form_data.get("Confidence", "")
        call_sid = form_data.get("CallSid", "")
        
        # Handle confidence score
        try:
            confidence = float(confidence_str) if confidence_str else 1.0
        except (ValueError, TypeError):
            confidence = 1.0
        
        print(f"Transcript: '{transcript}' (Final: {is_final}, Confidence: {confidence})")
        
        # Process final transcripts with content
        if is_final and transcript and len(transcript) > 1:
            print(f"Processing transcript: '{transcript}'")
            
            # Initialize or get session
            if call_sid not in sessions:
                sessions[call_sid] = {
                    "chat_history": [],
                    "created_at": time.time()
                }
                print(f"New session created for call: {call_sid}")
            
            session = sessions[call_sid]
            
            # Generate AI response
            ai_response, response_time = await generate_ai_response(
                session["chat_history"], transcript
            )
            
            print(f"AI response: '{ai_response}'")
            
            # Store the AI response in the session for the next TeXML call
            session["pending_response"] = ai_response
            
            print("AI response stored, waiting for next TeXML request")
        
        return Response(content="OK", media_type="text/plain")
        
    except Exception as e:
        print(f"Transcription endpoint error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return Response(content="ERROR", media_type="text/plain")

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "message": "Telnyx TeXML Voice Assistant",
        "endpoints": {
            "texml": "/texml (GET/POST) - Main call handling",
            "transcription": "/transcription (POST) - Speech processing",
            "health": "/health - Health check"
        },
        "features": [
            "TeXML Application with Transcription",
            "Groq LLM integration", 
            "ElevenLabs TTS",
            "Real-time conversation processing"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "domain": DOMAIN,
        "model": CHAT_MODEL,
        "approach": "TeXML + Transcription",
        "active_sessions": len(sessions)
    }

# Cleanup old sessions periodically
import asyncio
async def cleanup_sessions():
    """Remove old conversation sessions"""
    while True:
        try:
            current_time = time.time()
            old_sessions = [
                call_sid for call_sid, session in sessions.items()
                if current_time - session["created_at"] > 3600  # 1 hour
            ]
            
            for call_sid in old_sessions:
                del sessions[call_sid]
                print(f"Cleaned up session: {call_sid}")
                
        except Exception as e:
            print(f"Session cleanup error: {e}")
            
        await asyncio.sleep(300)  # Run every 5 minutes

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(cleanup_sessions())

if __name__ == "__main__":
    print(f"Starting Telnyx TeXML Voice Assistant")
    print(f"Domain: {DOMAIN}")
    print(f"Model: {CHAT_MODEL}")
    print(f"Voice: {ELEVENLABS_VOICE_ID}")
    
    print(f"Environment check:")
    print(f"  - GROQ_API_KEY: {'Set' if GROQ_API_KEY else 'NOT SET'}")
    print(f"  - EL_API_KEY: {'Set' if ELEVENLABS_API_KEY else 'NOT SET'}")
    print(f"  - TELNYX_API_KEY: {'Set' if TELNYX_API_KEY else 'NOT SET'}")
    
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)