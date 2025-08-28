import os
import json
import uvicorn
import asyncio
import time
import aiohttp
from groq import Groq
from fastapi import FastAPI, Request, Form
from fastapi.responses import Response
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from urllib.parse import urlencode

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

# Base URL for TeXML callbacks
BASE_URL = f"https://{DOMAIN}" if "localhost" not in DOMAIN else f"http://{DOMAIN}"

WELCOME_GREETING = "Hello, how can I help you today?"

# --- OPTIMIZED: Shorter, more focused system prompt for voice ---
SYSTEM_PROMPT = """You are a helpful voice assistant. Keep responses conversational and concise since this is a phone call. 
Rules:
1. Be direct and brief - aim for 1-2 sentences per response
2. Spell out numbers (say 'twenty-three' not '23')
3. No special characters, bullets, or formatting
4. Sound natural and friendly
5. If you don't understand, ask the caller to repeat their question"""

# --- API Initializations ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "FGY2WhTYpPnrIDTdsKH5")

groq_client = Groq(api_key=GROQ_API_KEY)

# --- Chat configuration ---
CHAT_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.1-70b-versatile")
MAX_TOKENS = 100
TEMPERATURE = 0.7

# --- HTTP session for connection pooling ---
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

def get_session(call_sid: str) -> dict:
    """Get or create session for call"""
    if call_sid not in sessions:
        sessions[call_sid] = {
            "chat_history": [],
            "turn_count": 0,
            "created_at": time.time()
        }
        print(f"🆕 New session created for call: {call_sid}")
    return sessions[call_sid]

@app.get("/texml")
@app.post("/texml")
async def texml_initial(request: Request):
    """Initial TeXML endpoint that starts the conversation"""
    try:
        print(f"🔗 Initial TeXML endpoint called with {request.method}")
        
        # Get call parameters
        if request.method == "GET":
            params = dict(request.query_params)
        else:
            form = await request.form()
            params = dict(form)
        
        call_sid = params.get("CallSid", "unknown")
        print(f"📞 Starting call: {call_sid}")
        
        # Initialize session
        get_session(call_sid)
        
        # Create initial TeXML with greeting and first gather
        xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">{WELCOME_GREETING}</Say>
    <Gather input="speech" action="{BASE_URL}/process-speech" method="POST" speechTimeout="5" speechModel="phone_call">
    </Gather>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">I didn't hear anything. Goodbye!</Say>
    <Hangup/>
</Response>"""
        
        print(f"✅ Initial TeXML response generated")
        return Response(content=xml_response, media_type="text/xml")
        
    except Exception as e:
        print(f"💥 Initial TeXML error: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        
        fallback_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">Sorry, there was an error. Please try again.</Say>
    <Hangup/>
</Response>"""
        return Response(content=fallback_xml, media_type="text/xml")

@app.post("/process-speech")
async def process_speech(request: Request):
    """Process speech input from Gather and return AI response"""
    try:
        print(f"🎤 Processing speech input")
        
        # Parse form data
        form = await request.form()
        params = dict(form)
        
        print(f"📥 Speech processing params: {params}")
        
        call_sid = params.get("CallSid", "unknown")
        speech_result = params.get("SpeechResult", "").strip()
        confidence = params.get("Confidence", "1.0")
        
        print(f"🎤 Speech result: '{speech_result}' (Confidence: {confidence})")
        print(f"📞 Call SID: {call_sid}")
        
        # Get session
        session = get_session(call_sid)
        session["turn_count"] += 1
        
        print(f"📊 Session turn count: {session['turn_count']}")
        
        # Check if we got any speech input
        if not speech_result:
            print(f"❌ No speech result received")
            xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">I didn't hear anything. Could you please try again?</Say>
    <Gather input="speech" action="{BASE_URL}/process-speech" method="POST" speechTimeout="5" speechModel="phone_call">
    </Gather>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">Thanks for calling. Goodbye!</Say>
    <Hangup/>
</Response>"""
            return Response(content=xml_response, media_type="text/xml")
        
        if len(speech_result) < 2:
            print(f"❌ Speech too short: '{speech_result}'")
            xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">I didn't catch that clearly. Could you please repeat your question?</Say>
    <Gather input="speech" action="{BASE_URL}/process-speech" method="POST" speechTimeout="5" speechModel="phone_call">
    </Gather>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">Thanks for calling. Goodbye!</Say>
    <Hangup/>
</Response>"""
            return Response(content=xml_response, media_type="text/xml")
        
        # Generate AI response
        print(f"🤖 Generating AI response for: '{speech_result}'")
        print(f"📚 Current chat history length: {len(session['chat_history'])}")
        
        ai_response, api_time = await groq_response_streaming(
            session["chat_history"], speech_result
        )
        
        print(f"🤖 AI Response generated: '{ai_response}' (took {api_time*1000:.0f}ms)")
        
        # Determine if we should continue the conversation
        max_turns = int(os.getenv("MAX_CONVERSATION_TURNS", "10"))
        should_continue = session["turn_count"] < max_turns
        
        print(f"🔄 Should continue? {should_continue} (turn {session['turn_count']}/{max_turns})")
        
        # Check if response indicates end of conversation
        end_phrases = ["goodbye", "bye", "thanks for calling", "have a great day", "talk to you later"]
        ai_response_lower = ai_response.lower()
        contains_end_phrase = any(phrase in ai_response_lower for phrase in end_phrases)
        
        if contains_end_phrase:
            print(f"🛑 End phrase detected in AI response")
            should_continue = False
        
        if should_continue:
            print(f"✅ Continuing conversation...")
            # Continue conversation with another gather
            xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">{ai_response}</Say>
    <Gather input="speech" action="{BASE_URL}/process-speech" method="POST" speechTimeout="5" speechModel="phone_call">
    </Gather>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">Thanks for calling. Have a great day!</Say>
    <Hangup/>
</Response>"""
        else:
            print(f"🛑 Ending conversation...")
            # End conversation
            xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">{ai_response}</Say>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">Thanks for calling. Have a great day!</Say>
    <Hangup/>
</Response>"""
        
        print(f"📤 Returning TeXML response (length: {len(xml_response)} chars)")
        print(f"🔍 XML Preview: {xml_response[:200]}...")
        return Response(content=xml_response, media_type="text/xml")
        
    except Exception as e:
        print(f"💥 Speech processing error: {e}")
        import traceback
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        
        # Error fallback with more specific messaging
        xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">I'm sorry, I encountered a technical error. Let me try again.</Say>
    <Gather input="speech" action="{BASE_URL}/process-speech" method="POST" speechTimeout="5" speechModel="phone_call">
    </Gather>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">I'm having technical difficulties. Please try calling again later.</Say>
    <Hangup/>
</Response>"""
        return Response(content=xml_response, media_type="text/xml")



@app.get("/")
async def root():
    """Simple root endpoint for testing"""
    return {
        "message": "Telnyx Voice Assistant API - TeXML Only",
        "endpoints": {
            "texml": "/texml (GET/POST) - Initial call handler",
            "process_speech": "/process-speech (POST) - Speech processing",
            "health": "/health",
            "sessions": "/sessions (GET) - Debug endpoint"
        },
        "features": [
            "TeXML-only approach",
            "Gather with speech input", 
            "Groq LLM processing",
            "ElevenLabs TTS",
            "Multi-turn conversation support"
        ]
    }

@app.get("/health")
async def health_check():
    active_sessions = len(sessions)
    return {
        "status": "healthy", 
        "domain": DOMAIN,
        "base_url": BASE_URL,
        "model": CHAT_MODEL,
        "approach": "TeXML Only",
        "stt_provider": "Telnyx Gather Speech",
        "tts_provider": "ElevenLabs",
        "llm_provider": "Groq",
        "voice_id": ELEVENLABS_VOICE_ID,
        "active_sessions": active_sessions,
        "max_turns_per_call": os.getenv("MAX_CONVERSATION_TURNS", "10")
    }

@app.get("/sessions")
async def get_sessions():
    """Debug endpoint to view active sessions"""
    return {
        "active_sessions": len(sessions),
        "sessions": {
            call_sid: {
                "turn_count": session["turn_count"],
                "chat_history_length": len(session["chat_history"]),
                "created_at": session["created_at"]
            }
            for call_sid, session in sessions.items()
        }
    }

if __name__ == "__main__":
    print(f"🚀 Starting Telnyx voice assistant - TeXML Only approach")
    print(f"🌐 Domain: {DOMAIN}")
    print(f"🔗 Base URL: {BASE_URL}")
    print(f"🤖 Groq Model: {CHAT_MODEL}")
    print(f"🎵 ElevenLabs Voice: {ELEVENLABS_VOICE_ID}")
    
    # Verify environment variables
    print(f"✅ Environment check:")
    print(f"  - GROQ_API_KEY: {'Set' if GROQ_API_KEY else 'NOT SET'}")
    print(f"  - GROQ_MODEL_NAME: {CHAT_MODEL}")
    print(f"  - ELEVENLABS_VOICE_ID: {ELEVENLABS_VOICE_ID}")
    print(f"  - MAX_CONVERSATION_TURNS: {os.getenv('MAX_CONVERSATION_TURNS', '10')}")
    
    print(f"🔄 TeXML-Only Pipeline:")
    print(f"  Call → TeXML Greeting → Gather Speech → AI Response → Continue/End")
    print(f"  Benefits: Simpler architecture, reliable turn-based conversation")
    
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)