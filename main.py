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

async def start_voice_api_transcription(call_sid):
    """Start Voice API transcription for the call"""
    try:
        print(f"🎤 Starting Voice API transcription for call {call_sid}")
        
        url = f"https://api.telnyx.com/v2/calls/{call_sid}/actions/transcription_start"
        
        headers = {
            "Authorization": f"Bearer {TELNYX_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "language": "en",
            "transcription_engine": "B",
            "transcription_tracks": "inbound"
        }
        
        async with http_session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Voice API transcription started: {result}")
                return True
            else:
                error_text = await response.text()
                print(f"❌ Failed to start transcription ({response.status}): {error_text}")
                return False
                
    except Exception as e:
        print(f"💥 Error starting transcription: {e}")
        return False

async def speak_response_to_call(call_sid, response_text):
    """Send AI response back to the active call using Telnyx Voice API"""
    try:
        print(f"🗣️ Speaking response to call {call_sid}")
        
        # Telnyx Voice API endpoint for speak action
        url = f"https://api.telnyx.com/v2/calls/{call_sid}/actions/speak"
        
        headers = {
            "Authorization": f"Bearer {TELNYX_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "payload": response_text,
            "voice": f"ElevenLabs.Default.{ELEVENLABS_VOICE_ID}",
            "voice_settings": {
                "api_key_ref": "el_api_key"
            }
        }
        
        print(f"🔊 Sending speak request: {response_text[:50]}...")
        
        async with http_session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Speak request successful: {result}")
                print(f"🎧 Voice API transcription should continue listening...")
            else:
                error_text = await response.text()
                print(f"❌ Speak request failed ({response.status}): {error_text}")
                
    except Exception as e:
        print(f"💥 Error speaking to call: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")

# --- PURE VOICE API APPROACH ---

@app.get("/texml")
@app.post("/texml")
async def texml_endpoint(request: Request):
    """Simple TeXML that starts Voice API transcription"""
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
        
        call_sid = params.get("CallSid", "")
        
        # Simple TeXML that plays greeting and holds the call
        xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">{WELCOME_GREETING}</Say>
    <Pause length="60"/>
    <Say voice="ElevenLabs.Default.{ELEVENLABS_VOICE_ID}" api_key_ref="el_api_key">Thanks for calling!</Say>
    <Hangup/>
</Response>"""
        
        # Start Voice API transcription asynchronously
        if call_sid:
            asyncio.create_task(start_voice_api_transcription(call_sid))
        
        print(f"✅ TeXML response generated successfully")
        print(f"🚀 Using Pure Voice API approach")
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

@app.post("/webhooks/voice")
async def voice_webhook(request: Request):
    """Handle Voice API webhooks including transcription events"""
    try:
        print(f"========== VOICE WEBHOOK DEBUG ==========")
        print(f"Method: {request.method}")
        print(f"URL: {request.url}")
        print(f"Headers: {dict(request.headers)}")
        
        # Try to parse as JSON first, then fall back to form data
        webhook_data = {}
        content_type = request.headers.get("content-type", "")
        print(f"Content-Type: {content_type}")
        
        try:
            # Try JSON first
            webhook_data = await request.json()
            print(f"Successfully parsed as JSON")
            print(f"JSON webhook data: {json.dumps(webhook_data, indent=2)}")
        except Exception as json_error:
            print(f"JSON parsing failed: {json_error}")
            try:
                # Fall back to form data
                form = await request.form()
                webhook_data = dict(form)
                print(f"Successfully parsed as form data")
                print(f"Form webhook data: {webhook_data}")
            except Exception as form_error:
                print(f"Form parsing failed: {form_error}")
                try:
                    # Last resort - query params
                    webhook_data = dict(request.query_params)
                    print(f"Using query params as fallback")
                    print(f"Query webhook data: {webhook_data}")
                except Exception as query_error:
                    print(f"Query parsing failed: {query_error}")
        
        if not webhook_data:
            print("WARNING: No webhook data received at all")
            return Response(content="OK", media_type="text/plain")
        
        # Detailed webhook data analysis
        print(f"========== WEBHOOK DATA ANALYSIS ==========")
        print(f"Total fields received: {len(webhook_data)}")
        print(f"All field names: {list(webhook_data.keys())}")
        
        # Handle different webhook formats
        event_type = ""
        payload = {}
        
        # Check if it's the nested format (data.event_type)
        if "data" in webhook_data:
            print("Detected nested format with 'data' field")
            event_type = webhook_data.get("data", {}).get("event_type", "")
            payload = webhook_data.get("data", {}).get("payload", {})
            print(f"Nested event_type: {event_type}")
            print(f"Nested payload keys: {list(payload.keys()) if isinstance(payload, dict) else 'Not a dict'}")
        else:
            print("Detected flat format, checking for event_type variants")
            # Check multiple possible event type field names
            for field in ["event_type", "EventType", "Event", "event", "Type", "CallStatus"]:
                if field in webhook_data:
                    event_type = webhook_data.get(field, "")
                    print(f"Found event type in field '{field}': {event_type}")
                    break
            payload = webhook_data
        
        print(f"Final event_type: '{event_type}'")
        print(f"========== EVENT PROCESSING ==========")
        
        if event_type == "call.transcription":
            print("Processing call.transcription event")
            await handle_transcription_event(payload)
        elif event_type == "call.answered":
            print("Processing call.answered event")
            call_control_id = payload.get("call_control_id", payload.get("CallControlId", ""))
            if call_control_id:
                print(f"Call answered, starting transcription for {call_control_id}")
                await start_voice_api_transcription(call_control_id)
            else:
                print("No call_control_id found in call.answered event")
        elif "transcription" in str(webhook_data).lower():
            print("Detected transcription-related data via keyword search")
            await handle_transcription_fallback(webhook_data)
        else:
            print(f"Unhandled event type: '{event_type}'")
            print(f"Searching for transcription keywords in data...")
            
            # Look for any transcription-related keywords
            data_str = str(webhook_data).lower()
            transcription_keywords = ["transcript", "speech", "recognition", "audio", "voice"]
            found_keywords = [kw for kw in transcription_keywords if kw in data_str]
            
            if found_keywords:
                print(f"Found transcription keywords: {found_keywords}")
                await handle_transcription_fallback(webhook_data)
            else:
                print("No transcription keywords found")
                print("This might be a call status or other non-transcription event")
        
        print(f"========== WEBHOOK PROCESSING COMPLETE ==========")
        return Response(content="OK", media_type="text/plain")
        
    except Exception as e:
        print(f"CRITICAL ERROR in voice webhook: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return Response(content="ERROR", media_type="text/plain")

async def handle_transcription_event(payload):
    """Handle transcription events from Voice API"""
    try:
        call_control_id = payload.get("call_control_id", "")
        transcription_data = payload.get("transcription_data", {})
        
        transcript = transcription_data.get("transcript", "").strip()
        is_final = transcription_data.get("is_final", False)
        confidence = transcription_data.get("confidence", 1.0)
        
        print(f"🎤 Transcript: '{transcript}' (Final: {is_final}, Confidence: {confidence})")
        
        # Only process final transcriptions with content
        if is_final and transcript and len(transcript) > 1:
            print(f"🤖 Processing final transcript: '{transcript}'")
            
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
            
            # Generate AI response
            response_text, api_time = await groq_response_streaming(
                sessions[call_control_id]["chat_history"], transcript
            )
            
            print(f"🤖 AI Response: '{response_text}'")
            
            # Send the AI response back to the call
            await speak_response_to_call(call_control_id, response_text)
        
    except Exception as e:
        print(f"💥 Error handling transcription event: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")

async def handle_transcription_fallback(webhook_data):
    """Fallback handler for transcription data in various formats"""
    try:
        print(f"🔄 Attempting fallback transcription parsing...")
        
        # Try to extract transcript from various possible field names
        transcript = ""
        is_final = False
        confidence = 1.0
        call_control_id = ""
        
        # Check multiple possible transcript field names
        for field in ["transcript", "Transcript", "transcription_data", "text", "speech_result"]:
            if field in webhook_data and webhook_data[field]:
                transcript = str(webhook_data[field]).strip()
                break
        
        # Check for final status
        for field in ["is_final", "IsFinal", "final", "Final"]:
            if field in webhook_data:
                is_final = str(webhook_data[field]).lower() in ["true", "1", "yes"]
                break
        
        # Check for call ID
        for field in ["call_control_id", "CallControlId", "call_id", "CallId"]:
            if field in webhook_data and webhook_data[field]:
                call_control_id = webhook_data[field]
                break
        
        print(f"🎤 Fallback transcript: '{transcript}' (Final: {is_final})")
        
        if is_final and transcript and len(transcript) > 1:
            print(f"🤖 Processing fallback transcript: '{transcript}'")
            
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
            
            # Generate AI response
            response_text, api_time = await groq_response_streaming(
                sessions[call_control_id]["chat_history"], transcript
            )
            
            print(f"🤖 AI Response: '{response_text}'")
            
            # Send the AI response back to the call
            await speak_response_to_call(call_control_id, response_text)
        
    except Exception as e:
        print(f"💥 Error in transcription fallback: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")

@app.get("/")
async def root():
    """Simple root endpoint for testing"""
    return {
        "message": "Telnyx Voice Assistant API - Pure Voice API Approach",
        "endpoints": {
            "texml": "/texml (GET/POST)",
            "voice_webhook": "/webhooks/voice (POST)",
            "health": "/health"
        },
        "features": [
            "Pure Telnyx Voice API approach",
            "Voice API transcription", 
            "Groq LLM processing",
            "ElevenLabs TTS",
            "Continuous conversation support"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "domain": DOMAIN,
        "model": CHAT_MODEL,
        "approach": "Pure Voice API",
        "stt_provider": "Telnyx Voice API Engine B",
        "tts_provider": "ElevenLabs",
        "llm_provider": "Groq",
        "voice_id": ELEVENLABS_VOICE_ID
    }

if __name__ == "__main__":
    print(f"🚀 Starting Telnyx voice assistant - Pure Voice API approach")
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
    
    print(f"🔄 Pure Voice API Pipeline:")
    print(f"  Call → Simple TeXML → Voice API Transcription → AI Response → Voice API Speak")
    print(f"  Expected: Better conversation continuity")
    
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)