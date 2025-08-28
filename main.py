import os
import json
import uvicorn
import asyncio
import time
import aiohttp
from groq import Groq
from fastapi import FastAPI, Request
from fastapi.responses import Response
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT", "8080"))

WELCOME_GREETING = "Hello, how can I help?"

SYSTEM_PROMPT = """You are a helpful voice assistant. Keep responses conversational and concise since this is a phone call. 
Rules:
1. Be direct and brief - aim for 1-2 sentences per response
2. Spell out numbers (say 'twenty-three' not '23')
3. No special characters, bullets, or formatting
4. Sound natural and friendly"""

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

CHAT_MODEL = os.getenv("GROQ_MODEL_NAME")
MAX_TOKENS = 100
TEMPERATURE = 0.7

http_session = None

async def create_http_session():
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=30,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=30,
    )
    session = aiohttp.ClientSession(connector=connector)
    return session

sessions: dict[str, dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_session
    http_session = await create_http_session()
    yield
    if http_session:
        await http_session.close()

app = FastAPI(lifespan=lifespan)

async def groq_response_streaming(chat_history, user_prompt):
    start_api = time.time()
    full_response_text = ""
    
    try:
        chat_history.append({"role": "user", "content": user_prompt})
        
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
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response_text += chunk.choices[0].delta.content
        
        chat_history.append({"role": "assistant", "content": full_response_text})
        
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

    except Exception as e:
        print(f"💥 Groq API error: {e}")
        full_response_text = "I encountered an error. Please try again."

    api_elapsed = time.time() - start_api
    print(f"⚡ Groq API: {api_elapsed*1000:.0f}ms")
    
    return full_response_text, api_elapsed

async def answer_call(call_control_id):
    try:
        print(f"📞 Answering call {call_control_id}")
        url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/answer"
        
        headers = {
            "Authorization": f"Bearer {TELNYX_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {}
        
        async with http_session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Call answered: {result}")
                return True
            else:
                error_text = await response.text()
                print(f"❌ Failed to answer call ({response.status}): {error_text}")
                return False
                
    except Exception as e:
        print(f"💥 Error answering call: {e}")
        return False

async def start_voice_api_transcription(call_control_id):
    try:
        print(f"🎤 Starting transcription for call {call_control_id}")
        url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/transcription_start"
        
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
                print(f"✅ Transcription started: {result}")
                return True
            else:
                error_text = await response.text()
                print(f"❌ Failed to start transcription ({response.status}): {error_text}")
                return False
                
    except Exception as e:
        print(f"💥 Error starting transcription: {e}")
        return False

async def stop_voice_api_transcription(call_control_id):
    try:
        print(f"🛑 Stopping transcription for call {call_control_id}")
        url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/transcription_stop"
        
        headers = {
            "Authorization": f"Bearer {TELNYX_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {}
        
        async with http_session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Transcription stopped: {result}")
                return True
            else:
                error_text = await response.text()
                print(f"❌ Failed to stop transcription ({response.status}): {error_text}")
                return False
                
    except Exception as e:
        print(f"💥 Error stopping transcription: {e}")
        return False

async def speak_response_to_call(call_control_id, response_text):
    try:
        print(f"🗣️ Speaking to call {call_control_id}: {response_text[:50]}...")
        url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/speak"
        
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
        
        async with http_session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Speak successful: {result}")
            else:
                error_text = await response.text()
                print(f"❌ Speak failed ({response.status}): {error_text}")
                
    except Exception as e:
        print(f"💥 Error speaking to call: {e}")

@app.post("/webhooks/voice")
async def voice_webhook(request: Request):
    try:
        print("========== VOICE WEBHOOK RECEIVED ==========")
        webhook_data = await request.json()
        event_type = webhook_data.get("data", {}).get("event_type", "")
        payload = webhook_data.get("data", {}).get("payload", {})
        print(f"Received event: {event_type}, payload keys: {list(payload.keys())}")
        
        call_control_id = payload.get("call_control_id", "")
        
        if event_type == "call.initiated":
            print(f"Processing call.initiated for {call_control_id}")
            await answer_call(call_control_id)
        elif event_type == "call.answered":
            print(f"Processing call.answered for {call_control_id}")
            await speak_response_to_call(call_control_id, WELCOME_GREETING)
            await start_voice_api_transcription(call_control_id)
            # Update session state
            if call_control_id in sessions:
                sessions[call_control_id]["transcription_active"] = True
        elif event_type == "call.transcription":
            print(f"Processing call.transcription for {call_control_id}")
            await handle_transcription_event(payload)
        elif event_type == "call.speak.ended":
            print(f"Processing call.speak.ended for {call_control_id}")
            if call_control_id in sessions and not sessions[call_control_id].get("transcription_active", False):
                await start_voice_api_transcription(call_control_id)
                sessions[call_control_id]["transcription_active"] = True
        
        print("========== WEBHOOK PROCESSING COMPLETE ==========")
        return Response(content="OK", media_type="text/plain")
        
    except Exception as e:
        print(f"CRITICAL ERROR in voice webhook: {e}")
        return Response(content="ERROR", media_type="text/plain")

async def handle_transcription_event(payload):
    try:
        call_control_id = payload.get("call_control_id", "")
        transcription_data = payload.get("transcription_data", {})
        print(f"Full transcription data: {transcription_data}")
        
        transcript = transcription_data.get("transcript", "").strip()
        is_final = transcription_data.get("is_final", False)
        confidence = transcription_data.get("confidence", 1.0)
        print(f"🎤 Transcript: '{transcript}' (Final: {is_final}, Confidence: {confidence})")
        
        if is_final and transcript and len(transcript) > 1:
            print(f"🤖 Processing final transcript: '{transcript}'")
            if call_control_id not in sessions:
                sessions[call_control_id] = {
                    "chat_history": [],
                    "transcription_active": False  # Will be set to True after first start
                }
                print(f"🆕 New session for {call_control_id}")
            
            # Stop transcription before speaking AI response
            if sessions[call_control_id]["transcription_active"]:
                await stop_voice_api_transcription(call_control_id)
                sessions[call_control_id]["transcription_active"] = False
            
            response_text, api_time = await groq_response_streaming(
                sessions[call_control_id]["chat_history"], transcript
            )
            print(f"🤖 AI Response: '{response_text}'")
            
            await speak_response_to_call(call_control_id, response_text)
        
    except Exception as e:
        print(f"💥 Error handling transcription: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)