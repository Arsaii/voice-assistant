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
        full_response_text = "I encountered an error. Please try again."

    api_elapsed = time.time() - start_api
    
    return full_response_text, api_elapsed

async def answer_call(call_control_id):
    try:
        url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/answer"
        
        headers = {
            "Authorization": f"Bearer {TELNYX_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {}
        
        async with http_session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                return False
            return True
                
    except Exception:
        return False

async def start_voice_api_transcription(call_control_id):
    try:
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
            if response.status != 200:
                error_text = await response.text()
                return False
            return True
                
    except Exception:
        return False

async def speak_response_to_call(call_control_id, response_text):
    try:
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
            if response.status != 200:
                error_text = await response.text()
                
    except Exception:
        pass

@app.post("/webhooks/voice")
async def voice_webhook(request: Request):
    try:
        webhook_data = await request.json()
        event_type = webhook_data.get("data", {}).get("event_type", "")
        payload = webhook_data.get("data", {}).get("payload", {})
        
        call_control_id = payload.get("call_control_id", "")
        
        if event_type == "call.initiated":
            await answer_call(call_control_id)
        elif event_type == "call.answered":
            await speak_response_to_call(call_control_id, WELCOME_GREETING)
            await start_voice_api_transcription(call_control_id)
        elif event_type == "call.transcription":
            await handle_transcription_event(payload)
        
        return Response(content="OK", media_type="text/plain")
        
    except Exception:
        return Response(content="ERROR", media_type="text/plain")

async def handle_transcription_event(payload):
    try:
        call_control_id = payload.get("call_control_id", "")
        transcription_data = payload.get("transcription_data", {})
        
        transcript = transcription_data.get("transcript", "").strip()
        is_final = transcription_data.get("is_final", False)
        
        if is_final and transcript and len(transcript) > 1:
            if call_control_id not in sessions:
                sessions[call_control_id] = {
                    "chat_history": []
                }
            
            response_text, _ = await groq_response_streaming(
                sessions[call_control_id]["chat_history"], transcript
            )
            
            await speak_response_to_call(call_control_id, response_text)
        
    except Exception:
        pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)