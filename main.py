import os
import json
import uvicorn
import asyncio
import time
from groq import Groq
from fastapi import FastAPI, Request, Form
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

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "FGY2WhTYpPnrIDTdsKH5")
ELEVENLABS_API_KEY_REF = "el_api_key"  # Must match your TTS credential name in Telnyx portal

groq_client = Groq(api_key=GROQ_API_KEY)

CHAT_MODEL = os.getenv("GROQ_MODEL_NAME")
MAX_TOKENS = 100
TEMPERATURE = 0.7

sessions: dict[str, dict] = {}

app = FastAPI()

def groq_response_sync(chat_history, user_prompt):
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
    
    return full_response_text

def generate_texml_say_gather(text: str):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say engine="elevenlabs" apiKeyRef="{ELEVENLABS_API_KEY_REF}" voice="{ELEVENLABS_VOICE_ID}">{text}</Say>
    <Gather input="speech" action="/webhooks/voice" method="POST" speechTimeout="auto" />
</Response>"""

@app.post("/webhooks/voice")
async def voice_webhook(request: Request):
    try:
        form_data = await request.form()
        print("========== TEXML WEBHOOK RECEIVED ==========")
        print(f"Form data: {dict(form_data)}")
        
        call_sid = form_data.get('CallSid', "")
        
        if call_sid not in sessions:
            sessions[call_sid] = {
                "chat_history": []
            }
            print(f"🆕 New session for {call_sid}")
        
        if 'SpeechResult' in form_data:
            transcript = form_data['SpeechResult'].strip()
            confidence = float(form_data.get('Confidence', 1.0))
            print(f"🎤 Transcript: '{transcript}' (Confidence: {confidence})")
            
            if transcript and len(transcript) > 1:
                response_text = groq_response_sync(
                    sessions[call_sid]["chat_history"], transcript
                )
                print(f"🤖 AI Response: '{response_text}'")
                
                xml = generate_texml_say_gather(response_text)
                print("========== WEBHOOK PROCESSING COMPLETE ==========")
                return Response(content=xml, media_type="application/xml")
        
        # Initial call (no SpeechResult)
        xml = generate_texml_say_gather(WELCOME_GREETING)
        print("========== WEBHOOK PROCESSING COMPLETE ==========")
        return Response(content=xml, media_type="application/xml")
        
    except Exception as e:
        print(f"CRITICAL ERROR in voice webhook: {e}")
        error_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Sorry, an error occurred. Goodbye.</Say>
    <Hangup />
</Response>"""
        return Response(content=error_xml, media_type="application/xml")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)