import os
import json
import uvicorn
import asyncio
import time
import aiohttp

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

# --- Groq API Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant") # Default Groq Llama 3.1 8B

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

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

# --- OPTIMIZED: Generation config for faster responses (mapped to Groq params) ---
# Note: Groq's API might not support all specific parameters like top_k directly,
# but it supports temperature, top_p, and max_tokens.
generation_params = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 100, # Shorter responses for voice
}

# --- REMOVED: Safety settings (Groq doesn't expose these directly via API params,
# but their models have inherent safety training) ---

# --- NEW: Groq LLM Client Class ---
class GroqLLMClient:
    """
    Client to interact with Groq's LLaMA API.
    Handles chat history and streaming responses.
    """
    def __init__(self, model_name: str, system_prompt: str, http_session: aiohttp.ClientSession, api_key: str):
        self.model_name = model_name
        self.http_session = http_session
        self.api_key = api_key
        self.history = [{"role": "system", "content": system_prompt}]
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.api_endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_messages(self):
        return self.history

    async def stream_chat_completions(self, user_prompt: str, gen_params: dict):
        """
        Sends a chat completion request to the Groq API and streams the response.
        """
        # Add user's latest message to history for this turn
        self.add_message("user", user_prompt)
        
        payload = {
            "model": self.model_name,
            "messages": self.get_messages(),
            "stream": True,
            **gen_params # Unpack generation parameters
        }

        full_assistant_response = ""
        try:
            async with self.http_session.post(
                self.api_endpoint,
                headers=self.headers,
                json=payload,
                timeout=6.0 # Reduced timeout for faster failure detection
            ) as response:
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

                async for chunk_bytes in response.content.iter_any():
                    chunk_str = chunk_bytes.decode('utf-8')
                    for line in chunk_str.splitlines():
                        if line.startswith("data: "):
                            data = line[len("data: "):].strip()
                            if data == "[DONE]":
                                break # End of stream
                            try:
                                chunk_data = json.loads(data)
                                if chunk_data.get("choices"):
                                    delta = chunk_data["choices"][0].get("delta", {})
                                    text = delta.get("content", "")
                                    finish_reason = chunk_data["choices"][0].get("finish_reason")
                                    
                                    full_assistant_response += text # Build full response for history

                                    # Yield a simplified chunk object for consistency with original structure
                                    class LLMChunk:
                                        def __init__(self, text_content="", finish_reason_code=None):
                                            self.text = text_content
                                            # Groq's response structure is similar to OpenAI, no 'candidates' object but finish_reason in choices
                                            # We'll adapt it to match the previous structure for easier integration.
                                            self.candidates = [type('obj', (object,), {'finish_reason': finish_reason_code})()] if finish_reason_code else []

                                    yield LLMChunk(text, finish_reason)
                            except json.JSONDecodeError:
                                print(f"Warning: Could not decode JSON from chunk: {data}")
                                continue
        except aiohttp.ClientError as e:
            print(f"Groq API Client Error for {self.model_name}: {e}")
            raise
        except asyncio.TimeoutError:
            print(f"Groq API Timeout for {self.model_name}.")
            raise
        
        # Add assistant's full response to history after completion
        self.add_message("assistant", full_assistant_response)

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

# Store active chat sessions (now GroqLLMClient instances)
sessions: dict[str, GroqLLMClient] = {}

# Global Groq Client (initialized in lifespan)
groq_client_instance: GroqLLMClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler (replaces deprecated startup/shutdown events)"""
    # Startup
    global http_session, groq_client_instance
    http_session = await create_http_session()
    print("✅ HTTP session initialized")
    
    # Initialize the global Groq client instance
    # Each WebSocket session will get its own history via a new GroqLLMClient instance
    # that uses this shared http_session.
    # We create a placeholder client here that will be used to initialize per-call sessions
    # (or could be used if you wanted a single shared history across calls - not recommended for voice)
    
    # NOTE: The GroqLLMClient manages its own history.
    # So, we don't need a "global" groq_client_instance that holds history.
    # Instead, each call_sid will create its own GroqLLMClient in websocket_endpoint.
    # We still need the http_session to be global.
    
    yield
    
    # Shutdown
    if http_session:
        await http_session.close()
    print("✅ HTTP session closed")

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# RENAMED: gemini_response_streaming -> groq_response_streaming
async def groq_response_streaming(groq_chat_client: GroqLLMClient, user_prompt: str, websocket: WebSocket):
    """
    Streaming with intelligent buffering for better performance using Groq.
    Buffers chunks until we have meaningful content or sentence boundaries.
    """
    start_api = time.time()
    full_response_text = ""
    buffer = ""
    buffer_size = 20  # Minimum characters before sending
    
    try:
        response_stream = await asyncio.wait_for(
            groq_chat_client.stream_chat_completions(user_prompt, generation_params),
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
                # Groq's finish reasons are similar to OpenAI's: 'stop', 'length', 'tool_calls', 'content_filter'
                finish_reason = chunk.candidates[0].finish_reason
                
                # Groq/Llama models have built-in safety, but if a content filter is triggered:
                if finish_reason == 'content_filter':
                    error_message = "I'm sorry, I can't respond to that right now. Could you try rephrasing your question?"
                elif finish_reason == 'length':
                    # This means max_tokens was hit. The response is complete up to that point.
                    pass # Don't error out, just proceed to send remaining buffer and finish signal
                elif finish_reason == 'stop':
                    pass # Normal completion, proceed
                else: # Any other unexpected finish reason
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
        print("⚠️ Groq API timeout (6s)")
        error_message = "I'm having trouble right now. Could you try again?"
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": error_message,
            "last": True
        }))
        return error_message, time.time() - start_api
        
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
                # Initialize a new GroqLLMClient instance for each call_sid
                sessions[call_sid] = GroqLLMClient(
                    model_name=GROQ_MODEL_NAME,
                    system_prompt=SYSTEM_PROMPT,
                    http_session=http_session, # Use the shared http_session
                    api_key=GROQ_API_KEY
                )
                print(f"✅ Setup for call: {call_sid} using Groq Llama: {GROQ_MODEL_NAME}")
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
        "llm_provider": "Groq",
        "llm_model": GROQ_MODEL_NAME,
        "optimizations": [
            "connection_pooling", 
            "smart_buffering",
            "fast_tts_config",
            "reduced_timeouts",
            "aggressive_vad",
            "groq_inference" # New optimization
        ]
    }

if __name__ == "__main__":
    print(f"🚀 Starting voice assistant on port {PORT}")
    print(f"🔗 WebSocket URL: {WS_URL}")
    print(f"🌐 Domain: {DOMAIN}")
    print(f"Using Groq model: {GROQ_MODEL_NAME}")
    
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)
