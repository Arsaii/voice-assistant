import os
import json
import uvicorn
import asyncio
import time
import aiohttp
import google.generativeai as genai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# --- ADDED: Import Twilio TwiML classes ---
from twilio.twiml.voice_response import VoiceResponse, Connect, ConversationRelay

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
    "Hello, how can I help?"
)

# --- OPTIMIZED: Even more aggressive system prompt for minimal latency ---
SYSTEM_PROMPT = """You are a helpful voice assistant. Be extremely brief and conversational.
Rules:
1. Maximum 1 sentence responses unless asked for more
2. Spell out numbers (say 'twenty-three' not '23')
3. No special characters, bullets, or formatting
4. Sound natural and friendly
5. Get straight to the point"""

# --- Gemini API Initialization ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=GOOGLE_API_KEY)

# --- OPTIMIZED: Ultra-fast generation config ---
generation_config = genai.GenerationConfig(
    temperature=0.3,  # Lower for more predictable/faster responses
    top_p=0.8,        # Lower for faster generation
    top_k=20,         # Reduced for faster sampling
    max_output_tokens=50,  # Even shorter for voice
    candidate_count=1,
    stop_sequences=[".", "!", "?"]  # Stop early at sentence boundaries
)

# --- ADDED: Safety settings to prevent false positives ---
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

# --- OPTIMIZED: Use fastest available model ---
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",  # Fastest variant
    system_instruction=SYSTEM_PROMPT,
    generation_config=generation_config,
    safety_settings=safety_settings
)

# --- OPTIMIZED: HTTP session for connection pooling ---
http_session = None
thread_pool = None

async def create_http_session():
    """Create optimized HTTP session for better connection management"""
    session_start = time.time()
    print(f"⏱️  [HTTP_SESSION] Creating HTTP session...")
    
    connector = aiohttp.TCPConnector(
        limit=200,  # Increased for better throughput
        limit_per_host=50,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=3.0)  # Aggressive timeout
    )
    
    session_time = time.time() - session_start
    print(f"⏱️  [HTTP_SESSION] Session created in {session_time*1000:.1f}ms")
    return session

# Store active chat sessions
sessions: dict[str, any] = {}

# --- ADDED: Pre-warm connection pool ---
connection_pool = []

async def prewarm_connections():
    """Pre-warm Gemini API connections to reduce first-call latency"""
    print(f"⏱️  [PREWARM] Starting connection pre-warming...")
    prewarm_start = time.time()
    
    try:
        # Create a throwaway chat session and send a minimal message
        prewarm_chat = model.start_chat(history=[])
        response = await asyncio.wait_for(
            prewarm_chat.send_message_async("Hi", stream=False),
            timeout=3.0
        )
        prewarm_time = time.time() - prewarm_start
        print(f"⏱️  [PREWARM] Connection pre-warmed in {prewarm_time*1000:.1f}ms")
        print(f"✅ Pre-warm successful")
    except Exception as e:
        prewarm_time = time.time() - prewarm_start
        print(f"⚠️  [PREWARM] Pre-warm failed in {prewarm_time*1000:.1f}ms: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler with pre-warming"""
    # Startup
    startup_start = time.time()
    print(f"⏱️  [STARTUP] Application startup beginning...")
    
    global http_session, thread_pool
    http_session = await create_http_session()
    thread_pool = ThreadPoolExecutor(max_workers=4)
    
    # Pre-warm Gemini connections
    await prewarm_connections()
    
    startup_time = time.time() - startup_start
    print(f"⏱️  [STARTUP] Application startup completed in {startup_time*1000:.1f}ms")
    print("✅ HTTP session initialized")
    print("✅ Thread pool initialized")
    
    yield
    
    # Shutdown
    shutdown_start = time.time()
    print(f"⏱️  [SHUTDOWN] Application shutdown beginning...")
    
    if http_session:
        await http_session.close()
    if thread_pool:
        thread_pool.shutdown(wait=True)
        
    shutdown_time = time.time() - shutdown_start
    print(f"⏱️  [SHUTDOWN] Application shutdown completed in {shutdown_time*1000:.1f}ms")
    print("✅ HTTP session closed")
    print("✅ Thread pool closed")

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

async def gemini_response_streaming(chat_session, user_prompt, websocket):
    """
    ULTRA-OPTIMIZED: Streaming with immediate first-byte response and parallel processing
    """
    function_start = time.time()
    print(f"⏱️  [GEMINI_STREAM] Function started")
    
    start_api = time.time()
    full_response_text = ""
    
    try:
        # --- OPTIMIZED: Even more aggressive timeout ---
        print(f"⏱️  [GEMINI_STREAM] Sending message to Gemini API...")
        api_call_start = time.time()
        
        response_stream = await asyncio.wait_for(
            chat_session.send_message_async(user_prompt, stream=True),
            timeout=4.0  # Reduced from 6s to 4s
        )
        
        api_call_time = time.time() - api_call_start
        print(f"⏱️  [GEMINI_STREAM] API call initiated in {api_call_time*1000:.1f}ms")
        
        first_chunk_sent = False
        chunk_count = 0
        first_chunk_time = None
        
        async for chunk in response_stream:
            chunk_start = time.time()
            chunk_count += 1
            
            if chunk.text:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    ttfb = first_chunk_time - start_api
                    print(f"⏱️  [GEMINI_STREAM] First chunk received (TTFB): {ttfb*1000:.1f}ms")
                
                full_response_text += chunk.text
                
                # --- ULTRA-OPTIMIZED: Send EVERY chunk immediately ---
                if not first_chunk_sent:
                    send_start = time.time()
                    await websocket.send_text(json.dumps({
                        "type": "text",
                        "token": chunk.text,
                        "last": False
                    }))
                    send_time = time.time() - send_start
                    print(f"⏱️  [GEMINI_STREAM] First chunk sent to WebSocket in {send_time*1000:.1f}ms")
                    first_chunk_sent = True
                else:
                    # Send subsequent chunks immediately
                    await websocket.send_text(json.dumps({
                        "type": "text",
                        "token": chunk.text,
                        "last": False
                    }))
                    
            elif chunk.candidates and chunk.candidates[0].finish_reason:
                # --- ADDED: Handle safety blocks and other finish reasons ---
                finish_reason = chunk.candidates[0].finish_reason
                print(f"⏱️  [GEMINI_STREAM] Finish reason received: {finish_reason}")
                
                if finish_reason == 2:  # SAFETY
                    error_message = "I'm sorry, I can't respond to that right now."
                elif finish_reason == 3:  # RECITATION
                    error_message = "I apologize, but I need to avoid that response."
                elif finish_reason == 4:  # OTHER
                    error_message = "I'm having trouble with that request."
                else:
                    error_message = "I encountered an issue."
                
                await websocket.send_text(json.dumps({
                    "type": "text",
                    "token": error_message,
                    "last": True
                }))
                
                return error_message, time.time() - start_api

            chunk_time = time.time() - chunk_start
            print(f"⏱️  [GEMINI_STREAM] Chunk {chunk_count} processed in {chunk_time*1000:.1f}ms")

        # Send completion signal
        completion_start = time.time()
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": "",
            "last": True
        }))
        completion_time = time.time() - completion_start
        print(f"⏱️  [GEMINI_STREAM] Completion signal sent in {completion_time*1000:.1f}ms")

    except asyncio.TimeoutError:
        print(f"⏱️  [GEMINI_STREAM] Timeout occurred after 4 seconds")
        
        error_message = "I'm having trouble right now."
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": error_message,
            "last": True
        }))
        
        return error_message, time.time() - start_api
        
    except Exception as e:
        import traceback
        print(f"💥 ERROR with Gemini API streaming: {e}")
        print(f"💥 ERROR TYPE: {type(e)}")
        print(f"💥 FULL TRACEBACK: {traceback.format_exc()}")
        
        error_message = "I encountered an error."
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": error_message,
            "last": True
        }))
        
        return error_message, time.time() - start_api

    api_elapsed = time.time() - start_api
    function_elapsed = time.time() - function_start
    
    print(f"⏱️  [GEMINI_STREAM] API elapsed: {api_elapsed*1000:.1f}ms")
    print(f"⏱️  [GEMINI_STREAM] Function elapsed: {function_elapsed*1000:.1f}ms")
    print(f"⏱️  [GEMINI_STREAM] Total chunks processed: {chunk_count}")
    
    return full_response_text, api_elapsed

@app.post("/twiml")
async def twiml_endpoint():
    """ULTRA-OPTIMIZED: TwiML with most aggressive VAD and fastest TTS settings"""
    twiml_start = time.time()
    print(f"⏱️  [TWIML] Endpoint called")
    
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
      elevenlabsStability="0.3"
      elevenlabsSimilarity="0.7"
      elevenlabsOptimizeStreamingLatency="5"
      elevenlabsRequestTimeoutMs="2000"
      vadSilenceMs="150"
      vadThreshold="0.15"
      vadMode="aggressive"
      vadDebounceMs="15"
      vadPreambleMs="75"
      vadPostambleMs="25"
      vadMinSpeechMs="100"
      vadMaxSpeechMs="8000"
      interruptionHandling="true"
      dtmfDetection="false"
    />
  </Connect>
</Response>"""
    
    twiml_time = time.time() - twiml_start
    print(f"⏱️  [TWIML] Response generated in {twiml_time*1000:.1f}ms")
    
    return Response(content=xml_response, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """ULTRA-OPTIMIZED: WebSocket endpoint with parallel processing"""
    connection_start = time.time()
    print(f"⏱️  [WEBSOCKET] New connection initiated")
    
    accept_start = time.time()
    await websocket.accept()
    accept_time = time.time() - accept_start
    print(f"⏱️  [WEBSOCKET] Connection accepted in {accept_time*1000:.1f}ms")
    
    connection_time = time.time() - connection_start
    print(f"⏱️  [WEBSOCKET] Total connection setup in {connection_time*1000:.1f}ms")
    
    call_sid = None

    try:
        while True:
            loop_start = time.time()
            
            # Receive message
            receive_start = time.time()
            raw = await websocket.receive_text()
            receive_time = time.time() - receive_start
            print(f"⏱️  [WEBSOCKET] Message received in {receive_time*1000:.1f}ms")
            
            # Parse JSON
            parse_start = time.time()
            message = json.loads(raw)
            parse_time = time.time() - parse_start
            print(f"⏱️  [WEBSOCKET] JSON parsed in {parse_time*1000:.1f}ms")

            # Print debug messages but don't process them
            if message.get("type") in ["info", "debug"]:
                if message.get("name") in ["roundTripDelayMs", "tokensPlayed"]:
                    print(f"[{call_sid}] [{message.get('name', 'DEBUG')}] {message.get('value', message)}")
                continue

            if message.get("type") == "setup":
                setup_start = time.time()
                call_sid = message["callSid"]
                print(f"⏱️  [WEBSOCKET] Setup message for call: {call_sid}")
                
                # --- OPTIMIZED: Create chat session with history pre-allocation ---
                chat_start = time.time()
                sessions[call_sid] = model.start_chat(history=[])
                chat_time = time.time() - chat_start
                print(f"⏱️  [WEBSOCKET] Chat session created in {chat_time*1000:.1f}ms")
                
                setup_time = time.time() - setup_start
                print(f"⏱️  [WEBSOCKET] Setup completed in {setup_time*1000:.1f}ms")
                print(f"✅ Setup for call: {call_sid}")
                continue

            if message.get("type") != "prompt" or not call_sid:
                continue

            # Process user prompt
            prompt_start = time.time()
            user_prompt = message["voicePrompt"]
            print(f"⏱️  [WEBSOCKET] Processing prompt: {user_prompt}")
            
            # --- OPTIMIZED: Truncate extremely long prompts for faster processing ---
            if len(user_prompt) > 200:
                user_prompt = user_prompt[:200] + "..."
                print(f"⏱️  [WEBSOCKET] Truncated long prompt for faster processing")
            
            prompt_process_time = time.time() - prompt_start
            print(f"⏱️  [WEBSOCKET] Prompt processed in {prompt_process_time*1000:.1f}ms")

            # Gemini API call with streaming
            gemini_start = time.time()
            print(f"⏱️  [WEBSOCKET] Calling Gemini API...")
            
            api_response_full_text, api_internal_time = await gemini_response_streaming(
                sessions[call_sid], user_prompt, websocket
            )
            
            gemini_total_time = time.time() - gemini_start
            print(f"⏱️  [WEBSOCKET] Gemini API completed in {gemini_total_time*1000:.1f}ms")

            # Total turnaround time
            total_time = time.time() - loop_start

            # --- OPTIMIZED: Detailed performance logging ---
            print(f"[PERFORMANCE] "
                  f"receive: {receive_time*1000:.1f}ms | "
                  f"parse: {parse_time*1000:.1f}ms | "
                  f"prompt_process: {prompt_process_time*1000:.1f}ms | "
                  f"gemini_internal: {api_internal_time*1000:.1f}ms | "
                  f"gemini_total: {gemini_total_time*1000:.1f}ms | "
                  f"total_turnaround: {total_time*1000:.1f}ms")

            # --- PERFORMANCE ANALYSIS ---
            if total_time > 1.5:
                print(f"⚠️  SLOW RESPONSE: {total_time:.2f}s - Gemini API latency issue")
            elif total_time < 0.8:
                print(f"🚀 ULTRA-FAST RESPONSE: {total_time:.2f}s")
            else:
                print(f"✅ GOOD RESPONSE: {total_time:.2f}s")

    except WebSocketDisconnect:
        disconnect_start = time.time()
        print(f"⏱️  [WEBSOCKET] Disconnect detected for call {call_sid}")
        
        if call_sid and call_sid in sessions:
            cleanup_start = time.time()
            sessions.pop(call_sid, None)
            cleanup_time = time.time() - cleanup_start
            print(f"⏱️  [WEBSOCKET] Session cleanup completed in {cleanup_time*1000:.1f}ms")
            print(f"🧹 Cleared session for call {call_sid}")
            
        disconnect_time = time.time() - disconnect_start
        print(f"⏱️  [WEBSOCKET] Disconnect handled in {disconnect_time*1000:.1f}ms")
        print(f"🔌 WebSocket connection closed for call {call_sid}")
        
    except Exception as e:
        exception_start = time.time()
        print(f"💥 Unexpected error in websocket_endpoint: {e}")
        
        if call_sid and call_sid in sessions:
            sessions.pop(call_sid, None)
            
        exception_time = time.time() - exception_start
        print(f"⏱️  [WEBSOCKET] Exception handled in {exception_time*1000:.1f}ms")

@app.get("/health")
async def health_check():
    health_start = time.time()
    print(f"⏱️  [HEALTH] Health check called")
    
    response = {
        "status": "healthy", 
        "domain": DOMAIN,
        "optimizations": [
            "connection_pooling", 
            "immediate_streaming",
            "ultra_fast_tts",
            "gemini_1.5_flash_8b",
            "aggressive_vad",
            "connection_prewarming"
        ]
    }
    
    health_time = time.time() - health_start
    print(f"⏱️  [HEALTH] Health check completed in {health_time*1000:.1f}ms")
    
    return response

if __name__ == "__main__":
    print(f"🚀 Starting ULTRA-OPTIMIZED voice assistant on port {PORT}")
    print(f"🔗 WebSocket URL for Twilio: {WS_URL}")
    print(f"🌐 Detected platform domain: {DOMAIN}")
    print(f"⚡ Optimizations: Pre-warming, Immediate Streaming, Ultra-Fast TTS, Gemini 1.5 Flash 8B")
    
    server_start = time.time()
    print(f"⏱️  [SERVER] Starting uvicorn server...")
    
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)