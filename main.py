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

# --- OPTIMIZED: Shorter, more focused system prompt for voice ---
SYSTEM_PROMPT = """You are a helpful voice assistant. Keep responses conversational and concise since this is a phone call. 
Rules:
1. Be direct and brief - aim for 1-2 sentences per response
2. Spell out numbers (say 'twenty-three' not '23')
3. No special characters, bullets, or formatting
4. Sound natural and friendly"""

# --- Gemini API Initialization ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=GOOGLE_API_KEY)

# --- OPTIMIZED: Generation config for faster responses ---
generation_config = genai.GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    max_output_tokens=100,  # Shorter responses for voice
    candidate_count=1,
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

# --- OPTIMIZED: Regular model with generation config and safety settings ---
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",  # More stable than 2.5-flash
    system_instruction=SYSTEM_PROMPT,
    generation_config=generation_config,
    safety_settings=safety_settings
)

# --- OPTIMIZED: HTTP session for connection pooling ---
http_session = None

async def create_http_session():
    """Create optimized HTTP session for better connection management"""
    session_start = time.time()
    print(f"⏱️  [HTTP_SESSION] Creating HTTP session...")
    
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=30,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=30,
    )
    session = aiohttp.ClientSession(connector=connector)
    
    session_time = time.time() - session_start
    print(f"⏱️  [HTTP_SESSION] Session created in {session_time*1000:.1f}ms")
    return session

# Store active chat sessions
sessions: dict[str, any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler (replaces deprecated startup/shutdown events)"""
    # Startup
    startup_start = time.time()
    print(f"⏱️  [STARTUP] Application startup beginning...")
    
    global http_session
    http_session = await create_http_session()
    
    startup_time = time.time() - startup_start
    print(f"⏱️  [STARTUP] Application startup completed in {startup_time*1000:.1f}ms")
    print("✅ HTTP session initialized")
    
    yield
    
    # Shutdown
    shutdown_start = time.time()
    print(f"⏱️  [SHUTDOWN] Application shutdown beginning...")
    
    if http_session:
        await http_session.close()
        
    shutdown_time = time.time() - shutdown_start
    print(f"⏱️  [SHUTDOWN] Application shutdown completed in {shutdown_time*1000:.1f}ms")
    print("✅ HTTP session closed")

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

async def gemini_response_streaming(chat_session, user_prompt, websocket):
    """
    OPTIMIZED: Streaming with intelligent buffering for better performance.
    Buffers chunks until we have meaningful content or sentence boundaries.
    """
    function_start = time.time()
    print(f"⏱️  [GEMINI_STREAM] Function started")
    
    start_api = time.time()
    full_response_text = ""
    buffer = ""
    buffer_size = 20  # Minimum characters before sending (reduced for even faster response)
    
    try:
        # --- OPTIMIZED: Reduced timeout for faster failure detection ---
        print(f"⏱️  [GEMINI_STREAM] Sending message to Gemini API...")
        api_call_start = time.time()
        
        response_stream = await asyncio.wait_for(
            chat_session.send_message_async(user_prompt, stream=True),
            timeout=6.0  # Reduced from 8s to 6s
        )
        
        api_call_time = time.time() - api_call_start
        print(f"⏱️  [GEMINI_STREAM] API call initiated in {api_call_time*1000:.1f}ms")
        
        first_chunk_sent = False
        chunk_count = 0
        first_chunk_time = None
        
        async for chunk in response_stream:
            chunk_start = time.time()
            chunk_count += 1
            
            if chunk.text:  # This is where the error occurred
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    ttfb = first_chunk_time - start_api
                    print(f"⏱️  [GEMINI_STREAM] First chunk received (TTFB): {ttfb*1000:.1f}ms")
                
                full_response_text += chunk.text
                buffer += chunk.text
                
                # --- OPTIMIZED: Send first chunk immediately for faster TTFB ---
                if not first_chunk_sent and len(buffer) >= 5:  # Even smaller first chunk
                    send_start = time.time()
                    await websocket.send_text(json.dumps({
                        "type": "text",
                        "token": buffer,
                        "last": False
                    }))
                    send_time = time.time() - send_start
                    print(f"⏱️  [GEMINI_STREAM] First chunk sent to WebSocket in {send_time*1000:.1f}ms")
                    buffer = ""
                    first_chunk_sent = True
                    continue
                
                # --- OPTIMIZED: Smart buffering based on content and length ---
                should_send = (
                    len(buffer) >= buffer_size or 
                    any(punct in buffer for punct in ['.', '!', '?', '\n']) or
                    buffer.endswith(', ') or  # Natural pause points
                    len(buffer) >= 50  # Don't let buffer get too large
                )
                
                if should_send and first_chunk_sent:
                    send_start = time.time()
                    await websocket.send_text(json.dumps({
                        "type": "text",
                        "token": buffer,
                        "last": False
                    }))
                    send_time = time.time() - send_start
                    print(f"⏱️  [GEMINI_STREAM] Chunk {chunk_count} sent to WebSocket in {send_time*1000:.1f}ms")
                    buffer = ""
                    
            elif chunk.candidates and chunk.candidates[0].finish_reason:
                # --- ADDED: Handle safety blocks and other finish reasons ---
                finish_reason = chunk.candidates[0].finish_reason
                print(f"⏱️  [GEMINI_STREAM] Finish reason received: {finish_reason}")
                
                if finish_reason == 2:  # SAFETY
                    error_message = "I'm sorry, I can't respond to that right now. Could you try rephrasing your question?"
                elif finish_reason == 3:  # RECITATION
                    error_message = "I apologize, but I need to avoid that response. Could you ask something else?"
                elif finish_reason == 4:  # OTHER
                    error_message = "I'm having trouble with that request. Could you try again?"
                else:
                    error_message = "I encountered an issue. Please try again."
                
                error_send_start = time.time()
                await websocket.send_text(json.dumps({
                    "type": "text",
                    "token": error_message,
                    "last": True
                }))
                error_send_time = time.time() - error_send_start
                print(f"⏱️  [GEMINI_STREAM] Error message sent in {error_send_time*1000:.1f}ms")
                
                function_time = time.time() - function_start
                print(f"⏱️  [GEMINI_STREAM] Function completed (with error) in {function_time*1000:.1f}ms")
                return error_message, time.time() - start_api

            chunk_time = time.time() - chunk_start
            print(f"⏱️  [GEMINI_STREAM] Chunk {chunk_count} processed in {chunk_time*1000:.1f}ms")

        # Send any remaining buffer
        if buffer:
            final_send_start = time.time()
            await websocket.send_text(json.dumps({
                "type": "text",
                "token": buffer,
                "last": False
            }))
            final_send_time = time.time() - final_send_start
            print(f"⏱️  [GEMINI_STREAM] Final buffer sent in {final_send_time*1000:.1f}ms")
        
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
        timeout_start = time.time()
        print(f"⏱️  [GEMINI_STREAM] Timeout occurred after 8 seconds")
        
        error_message = "I'm having trouble right now. Could you try again?"
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": error_message,
            "last": True
        }))
        
        timeout_time = time.time() - timeout_start
        print(f"⏱️  [GEMINI_STREAM] Timeout handled in {timeout_time*1000:.1f}ms")
        
        function_time = time.time() - function_start
        print(f"⏱️  [GEMINI_STREAM] Function completed (timeout) in {function_time*1000:.1f}ms")
        return error_message, time.time() - start_api
        
    except Exception as e:
        error_start = time.time()
        import traceback
        print(f"💥 ERROR with Gemini API streaming: {e}")
        print(f"💥 ERROR TYPE: {type(e)}")
        print(f"💥 FULL TRACEBACK: {traceback.format_exc()}")
        
        error_message = "I encountered an error. Please try again."
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": error_message,
            "last": True
        }))
        
        error_time = time.time() - error_start
        print(f"⏱️  [GEMINI_STREAM] Error handled in {error_time*1000:.1f}ms")
        
        function_time = time.time() - function_start
        print(f"⏱️  [GEMINI_STREAM] Function completed (error) in {function_time*1000:.1f}ms")
        return error_message, time.time() - start_api

    api_elapsed = time.time() - start_api
    function_elapsed = time.time() - function_start
    
    print(f"⏱️  [GEMINI_STREAM] API elapsed: {api_elapsed*1000:.1f}ms")
    print(f"⏱️  [GEMINI_STREAM] Function elapsed: {function_elapsed*1000:.1f}ms")
    print(f"⏱️  [GEMINI_STREAM] Total chunks processed: {chunk_count}")
    
    return full_response_text, api_elapsed

@app.post("/twiml")
async def twiml_endpoint():
    """OPTIMIZED: Return TwiML with shorter VAD settings for more responsive detection"""
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
    
    twiml_time = time.time() - twiml_start
    print(f"⏱️  [TWIML] Response generated in {twiml_time*1000:.1f}ms")
    
    return Response(content=xml_response, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """OPTIMIZED: WebSocket endpoint with detailed performance profiling"""
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
                
                # --- OPTIMIZED: Start chat with generation config ---
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
            if total_time > 2.0:
                print(f"⚠️  SLOW RESPONSE: {total_time:.2f}s - Consider optimizing")
            elif total_time < 1.0:
                print(f"✅ FAST RESPONSE: {total_time:.2f}s")

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
            "smart_buffering",
            "fast_tts_config",
            "reduced_timeouts",
            "aggressive_vad"
        ]
    }
    
    health_time = time.time() - health_start
    print(f"⏱️  [HEALTH] Health check completed in {health_time*1000:.1f}ms")
    
    return response

if __name__ == "__main__":
    print(f"🚀 Starting optimized voice assistant on port {PORT}")
    print(f"🔗 WebSocket URL for Twilio: {WS_URL}")
    print(f"🌐 Detected platform domain: {DOMAIN}")
    print(f"⚡ Optimizations: Connection Pooling, Smart Buffering, Fast TTS, Aggressive VAD")
    
    # --- OPTIMIZED: Adjusted worker count for voice workload ---
    server_start = time.time()
    print(f"⏱️  [SERVER] Starting uvicorn server...")
    
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)