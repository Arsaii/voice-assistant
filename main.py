import os
import json
import time
from fastapi import FastAPI, Request, WebSocket, HTTPException, status
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect, ConversationRelay
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client as TwilioClient # Renamed to avoid conflict
import google.generativeai as genai
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# --- Configuration ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # For Gemini

# Initialize Twilio Client (used for sending welcome messages if needed, not directly in this flow)
# Ensure you have your Twilio Account SID and Auth Token set as environment variables
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Configure Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini model initialization (using a global dict to store chat sessions)
sessions = {}

# --- WebSocket Endpoint ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    call_sid = None
    chat = None # Initialize chat session

    print(f"WebSocket connection accepted for a new call.")

    try:
        while True:
            t0_loop_start = time.time() # To measure the time for one loop iteration
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)

            # *** IMPORTANT: Print ALL incoming messages for debugging ***
            # This is crucial for seeing the debug messages from Twilio
            print(f"WS Incoming ({time.time():.4f}s from start): {json.dumps(message, indent=2)}")

            # Handle Twilio setup message
            if message.get("event") == "start":
                call_sid = message["start"]["callSid"]
                stream_sid = message["start"]["streamSid"] # Added stream_sid for full context
                print(f"[{call_sid}] Setup for call: {call_sid}, Stream SID: {stream_sid}")
                # Initialize Gemini chat session for this call
                sessions[call_sid] = genai.GenerativeModel("gemini-2.5-flash").start_chat(history=[])
                chat = sessions[call_sid]
                continue

            # Handle Twilio media (audio) messages (STT results)
            if message.get("event") == "media":
                # Only process if it's a "speech_final" event from Twilio's STT
                if message["media"].get("track") == "inbound" and message["media"].get("status") == "speech_final":
                    prompt_text = message["media"]["transcript"].strip()
                    if not prompt_text:
                        print(f"[{call_sid}] Received empty prompt, skipping.")
                        continue

                    print(f"[{call_sid}] Received prompt: {prompt_text}")

                    # Profiling start for this turn
                    profile_start_parse_prep = time.time()

                    # Your existing prompt processing logic
                    if chat is None:
                        # This should ideally not happen if "start" message is handled
                        print(f"[{call_sid}] Chat session not initialized, re-initializing.")
                        sessions[call_sid] = genai.GenerativeModel("gemini-2.5-flash").start_chat(history=[])
                        chat = sessions[call_sid]

                    # Profiling end for parse/prep
                    profile_end_parse_prep = time.time()

                    # Gemini API call and streaming chunks back to Twilio
                    response_text_chunks = []
                    profile_start_api_sending_chunks = time.time()

                    try:
                        # Generate content from Gemini
                        gemini_response = await asyncio.to_thread(chat.send_message, prompt_text, stream=True)

                        for chunk in gemini_response:
                            if chunk.text: # Ensure there's text in the chunk
                                response_text_chunks.append(chunk.text)
                                # Send chunk to Twilio ConversationRelay
                                await websocket.send_text(json.dumps({
                                    "type": "text",
                                    "token": chunk.text,
                                    "last": False # Not the last chunk yet
                                }))
                                # print(f"[{call_sid}] Sent chunk: '{chunk.text}'") # Uncomment for very verbose logging
                        
                        # Send the final 'last: true' message to Twilio
                        await websocket.send_text(json.dumps({
                            "type": "text",
                            "token": "", # Can be empty for the last token
                            "last": True
                        }))
                        print(f"[{call_sid}] Sent final 'last: true' message.")

                    except Exception as e:
                        print(f"[{call_sid}] Error during Gemini API call or sending chunks: {e}")
                        # Send an error message back to the user via TTS
                        await websocket.send_text(json.dumps({
                            "type": "text",
                            "token": "I'm sorry, I encountered an error. Please try again.",
                            "last": True
                        }))
                    
                    profile_end_api_sending_chunks = time.time()

                    # Total turnaround time for this prompt
                    total_turnaround_time = time.time() - t0_loop_start # From when prompt was received to after sending 'last:true'

                    print(f"[{call_sid}] [PROFILE] parse/prep: {profile_end_parse_prep - profile_start_parse_prep:.4f}s | "
                          f"api_and_sending_chunks: {profile_end_api_sending_chunks - profile_start_api_sending_chunks:.4f}s | "
                          f"total_turnaround: {total_turnaround_time:.4f}s") # This `total_turnaround` is misleading, it's just the loop duration

                elif message["media"].get("track") == "inbound" and message["media"].get("status") == "speech_interim":
                    # print(f"[{call_sid}] Interim transcript: {message['media']['transcript']}")
                    pass # You can process interim results if needed for interruption handling etc.

            # Handle Twilio mark (acknowledgement) messages
            elif message.get("event") == "mark":
                print(f"[{call_sid}] Received mark: {message['mark']['name']}")

            # Handle Twilio stop message (call ended)
            elif message.get("event") == "stop":
                print(f"[{call_sid}] WebSocket connection closed for call {call_sid}")
                if call_sid in sessions:
                    del sessions[call_sid]
                    print(f"[{call_sid}] Cleared session for call {call_sid}")
                break # Exit the WebSocket loop

            # Handle Twilio debug messages (NEW - CRITICAL FOR YOU)
            elif message.get("type") == "debug":
                print(f"[{call_sid}] [DEBUG MSG] {json.dumps(message, indent=2)}")
                # Pay close attention to 'speaker-events' and 'tokens-played' here
                # Example:
                # {"type": "debug", "debug": {"event": "speaker-events", "name": "agentSpeakingStarted", "timestamp": "..."}}
                # {"type": "debug", "debug": {"event": "tokens-played", "tokens": [{"token": "Hello,", "start": "...", "end": "..."}], "timestamp": "..."}}
                # Compare these timestamps with your server's `api_and_sending_chunks` end time.

    except Exception as e:
        print(f"WebSocket error for call {call_sid}: {e}")
    finally:
        if websocket.client_state == 1: # WebSocketState.CONNECTED
            await websocket.close()
        print(f"INFO:     connection closed for call {call_sid}")


# --- TwiML Webhook Endpoint ---

@app.post("/twiml")
async def twiml_webhook(request: Request):
    """
    Twilio will hit this endpoint when a call comes in.
    It tells Twilio to connect to our WebSocket server for real-time interaction.
    """
    try:
        # You can get call information from request.form if needed
        # call_sid = request.form.get("CallSid")
        
        response = VoiceResponse()
        connect = Connect()
        
        # NOTE: Railway's public URL format might be different for your deployment.
        # Make sure this is the correct external URL where your WebSocket is accessible.
        # It's usually your Railway app's domain.railway.app/ws
        websocket_url = os.getenv("WEBSOCKET_URL", "wss://your-railway-app-domain.railway.app/ws")

        # Welcome greeting can be handled by ConversationRelay directly.
        # This will be played by ElevenLabs at the start of the call.
        welcome_greeting = "Hello, I am your AI assistant. How can I help you today?"

        # --- CRITICAL DEBUGGING ATTRIBUTES ADDED HERE ---
        # "debugging": Provides general debug info
        # "speaker-events": Sends messages when agent starts/stops speaking
        # "tokens-played": Sends messages about which tokens were played and their timing
        # "elevenlabsTextNormalization="off": Can slightly reduce latency for ElevenLabs
        
        connect.conversation_relay(
            url=websocket_url,
            welcome_greeting=welcome_greeting,
            tts_provider="ElevenLabs",
            voice="FGY2WhTYpPnrIDTdsKH5", # Ensure this is an ElevenLabs voice
            debug="debugging speaker-events tokens-played", # <-- ADDED DEBUGGING
            elevenlabs_text_normalization="off" # <-- ADDED TO REDUCE POTENTIAL LATENCY
        )
        response.append(connect)
        
        print(f"Returning TwiML: {response}")
        return Response(content=str(response), media_type="application/xml")

    except Exception as e:
        print(f"Error in TwiML webhook: {e}")
        # Return a simple voice response for error
        response = VoiceResponse()
        response.say("I am sorry, there was an error connecting. Please try again.")
        return Response(content=str(response), media_type="application/xml"), status.HTTP_500_INTERNAL_SERVER_ERROR

# --- Health Check Endpoint (Optional but Recommended for Deployments) ---
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- For local testing (if you run 'python main.py') ---
# This part is typically not used in a production Docker deployment like Railway,
# where uvicorn is started via a command.
if __name__ == "__main__":
    import uvicorn
    # Make sure to set WEBSOCKET_URL in your environment or .env file
    # For local testing, it might be ws://localhost:8000/ws
    # For Ngrok, it would be wss://your-ngrok-domain.ngrok.io/ws
    # For Railway, wss://your-railway-app-domain.railway.app/ws
    
    # Example for local development with Ngrok
    # If using ngrok, your WEBSOCKET_URL should be like "wss://<ngrok_id>.ngrok-free.app/ws"
    # And your Twilio webhook for incoming calls should point to "https://<ngrok_id>.ngrok-free.app/twiml"
    
    print(f"Starting server with WebSocket URL: {os.getenv('WEBSOCKET_URL')}")
    uvicorn.run(app, host="0.0.0.0", port=8000)