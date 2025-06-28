import os
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs
import vlc  # VLC binding

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ELEVENLABS_API_KEY"] = os.getenv("ELEVENLABS_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize clients
# client = OpenAI()
eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
client = genai.Client()



grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)
config = types.GenerateContentConfig(
    system_instruction="You are a helpful assistant who answers users' questions in a short and concise manner. You always end your answers with a deez nuts joke.",
    tools=[grounding_tool]
)

# 1️⃣ Ask assistant
# response = client.responses.create(
#     model="gpt-4.1-mini",
#     instructions="You are an Indian helpful assistant who answers users questions. You end all answers with a deez nuts joke.",
#     tools=[{"type": "web_search_preview", "user_location": {
#         "type": "approximate", "country": "IN", "city": "Hyderabad", "region": "Hyderabad"
#     }}],
#     input="what is the capital of France?",
# )
# text = response.output_text.strip()
# print("Assistant said:", text)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="what is the capital of France?",
    config=config,
)
text= response.text.strip()
print("Assistant said:", text)

# 2️⃣ Generate TTS and play with VLC
def tts_and_play(text, voice_id, output_path="response.mp3"):
    # Stream and save TTS
    stream = eleven.text_to_speech.stream(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )
    with open(output_path, "wb") as f:
        for chunk in stream:
            if chunk:
                f.write(chunk)
    print(f"Saved audio to {output_path}")

    # Play with VLC
    player = vlc.MediaPlayer(output_path)
    player.play()
    # Wait while it plays
    import time
    time.sleep(0.5)
    duration = player.get_length() / 1000  # in seconds
    time.sleep(duration)
    player.stop()
    print("Playback finished.")

tts_and_play(text, voice_id="LruHrtVF6PSyGItzMNHS")
