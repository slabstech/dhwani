import gradio as gr
import os
import requests
import json
import logging
from pydub import AudioSegment
from pydub.playback import play
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(filename='execution.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Mapping of user-friendly language names to language IDs
language_mapping = {
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "English": "eng_Latn",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic)": "kas_Arab",
    "Kashmiri (Devanagari)": "kas_Deva",
    "Konkani": "gom_Deva",
    "Malayalam": "mal_Mlym",
    "Manipuri (Bengali)": "mni_Beng",
    "Manipuri (Meitei)": "mni_Mtei",
    "Maithili": "mai_Deva",
    "Marathi": "mar_Deva",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi (Arabic)": "snd_Arab",
    "Sindhi (Devanagari)": "snd_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab"
}

def get_endpoint(use_gpu, use_localhost, service_type):
    logging.info(f"Getting endpoint for service: {service_type}, use_gpu: {use_gpu}, use_localhost: {use_localhost}")
    device_type_ep = "" if use_gpu else "-cpu"
    if use_localhost:
        base_url = f'http://localhost:{config["endpoints"][service_type]["localhost"]}'
    else:
        base_url = f'{os.getenv("REMOTE_ENDPOINT")}{device_type_ep}'
    logging.info(f"Endpoint for {service_type}: {base_url}")
    return base_url

def transcribe_audio(audio_path, use_gpu, use_localhost):
    logging.info(f"Transcribing audio from {audio_path}, use_gpu: {use_gpu}, use_localhost: {use_localhost}")
    base_url = get_endpoint(use_gpu, use_localhost, "asr")
    url = f'{base_url}/transcribe/?language=kannada'
    files = {'file': open(audio_path, 'rb')}
    api_token = os.getenv("API_TOKEN")
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        response = requests.post(url, files=files, headers=headers)
        response.raise_for_status()
        transcription = response.json()
        logging.info(f"Transcription successful: {transcription}")
        return transcription.get('text', '')
    except requests.exceptions.RequestException as e:
        logging.error(f"Transcription failed: {e}")
        return ""

def get_audio(input_text, voice_description=config["voice_description"]):
    try:
        # Define the API endpoint and headers
        url = f'{os.getenv("REMOTE_ENDPOINT")}/v1/audio/speech'
        api_token = os.getenv("HF_TOKEN")
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
        # Define the request payload
        payload = {
            "input": input_text,
            "voice": voice_description
        }
        # Send the POST request
        response = requests.post(url, json=payload, headers=headers, stream=True)
        # Check if the request was successful
        if response.status_code == 200:
            logger.info(f"API request successful. Status code: {response.status_code}")
            # Save the audio file
            audio_file_path = "output_audio.mp3"
            with open(audio_file_path, "wb") as audio_file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        audio_file.write(chunk)
            logger.info(f"Audio file saved to: {audio_file_path}")
            # Return the path to the saved audio file
            return audio_file_path
        else:
            logger.error(f"API request failed. Status code: {response.status_code}, {response.text}")
            return f"Error: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {e}")
        return f"Request error: {e}"
    except Exception as e:
        logger.error(f"General exception: {e}")
        return f"Error: {e}"

def send_to_mistral(transcription):
    api_key = os.getenv("MISTRAL_API_KEY")
    model = "mistral-saba-latest"
    client = Mistral(api_key=api_key)

    system_prompt = "You are a helpful assistant. Provide a concise response in one sentence maximum to the user's query."

    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": transcription,
            },
        ]
    )
    return chat_response.choices[0].message.content

# Create the Gradio interface
with gr.Blocks(title="Dhwani - Voice AI for Kannada /ಕನ್ನಡಕ್ಕಾಗಿ ಧ್ವನಿ AI ") as demo:
    gr.Markdown("# Voice AI for Kannada")
    gr.Markdown("Record your voice and get your answer in Kannada/ ನಿಮ್ಮ ಧ್ವನಿಯನ್ನು ರೆಕಾರ್ಡ್ ಮಾಡಿ ಮತ್ತು ನಿಮ್ಮ ಉತ್ತರವನ್ನು ಕನ್ನಡದಲ್ಲಿ ಪಡೆಯಿರಿ")
    gr.Markdown("Click on Recording button, to ask your question / ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಲು ರೆಕಾರ್ಡಿಂಗ್ ಬಟನ್ ಮೇಲೆ ಕ್ಲಿಕ್ ಮಾಡಿ")
    gr.Markdown("Click on stop recording button, to submit your question/ ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಸಲ್ಲಿಸಲು ರೆಕಾರ್ಡಿಂಗ್ ನಿಲ್ಲಿಸು ಬಟನ್ ಮೇಲೆ ಕ್ಲಿಕ್ ಮಾಡಿ")

    audio_input = gr.Microphone(type="filepath", label="Record your voice")
    audio_output = gr.Audio(type="filepath", label="Playback", interactive=False)
    transcription_output = gr.Textbox(label="Transcription Result", interactive=False)
    mistral_output = gr.Textbox(label="Dhwani answer/ ಉತ್ತರ", interactive=False)
    tts_output = gr.Audio(label="Generated Audio", interactive=False)

    enable_tts_checkbox = gr.Checkbox(label="Enable Text-to-Speech", value=True, interactive=False, visible=False)
    use_gpu_checkbox = gr.Checkbox(label="Use GPU", value=True, interactive=True, visible=False)
    use_localhost_checkbox = gr.Checkbox(label="Use Localhost", value=False, interactive=False, visible=False)

    voice_description = gr.Textbox(value="Anu speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality.", label="Voice Description", interactive=False, visible=False)

    def process_audio(audio_path, use_gpu, use_localhost):
        logging.info(f"Processing audio from {audio_path}, use_gpu: {use_gpu}, use_localhost: {use_localhost}")
        transcription = transcribe_audio(audio_path, use_gpu, use_localhost)
        return transcription

    def on_transcription_complete(transcription, voice_description, enable_tts):
        mistral_response = send_to_mistral(transcription)
        if enable_tts:
            audio_file_path = get_audio(mistral_response, voice_description)
        else:
            audio_file_path = None
        return mistral_response, audio_file_path

    audio_input.stop_recording(
        fn=process_audio,
        inputs=[audio_input, use_gpu_checkbox, use_localhost_checkbox],
        outputs=transcription_output
    )

    transcription_output.change(
        fn=on_transcription_complete,
        inputs=[transcription_output, voice_description, enable_tts_checkbox],
        outputs=[mistral_output, tts_output]
    )

# Launch the interface
demo.launch()