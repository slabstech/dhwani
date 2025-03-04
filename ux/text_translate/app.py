import gradio as gr
import os
import requests
import json
import logging

# Set up logging
logging.basicConfig(filename='execution.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        port_mapping = {
            "asr": 10860,
            "translate": 8860,
            "tts": 9860  # Added TTS service port
        }
        base_url = f'http://localhost:7860'
    else:
        base_url = f'https://gaganyatri-translate-indic-server-cpu.hf.space'
    logging.info(f"Endpoint for {service_type}: {base_url}")
    return base_url

def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

def translate_text(transcription, src_lang, tgt_lang, use_gpu=False, use_localhost=False):
    logging.info(f"Translating text: {transcription}, src_lang: {src_lang}, tgt_lang: {tgt_lang}, use_gpu: {use_gpu}, use_localhost: {use_localhost}")
    base_url = get_endpoint(use_gpu, use_localhost, "translate")
    device_type = "cuda" if use_gpu else "cpu"
    url = f'{base_url}/translate?src_lang={src_lang}&tgt_lang={tgt_lang}&device_type={device_type}'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    chunk_size =15
    chunked_text = chunk_text(transcription, chunk_size=chunk_size)

    data = {
        "sentences": chunked_text,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        logging.info(f"Translation successful: {response.json()}")

        translated_texts = response.json().get('translations', [])
        merged_translated_text = ' '.join(translated_texts)
        return {'translations': [merged_translated_text]}
    except requests.exceptions.RequestException as e:
        logging.error(f"Translation failed: {e}")
        return {"translations": [""]}


# Create the Gradio interface
with gr.Blocks(title="Dhwani Translate - Convert to any Language") as demo:
    gr.Markdown("# Text Translate- To any lanugage")
    
    translate_src_language = gr.Dropdown(
        choices=list(language_mapping.keys()),
        label="Source Language - Fixed",
        value="Kannada",
        interactive=False
    )
    translate_tgt_language = gr.Dropdown(
        choices=list(language_mapping.keys()),
        label="Target Language",
        value="English"
    )

    with gr.Row():
        input_text = gr.Textbox(label="Enter Text", placeholder="Type your text here...")

    
    submit_button = gr.Button("Translate Text")
    
    translation_output = gr.Textbox(label="Translated Text", interactive=False)


    use_gpu_checkbox = gr.Checkbox(label="Use GPU", value=False, interactive=False, visible=False)
    use_localhost_checkbox = gr.Checkbox(label="Use Localhost", value=False, interactive=False, visible=False)
    #resubmit_button = gr.Button(value="Resubmit Translation")

    def on_transcription_complete(transcription, src_lang, tgt_lang, use_gpu, use_localhost):
        src_lang_id = language_mapping[src_lang]
        tgt_lang_id = language_mapping[tgt_lang]
        logging.info(f"Transcription complete: {transcription}, src_lang: {src_lang_id}, tgt_lang: {tgt_lang_id}, use_gpu: {use_gpu}, use_localhost: {use_localhost}")
        translation = translate_text(transcription, src_lang_id, tgt_lang_id, use_gpu, use_localhost)
        translated_text = translation['translations'][0]
        return translated_text

    submit_button.click(
        on_transcription_complete,
        inputs=[input_text, translate_src_language, translate_tgt_language, use_gpu_checkbox, use_localhost_checkbox ],
        outputs=[translation_output]
    )


# Launch the interface
demo.launch()