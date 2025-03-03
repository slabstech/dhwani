from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, Body, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import os
import tempfile
import io
import logging
from logging.handlers import RotatingFileHandler
from time import time, perf_counter
from typing import List, Dict, Annotated, Any
import argparse
import uvicorn
import zipfile
import soundfile as sf
import numpy as np
from contextlib import asynccontextmanager
from tts_config import SPEED, ResponseFormat, config
from logger import logger
from asr_manager import ASRModelManager
from tts_manager import TTSModelManager, lifespan
from translate_manager import ModelManager, ip

# Configure logging with log rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("transcription_api.log", maxBytes=10*1024*1024, backupCount=5), # 10MB per file, keep 5 backup files
        logging.StreamHandler() # This will also print logs to the console
    ]
)

# Allow requests from your Gradio Space’s domain
origins = [
    "https://gaganyatri-dhwani-voice-model.hf.space/",
    "https://gaganyatri-dhwani-voice-to-any.hf.space/",
        "https://gaganyatri-dhwani-tts.hf.space/",
    "https://gaganyatri-dhwani.hf.space/"
      # Replace with your Gradio Space URL
    "http://localhost:7860",  # For local testing
]

# Initialize ASR and Translation Model Managers
asr_manager = ASRModelManager()
model_manager = ModelManager()
tts_model_manager = TTSModelManager()

# Define the response models
class TranscriptionResponse(BaseModel):
    text: str

class BatchTranscriptionResponse(BaseModel):
    transcriptions: List[str]

class TranslationRequest(BaseModel):
    sentences: List[str]
    src_lang: str
    tgt_lang: str

class TranslationResponse(BaseModel):
    translations: List[str]

def get_translate_manager(src_lang: str, tgt_lang: str) -> TranslateManager:
    return model_manager.get_model(src_lang, tgt_lang)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/test")
def test_endpoint():
    return {"message": "Hello from FastAPI"}

def create_in_memory_zip(file_data):
    in_memory_zip = io.BytesIO()
    with zipfile.ZipFile(in_memory_zip, 'w') as zipf:
        for file_name, data in file_data.items():
            zipf.writestr(file_name, data)
    in_memory_zip.seek(0)
    return in_memory_zip

# Define a function to split text into smaller chunks
def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

@app.post("/v1/audio/speech")
async def generate_audio(
    input: Annotated[str, Body()] = config.input,
    voice: Annotated[str, Body()] = config.voice,
    model: Annotated[str, Body()] = config.model,
    response_format: Annotated[ResponseFormat, Body(include_in_schema=False)] = config.response_format,
    speed: Annotated[float, Body(include_in_schema=False)] = SPEED,
) -> StreamingResponse:
    tts, tokenizer, description_tokenizer = tts_model_manager.get_or_load_model(model)
    if speed != SPEED:
        logger.warning(
            "Specifying speed isn't supported by this model. Audio will be generated with the default speed"
        )
    start = perf_counter()
    # Tokenize the voice description
    input_ids = description_tokenizer(voice, return_tensors="pt").input_ids.to(DEVICE)
    # Tokenize the input text
    prompt_input_ids = tokenizer(input, return_tensors="pt").input_ids.to(DEVICE)
    # Generate the audio
    generation = tts.generate(
        input_ids=input_ids, prompt_input_ids=prompt_input_ids
    ).to(torch.float32)
    audio_arr = generation.cpu().numpy().squeeze()
    # Ensure device is a string
    device_str = str(DEVICE)
    logger.info(
        f"Took {perf_counter() - start:.2f} seconds to generate audio for {len(input.split())} words using {device_str.upper()}"
    )
    # Create an in-memory file
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_arr, tts.config.sampling_rate, format=response_format)
    audio_buffer.seek(0)
    return StreamingResponse(audio_buffer, media_type=f"audio/{response_format}")

@app.post("/v1/audio/speech_batch")
async def generate_audio_batch(
    input: Annotated[List[str], Body()] = config.input,
    voice: Annotated[List[str], Body()] = config.voice,
    model: Annotated[str, Body(include_in_schema=False)] = config.model,
    response_format: Annotated[ResponseFormat, Body()] = config.response_format,
    speed: Annotated[float, Body(include_in_schema=False)] = SPEED,
) -> StreamingResponse:
    tts, tokenizer, description_tokenizer = tts_model_manager.get_or_load_model(model)
    if speed != SPEED:
        logger.warning(
            "Specifying speed isn't supported by this model. Audio will be generated with the default speed"
        )
    start = perf_counter()
    length_of_input_text = len(input)
    # Set chunk size for text processing
    chunk_size = 15 # Adjust this value based on your needs
    # Prepare inputs for the model
    all_chunks = []
    all_descriptions = []
    for i, text in enumerate(input):
        chunks = chunk_text(text, chunk_size)
        all_chunks.extend(chunks)
        all_descriptions.extend([voice[i]] * len(chunks))
    description_inputs = description_tokenizer(all_descriptions, return_tensors="pt", padding=True).to("cuda")
    prompts = tokenizer(all_chunks, return_tensors="pt", padding=True).to("cuda")
    set_seed(0)
    generation = tts.generate(
        input_ids=description_inputs.input_ids,
        attention_mask=description_inputs.attention_mask,
        prompt_input_ids=prompts.input_ids,
        prompt_attention_mask=prompts.attention_mask,
        do_sample=True,
        return_dict_in_generate=True,
    )
    # Concatenate audio outputs
    audio_outputs = []
    current_index = 0
    for i, text in enumerate(input):
        chunks = chunk_text(text, chunk_size)
        chunk_audios = []
        for j in range(len(chunks)):
            audio_arr = generation.sequences[current_index][:generation.audios_length[current_index]].cpu().numpy().squeeze()
            audio_arr = audio_arr.astype('float32')
            chunk_audios.append(audio_arr)
            current_index += 1
        combined_audio = np.concatenate(chunk_audios)
        audio_outputs.append(combined_audio)
    # Save the final audio outputs in memory
    file_data = {}
    for i, audio in enumerate(audio_outputs):
        file_name = f"out_{i}.{response_format}"
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio, tts.config.sampling_rate, format=response_format)
        audio_bytes.seek(0)
        file_data[file_name] = audio_bytes.read()
    # Create in-memory zip file
    in_memory_zip = create_in_memory_zip(file_data)
    logger.info(
        f"Took {perf_counter() - start:.2f} seconds to generate audio"
    )
    return StreamingResponse(in_memory_zip, media_type="application/zip")

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest, translate_manager: TranslateManager = Depends(get_translate_manager)):
    input_sentences = request.sentences
    src_lang = request.src_lang
    tgt_lang = request.tgt_lang
    if not input_sentences:
        raise HTTPException(status_code=400, detail="Input sentences are required")
    batch = ip.preprocess_batch(
        input_sentences,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    # Tokenize the sentences and generate input encodings
    inputs = translate_manager.tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(translate_manager.device_type)
    # Generate translations using the model
    with torch.no_grad():
        generated_tokens = translate_manager.model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )
    # Decode the generated tokens into text
    with translate_manager.tokenizer.as_target_tokenizer():
        generated_tokens = translate_manager.tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    # Postprocess the translations, including entity replacement
    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    return TranslationResponse(translations=translations)

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...), language: str = Query(..., enum=list(asr_manager.model_language.keys()))):
    start_time = time()
    try:
        # Check file extension
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["wav", "mp3"]:
            logging.warning(f"Unsupported file format: {file_extension}")
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a WAV or MP3 file.")

        # Read the file content
        file_content = await file.read()

        # Convert MP3 to WAV if necessary
        if file_extension == "mp3":
            audio = AudioSegment.from_mp3(io.BytesIO(file_content))
        else:
            audio = AudioSegment.from_wav(io.BytesIO(file_content))

        # Check the sample rate of the WAV file
        sample_rate = audio.frame_rate

        # Convert WAV to the required format using ffmpeg if necessary
        if sample_rate != 16000:
            audio = audio.set_frame_rate(16000).set_channels(1)

        # Export the audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            audio.export(tmp_file.name, format="wav")
            tmp_file_path = tmp_file.name

        # Split the audio if necessary
        chunk_file_paths = asr_manager.split_audio(tmp_file_path)

        try:
            # Transcribe the audio
            language_id = asr_manager.model_language.get(language, asr_manager.default_language)

            if language_id != asr_manager.default_language:
                asr_manager.model = asr_manager.load_model(language_id)
                asr_manager.default_language = language_id

            asr_manager.model.cur_decoder = "rnnt"

            rnnt_texts = asr_manager.model.transcribe(chunk_file_paths, batch_size=1, language_id=language_id)

            # Flatten the list of transcriptions
            rnnt_text = " ".join([text for sublist in rnnt_texts for text in sublist])

            end_time = time()
            logging.info(f"Transcription completed in {end_time - start_time:.2f} seconds")
            return JSONResponse(content={"text": rnnt_text})
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg conversion failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {str(e)}")
        except Exception as e:
            logging.error(f"An error occurred during processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
        finally:
            # Clean up temporary files
            for chunk_file_path in chunk_file_paths:
                if os.path.exists(chunk_file_path):
                    os.remove(chunk_file_path)
    except HTTPException as e:
        logging.error(f"HTTPException: {str(e)}")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/transcribe_batch/", response_model=BatchTranscriptionResponse)
async def transcribe_audio_batch(files: List[UploadFile] = File(...), language: str = Query(..., enum=list(asr_manager.model_language.keys()))):
    start_time = time()
    tmp_file_paths = []
    transcriptions = []
    try:
        for file in files:
            # Check file extension
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in ["wav", "mp3"]:
                logging.warning(f"Unsupported file format: {file_extension}")
                raise HTTPException(status_code=400, detail="Unsupported file format. Please upload WAV or MP3 files.")

            # Read the file content
            file_content = await file.read()

            # Convert MP3 to WAV if necessary
            if file_extension == "mp3":
                audio = AudioSegment.from_mp3(io.BytesIO(file_content))
            else:
                audio = AudioSegment.from_wav(io.BytesIO(file_content))

            # Check the sample rate of the WAV file
            sample_rate = audio.frame_rate

            # Convert WAV to the required format using ffmpeg if necessary
            if sample_rate != 16000:
                audio = audio.set_frame_rate(16000).set_channels(1)

            # Export the audio to a temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                audio.export(tmp_file.name, format="wav")
                tmp_file_path = tmp_file.name

            # Split the audio if necessary
            chunk_file_paths = asr_manager.split_audio(tmp_file_path)
            tmp_file_paths.extend(chunk_file_paths)

        logging.info(f"Temporary file paths: {tmp_file_paths}")
        try:
            # Transcribe the audio files in batch
            language_id = asr_manager.model_language.get(language, asr_manager.default_language)

            if language_id != asr_manager.default_language:
                asr_manager.model = asr_manager.load_model(language_id)
                asr_manager.default_language = language_id

            asr_manager.model.cur_decoder = "rnnt"

            rnnt_texts = asr_manager.model.transcribe(tmp_file_paths, batch_size=len(files), language_id=language_id)

            logging.info(f"Raw transcriptions from model: {rnnt_texts}")
            end_time = time()
            logging.info(f"Transcription completed in {end_time - start_time:.2f} seconds")

            # Flatten the list of transcriptions
            transcriptions = [text for sublist in rnnt_texts for text in sublist]
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg conversion failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {str(e)}")
        except Exception as e:
            logging.error(f"An error occurred during processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
        finally:
            # Clean up temporary files
            for tmp_file_path in tmp_file_paths:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
    except HTTPException as e:
        logging.error(f"HTTPException: {str(e)}")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    return JSONResponse(content={"transcriptions": transcriptions})

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Translation Server")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type to run the model on (cuda or cpu).")
    parser.add_argument("--use_distilled", action="store_true", help="Use distilled models instead of base models.")
    parser.add_argument("--is_lazy_loading", action="store_true", help="Enable lazy loading of models.")
    parser.add_argument("--asr_language", type=str, default="kn", help="Default language for the ASR model.")
    return parser.parse_args()

# Run the server using Uvicorn
if __name__ == "__main__":
    args = parse_args()
    device_type = args.device
    use_distilled = args.use_distilled
    is_lazy_loading = args.is_lazy_loading

    # Initialize the model managers
    asr_manager = ASRModelManager(default_language=args.asr_language, device_type=device_type)
    model_manager = ModelManager(device_type, use_distilled, is_lazy_loading)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")