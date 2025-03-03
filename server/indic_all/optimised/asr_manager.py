import torch
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import os
import tempfile
import subprocess
import logging

class ASRModelManager:
    def __init__(self, default_language="kn", device_type="cuda"):
        self.default_language = default_language
        self.device_type = device_type
        self.model_language = {
            "kannada": "kn",
            "hindi": "hi",
            "malayalam": "ml",
            "assamese": "as",
            "bengali": "bn",
            "bodo": "brx",
            "dogri": "doi",
            "gujarati": "gu",
            "kashmiri": "ks",
            "konkani": "kok",
            "maithili": "mai",
            "manipuri": "mni",
            "marathi": "mr",
            "nepali": "ne",
            "odia": "or",
            "punjabi": "pa",
            "sanskrit": "sa",
            "santali": "sat",
            "sindhi": "sd",
            "tamil": "ta",
            "telugu": "te",
            "urdu": "ur"
        }
        self.config_models = {
            "as": "ai4bharat/indicconformer_stt_as_hybrid_rnnt_large",
            "bn": "ai4bharat/indicconformer_stt_bn_hybrid_rnnt_large",
            "brx": "ai4bharat/indicconformer_stt_brx_hybrid_rnnt_large",
            "doi": "ai4bharat/indicconformer_stt_doi_hybrid_rnnt_large",
            "gu": "ai4bharat/indicconformer_stt_gu_hybrid_rnnt_large",
            "hi": "ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large",
            "kn": "ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large",
            "ks": "ai4bharat/indicconformer_stt_ks_hybrid_rnnt_large",
            "kok": "ai4bharat/indicconformer_stt_kok_hybrid_rnnt_large",
            "mai": "ai4bharat/indicconformer_stt_mai_hybrid_rnnt_large",
            "ml": "ai4bharat/indicconformer_stt_ml_hybrid_rnnt_large",
            "mni": "ai4bharat/indicconformer_stt_mni_hybrid_rnnt_large",
            "mr": "ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large",
            "ne": "ai4bharat/indicconformer_stt_ne_hybrid_rnnt_large",
            "or": "ai4bharat/indicconformer_stt_or_hybrid_rnnt_large",
            "pa": "ai4bharat/indicconformer_stt_pa_hybrid_rnnt_large",
            "sa": "ai4bharat/indicconformer_stt_sa_hybrid_rnnt_large",
            "sat": "ai4bharat/indicconformer_stt_sat_hybrid_rnnt_large",
            "sd": "ai4bharat/indicconformer_stt_sd_hybrid_rnnt_large",
            "ta": "ai4bharat/indicconformer_stt_ta_hybrid_rnnt_large",
            "te": "ai4bharat/indicconformer_stt_te_hybrid_rnnt_large",
            "ur": "ai4bharat/indicconformer_stt_ur_hybrid_rnnt_large"
        }
        self.model = self.load_model(self.default_language)

    def load_model(self, language_id="kn"):
        model_name = self.config_models.get(language_id, self.config_models["kn"])

        model = nemo_asr.models.ASRModel.from_pretrained(model_name)

        device = torch.device(self.device_type if torch.cuda.is_available() and self.device_type == "cuda" else "cpu")
        model.freeze() # inference mode
        model = model.to(device) # transfer model to device

        return model

    def split_audio(self, file_path, chunk_duration_ms=15000):
        """
        Splits an audio file into chunks of specified duration if the audio duration exceeds the chunk duration.

        :param file_path: Path to the audio file.
        :param chunk_duration_ms: Duration of each chunk in milliseconds (default is 15000 ms or 15 seconds).
        """
        # Load the audio file
        audio = AudioSegment.from_file(file_path)

        # Get the duration of the audio in milliseconds
        duration_ms = len(audio)

        # Check if the duration is more than the specified chunk duration
        if duration_ms > chunk_duration_ms:
            # Calculate the number of chunks needed
            num_chunks = duration_ms // chunk_duration_ms
            if duration_ms % chunk_duration_ms != 0:
                num_chunks += 1

            # Split the audio into chunks
            chunks = [audio[i*chunk_duration_ms:(i+1)*chunk_duration_ms] for i in range(num_chunks)]

            # Create a directory to save the chunks
            output_dir = "audio_chunks"
            os.makedirs(output_dir, exist_ok=True)

            # Export each chunk to separate files
            chunk_file_paths = []
            for i, chunk in enumerate(chunks):
                chunk_file_path = os.path.join(output_dir, f"chunk_{i}.wav")
                chunk.export(chunk_file_path, format="wav")
                chunk_file_paths.append(chunk_file_path)
                print(f"Chunk {i} exported successfully to {chunk_file_path}.")

            return chunk_file_paths
        else:
            return [file_path]