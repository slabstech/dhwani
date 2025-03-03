import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
import numpy as np
from tts_config import SPEED, ResponseFormat, config
from logger import logger
from contextlib import asynccontextmanager
import io
import soundfile as sf
from time import perf_counter
from collections import OrderedDict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if DEVICE != "cpu" else torch.float32

class TTSModelManager:
    def __init__(self):
        self.model_tokenizer: OrderedDict[
            str, tuple[ParlerTTSForConditionalGeneration, AutoTokenizer]
        ] = OrderedDict()

    def load_model(
        self, model_name: str
    ) -> tuple[ParlerTTSForConditionalGeneration, AutoTokenizer]:
        logger.debug(f"Loading {model_name}...")
        start = perf_counter()
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(
            DEVICE,
            dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
        logger.info(
            f"Loaded {model_name} and tokenizer in {perf_counter() - start:.2f} seconds"
        )
        return model, tokenizer, description_tokenizer

    def get_or_load_model(
        self, model_name: str
    ) -> tuple[ParlerTTSForConditionalGeneration, Any]:
        if model_name not in self.model_tokenizer:
            logger.info(f"Model {model_name} isn't already loaded")
            if len(self.model_tokenizer) == config.max_models:
                logger.info("Unloading the oldest loaded model")
                del self.model_tokenizer[next(iter(self.model_tokenizer))]
            self.model_tokenizer[model_name] = self.load_model(model_name)
        return self.model_tokenizer[model_name]

@asynccontextmanager
async def lifespan(_: FastAPI):
    if not config.lazy_load_model:
        tts_model_manager.get_or_load_model(config.model)
    yield