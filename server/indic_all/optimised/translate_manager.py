import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from typing import Dict, Any
from logger import logger

class TranslateManager:
    def __init__(self, src_lang, tgt_lang, device_type="cuda", use_distilled=True):
        self.device_type = device_type
        self.tokenizer, self.model = self.initialize_model(src_lang, tgt_lang, use_distilled)

    def initialize_model(self, src_lang, tgt_lang, use_distilled):
        # Determine the model name based on the source and target languages and the model type
        if src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            model_name = "ai4bharat/indictrans2-en-indic-dist-200M" if use_distilled else "ai4bharat/indictrans2-en-indic-1B"
        elif not src_lang.startswith("eng") and tgt_lang.startswith("eng"):
            model_name = "ai4bharat/indictrans2-indic-en-dist-200M" if use_distilled else "ai4bharat/indictrans2-indic-en-1B"
        elif not src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            model_name = "ai4bharat/indictrans2-indic-indic-dist-320M" if use_distilled else "ai4bharat/indictrans2-indic-indic-1B"
        else:
            raise ValueError("Invalid language combination: English to English translation is not supported.")
        # Now model_name contains the correct model based on the source and target languages
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16, # performance might slightly vary for bfloat16
            attn_implementation="flash_attention_2"
        ).to(self.device_type)
        return tokenizer, model

class ModelManager:
    def __init__(self, device_type="cuda", use_distilled=True, is_lazy_loading=False):
        self.models: Dict[str, TranslateManager] = {}
        self.device_type = device_type
        self.use_distilled = use_distilled
        self.is_lazy_loading = is_lazy_loading
        if not is_lazy_loading:
            self.preload_models()

    def preload_models(self):
        # Preload all models at startup
        self.models['eng_indic'] = TranslateManager('eng_Latn', 'kan_Knda', self.device_type, self.use_distilled)
        self.models['indic_eng'] = TranslateManager('kan_Knda', 'eng_Latn', self.device_type, self.use_distilled)
        self.models['indic_indic'] = TranslateManager('kan_Knda', 'hin_Deva', self.device_type, self.use_distilled)

    def get_model(self, src_lang, tgt_lang) -> TranslateManager:
        if src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            key = 'eng_indic'
        elif not src_lang.startswith("eng") and tgt_lang.startswith("eng"):
            key = 'indic_eng'
        elif not src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            key = 'indic_indic'
        else:
            raise ValueError("Invalid language combination: English to English translation is not supported.")
        if key not in self.models:
            if self.is_lazy_loading:
                if key == 'eng_indic':
                    self.models[key] = TranslateManager('eng_Latn', 'kan_Knda', self.device_type, self.use_distilled)
                elif key == 'indic_eng':
                    self.models[key] = TranslateManager('kan_Knda', 'eng_Latn', self.device_type, self.use_distilled)
                elif key == 'indic_indic':
                    self.models[key] = TranslateManager('kan_Knda', 'hin_Deva', self.device_type, self.use_distilled)
            else:
                raise ValueError(f"Model for {key} is not preloaded and lazy loading is disabled.")
        return self.models[key]

# Initialize IndicProcessor
ip = IndicProcessor(inference=True)