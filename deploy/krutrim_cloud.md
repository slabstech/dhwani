Steps for Testing - Benchmarks

1. Run scripts/krutrim_setup.sh

2. run model_download.sh

3.
Run model_download

4. start fastapi-server 

python src/india_all_api.py --host 0.0.0.0 --port 8888 --device cuda --use_distilled --is_lazy_loading

5. Run health check, curl commands for all avaialbe api endpoints


tts - 

curl -X 'POST' \
  'https://<<Generated-session>>-8888.blr-1.aipod.olakrutrimsvc.com/v1/audio/speech' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": "ನಿಮ್ಮ ಇನ್‌ಪುಟ್ ಪಠ್ಯವನ್ನು ಇಲ್ಲಿ ಸೇರಿಸಿ",
  "voice": "Female speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality.",
  "model": "ai4bharat/indic-parler-tts",
  "response_format": "mp3",
  "speed": 1
}' -o kannada_audio.mp3


asr - single
curl -X 'POST'   'https://<<Generated-session>>-8888.blr-1.aipod.olakrutrimsvc.com/transcribe/?language=kannada'   -H 'accept: application/json'   -H 'Content-Type: multipart/form-data'   -F 'file=@kannada_sample_2.wav;type=audio/x-wav'


asr- batch

curl -X 'POST' \
  'https://<<Generated-session>>-8888.blr-1.aipod.olakrutrimsvc.com/transcribe_batch/?language=kannada' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@kannada_sample_2.wav;type=audio/x-wav' \
  -F 'files=@kannada_sample_1.wav;type=audio/x-wav' \
  -F 'files=@kannada_sample_3.wav;type=audio/x-wav' \
  -F 'files=@kannada_sample_4.wav;type=audio/x-wav'

