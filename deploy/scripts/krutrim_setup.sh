sudo apt-get update -y
sudo apt-get upgrade -y

sudo apt-get install -y python3-venv
sudo apt-get install -y python3-pip
sudo apt-get install -y ffmpeg
sudo apt install net-tools -y


python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

#cd src/asr_indic_server
#python src/asr_api.py


huggingface-cli download ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large

huggingface-cli download ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large

huggingface-cli download ai4bharat/indictrans2-indic-en-dist-200M

huggingface-cli download ai4bharat/indictrans2-en-indic-dist-200M

huggingface-cli download ai4bharat/indictrans2-indic-indic-dist-320M

huggingface_cli download ai4bharat/indic-parler-tts

huggingface_cli download vikhyatk/moondream2

#huggingface_cli download mistralai/Pixtral-12B-2409

python src/india_all_api.py --host 0.0.0.0 --port 8888 --device cuda --use_distilled --is_lazy_loading

