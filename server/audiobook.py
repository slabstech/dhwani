from elevenlabs import ElevenLabs
import os

# Simulate MCP Server 1: Manuscript Resource
class ManuscriptServer:
    def get_chapter(self, chapter_num):
        with open("manuscript.txt", "r") as f:
            chapters = f.read().split("\n\n")  # Assuming double newline separates chapters
        return chapters[chapter_num]

# Simulate MCP Server 2: ElevenLabs TTS Tool
class ElevenLabsServer:
    def __init__(self):
        self.client = ElevenLabs(api_key="YOUR_ELEVENLABS_API_KEY")

    def generate_audio(self, text, output_file):
        audio = self.client.generate(text=text, voice="Rachel", model="eleven_multilingual_v2")
        with open(output_file, "wb") as f:
            f.write(audio)

# MCP Client: Orchestrates the audiobook creation
class AudiobookClient:
    def __init__(self):
        self.manuscript = ManuscriptServer()
        self.tts = ElevenLabsServer()

    def create_audiobook(self, num_chapters):
        for i in range(num_chapters):
            text = self.manuscript.get_chapter(i)
            output_file = f"chapter_{i+1}.mp3"
            self.tts.generate_audio(text, output_file)
            print(f"Generated {output_file}")

# Run it
if __name__ == "__main__":
    client = AudiobookClient()
    client.create_audiobook(num_chapters=3)  # Adjust based on your manuscript