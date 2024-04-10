import os
from pydub import AudioSegment
from pydub.utils import make_chunks

def main(seconds, audio_file_name, save_path):
    def chunk_and_save(file):
        audio = AudioSegment.from_file(file)
        length = seconds * 1000 # this is in miliseconds
        chunks = make_chunks(audio, length)
        names = []
        for i, chunk in enumerate(chunks):
            _name = file.split("/")[-1]
            name = "{}_{}".format(i, _name)
            wav_path = os.path.join(save_path, name)
            chunk.export(wav_path, format="wav")
            names.append(wav_path)
        return names

    chunk_and_save(audio_file_name)

if __name__ == "__main__":
    # Input values directly
    seconds = 3  # Example: 10 seconds
    audio_file_name = "VoiceAssistant/wakeword/scripts/data/wakewords/noise.wav"  # Example: path to the audio file
    save_path = "VoiceAssistant/wakeword/scripts/data/0"  # Example: path to save the chunks
    
    main(seconds, audio_file_name, save_path)
