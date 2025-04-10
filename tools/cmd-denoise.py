import os
import argparse
import traceback
import soundfile as sf
import pyloudnorm as pyln

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm

path_denoise = "tools/denoise-model/speech_frcrn_ans_cirm_16k"
path_denoise = path_denoise if os.path.exists(path_denoise) else "damo/speech_frcrn_ans_cirm_16k"
ans = pipeline(Tasks.acoustic_noise_suppression, model=path_denoise)


def execute_denoise(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # print(input_folder)
    # print(list(os.listdir(input_folder).sort()))
    for name in tqdm(os.listdir(input_folder)):
        try:
            ans("%s/%s" % (input_folder, name), output_path="%s/%s" % (output_folder, name))
            match_loudness(f"{output_folder}/{name}")
        except:
            traceback.print_exc()


def match_loudness(audio_file):
    audio, rate = sf.read(audio_file)

    meter = pyln.Meter(rate)
    peak_audio = pyln.normalize.peak(audio, -1.0)
    _loudness = meter.integrated_loudness(peak_audio)
    norm_audio = pyln.normalize.loudness(peak_audio, _loudness, -23.0)

    # Save the normalized audio back to the target file (overwrite)
    sf.write(audio_file, norm_audio, rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_folder", type=str, required=True, help="Path to the folder containing WAV files."
    )
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Output folder to store transcriptions.")
    parser.add_argument(
        "-p", "--precision", type=str, default="float16", choices=["float16", "float32"], help="fp16 or fp32"
    )  # 还没接入
    cmd = parser.parse_args()
    execute_denoise(
        input_folder=cmd.input_folder,
        output_folder=cmd.output_folder,
    )
