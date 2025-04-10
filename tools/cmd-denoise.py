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
            match_loudness(f"{input_folder}/{name}", f"{output_folder}/{name}")
        except:
            traceback.print_exc()


def match_loudness(ref_file, target_file):
    # Load the reference audio (original, unprocessed)
    ref_audio, rate = sf.read(ref_file)

    # Load the target audio (denoised version)
    target_audio, _ = sf.read(target_file)

    # Create a loudness meter using the sample rate
    meter = pyln.Meter(rate)

    # Measure integrated loudness (LUFS) of both files
    ref_loudness = meter.integrated_loudness(ref_audio)
    target_loudness = meter.integrated_loudness(target_audio)

    # Normalize target audio to match reference loudness
    target_audio_normalized = pyln.normalize.loudness(target_audio, target_loudness, ref_loudness)

    # Save the normalized audio back to the target file (overwrite)
    sf.write(target_file, target_audio_normalized, rate)


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
