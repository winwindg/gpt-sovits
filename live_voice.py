import json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import shutil
import sys
import time
from typing import Generator

import librosa
import pynvml
import requests
import uvicorn
import torch

import config
import rewrite

current_dir = os.getcwd()
sys.path.append(current_dir)
sys.path.append("%s/GPT_SoVITS" % current_dir)

import subprocess
import wave
import numpy as np
import soundfile as sf
from fastapi import Response
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from io import BytesIO
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

local_dir = "./speakers"
share_dir = "/mnt/qnap/speakers"
public_url = "https://meiyins.oss-cn-hangzhou.aliyuncs.com/speakers"

cut_method_names = get_cut_method_names()
v2_languages: list = ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"]
tts_pipelines: dict = {}
default_sample_rate: int = 16000
app = FastAPI()


def create_config(t2s_weights_path, vits_weights_path):
    tts_config = TTS_Config()
    tts_config.device = "cuda"
    tts_config.is_half = True
    tts_config.version = "v2"
    tts_config.bert_base_path = config.bert_path
    tts_config.cnhuhbert_base_path = config.cnhubert_path
    tts_config.sampling_rate = default_sample_rate
    tts_config.t2s_weights_path = t2s_weights_path
    tts_config.vits_weights_path = vits_weights_path
    return tts_config


def get_tts_pipeline(speaker_name: str):
    global tts_pipelines
    if speaker_name not in tts_pipelines:
        tts_config = create_config(get_speaker_file(speaker_name, "model.ckpt"), get_speaker_file(speaker_name, "model.pth"))
        tts_pipelines[speaker_name] = TTS(tts_config)

    return tts_pipelines[speaker_name]


def get_speaker_meta(speaker_name: str):
    speaker_meta_path = get_speaker_file(speaker_name, "meta.json")
    if os.path.exists(speaker_meta_path):
        with open(speaker_meta_path, 'r', encoding='utf-8') as file:
            meta_dict = json.load(file)
            meta_dict["ref_audio_path"] = get_speaker_file(speaker_name, "sample.wav")
            return meta_dict
    else:
        raise Exception(f"Speaker {speaker_name}'s meta does not exist")


def get_speaker_file(speaker_name: str, file_name: str):
    file_path = f"{speaker_name}/{file_name}"
    local_path = os.path.join(local_dir, file_path)
    parent_path = os.path.dirname(local_path)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)

    if os.path.exists(local_path):
        return local_path

    share_path = os.path.join(share_dir, file_path)
    if os.path.exists(share_path):
        logging.info(f"Copying speaker's {share_path} to {local_path}")
        shutil.copyfile(share_path, local_path)
        return local_path

    speaker_url = os.path.join(public_url, file_path)
    response = requests.head(speaker_url)
    if response.status_code == 200:
        return download_file(speaker_url, local_path)
    else:
        raise Exception(f"Speaker {speaker_name}'s data does not exist")


def download_file(url, file_path):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    start_time = time.time()
    tmp_file_path = f"{file_path}.tmp"
    response = requests.get(url, headers=headers, verify=False, allow_redirects=True)
    if response.status_code == 200:
        with open(tmp_file_path, 'wb') as file:
            file.write(response.content)
        logging.info(f"Download {url} to {file_path}, time={time.time() - start_time:.2f}s")
        move_file(tmp_file_path, file_path)
        return file_path
    else:
        raise Exception(f"Download {url} failed, status_code={response.status_code}")


def move_file(src_path, dest_path):
    if src_path != dest_path and os.path.exists(src_path) and not os.path.exists(dest_path):
        shutil.move(src_path, dest_path)


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = "zh"
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = "zh"
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 16
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    return_fragment: bool = False
    speaker_name: str = ""


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=default_sample_rate):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    sf.write(io_buffer, data, rate, format='wav')

    if rate != default_sample_rate:
        io_buffer.seek(0)
        audio_data, _ = sf.read(io_buffer)
        resampled_data = librosa.resample(audio_data, orig_sr=rate, target_sr=default_sample_rate)
        io_buffer.seek(0)
        io_buffer.truncate(0)
        sf.write(io_buffer, resampled_data, default_sample_rate, format='wav')

    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


def release_gpu_memory():
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()

    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage_percent = meminfo.used / meminfo.total * 100
        if usage_percent > 70:
            logging.info(f"GPU {i} memory usage is {usage_percent:.2f}%. Release memory...")
            torch.cuda.empty_cache()


async def tts_handle(req: dict):
    """
    Text to speech handler.

    Args:
        req (dict):
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker synthesis
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.
            }
    returns:
        StreamingResponse: audio stream response.
    """

    try:
        meda_dict = get_speaker_meta(req.get("speaker_name"))
        for key in meda_dict:
            if key in req:
                req[key] = meda_dict[key]

        req["text"] = rewrite.transcribe(req["text"])
        tts_pipeline = get_tts_pipeline(req.get("speaker_name"))
        tts_generator = tts_pipeline.run(req)
        media_type = req.get("media_type")

        if req.get("streaming_mode") or req.get("return_fragment"):
            def streaming_generator(tts_generator: Generator, media_type: str):
                if media_type == "wav":
                    yield wave_header_chunk()
                    media_type = "raw"
                for sr, chunk in tts_generator:
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()

            # _media_type = f"audio/{media_type}" if not (streaming_mode and media_type in ["wav", "raw"]) else f"audio/x-{media_type}"
            return StreamingResponse(streaming_generator(tts_generator, media_type, ), media_type=f"audio/{media_type}")
        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        logging.error("TTS error", e)
        return JSONResponse(status_code=500, content={"code": 500, "msg": str(e)})
    finally:
        release_gpu_memory()


@app.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    return await tts_handle(request.model_dump())


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=9880, workers=1)
