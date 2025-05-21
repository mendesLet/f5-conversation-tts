import os
import yaml
import numpy as np
import pandas as pd
import soundfile as sf
import argparse
from pathlib import Path
from datasets import load_dataset
from cached_path import cached_path
from f5_tts.model import DiT, UNetT
from huggingface_hub import hf_hub_download
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)


def load_br_speech_references():
    dataset = load_dataset("freds0/BRSpeech-TTS-Leni", split="train")

    reference_files = {}
    for item in dataset:
        speaker_id = item['filepath'] 
        if speaker_id not in reference_files:
            reference_files[speaker_id] = item['audio']['array']

    return reference_files

def assign_speaker_voices(speakers, reference_files):
    """Assign a unique reference voice to each speaker"""
    speaker_voices = {}
    reference_list = list(reference_files.values())
    
    for i, speaker in enumerate(set(speakers)):
        # Assign a reference file to each speaker
        speaker_voices[speaker] = reference_list[i % len(reference_list)]
    return speaker_voices

def generate_conversation_audio(dialog_data, output_dir, model, vocoder, mel_spec_type="vocos"):
    """Generate audio for each turn in the conversation"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique speakers
    speakers = dialog_data['Speaker'].unique()
    
    # Load BR-Speech references
    reference_files = load_br_speech_references()
    
    # Assign voices to speakers
    speaker_voices = assign_speaker_voices(speakers, reference_files)
    
    # Generate audio for each turn
    for idx, row in dialog_data.iterrows():
        speaker = row['Speaker']
        text = row['Translated_Sentence']
        dialog_id = row['Dialog']
        turn = row['Turn']
        
        # Get reference audio for this speaker
        ref_audio = speaker_voices[speaker]
        
        # Generate output filename
        output_file = f"dialog_{dialog_id}_turn_{turn}_{speaker}.wav"
        output_path = os.path.join(output_dir, output_file)
        
        # Generate audio
        audio, final_sample_rate, _ = infer_process(
            ref_audio,
            text,  # Using the same text as reference (you might want to change this)
            text,
            model,
            vocoder,
            mel_spec_type=mel_spec_type,
            speed=1.0
        )
        
        # Save audio
        sf.write(output_path, audio, final_sample_rate)
        print(f"Generated audio for {output_file}")

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate conversation audio from dialog data")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated audio files")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load dialog data from Hugging Face
    dialog_data = pd.read_parquet(config['dataset']['dialog_data_path'])
    
    # Load model and vocoder (using F5-TTS)
    model_cls = DiT
    model_cfg = config['model']['config']
    repo_name = config['model']['repo']
    exp_name = config['model']['name']
    ckpt_step = config['model']['checkpoint_step']
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
    
    model = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=config['audio']['mel_spec_type'])
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False)
    
    # Generate conversation audio
    generate_conversation_audio(
        dialog_data, 
        args.output_dir, 
        model, 
        vocoder,
        mel_spec_type=config['audio']['mel_spec_type']
    )

if __name__ == "__main__":
    main() 