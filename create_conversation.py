import os
import yaml
import tempfile
import numpy as np
import pandas as pd
import soundfile as sf
import argparse
from pathlib import Path
from datasets import load_dataset
from cached_path import cached_path
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
)


def get_reference_audio(speaker_id, dataset):
    """Get reference audio for a specific speaker"""
    for item in dataset:
        if item['filepath'] == speaker_id:
            return item['audio']['array']
    return None

def assign_speaker_voices(speakers, dataset):
    """Assign a unique reference voice to each speaker"""
    speaker_voices = {}
    speaker_ids = list(set(speakers))
    
    # Get all available reference audios from the dataset
    available_references = []
    for item in dataset:
        available_references.append(item['audio']['array'])
    
    # Assign references in a round-robin fashion
    for i, speaker in enumerate(speaker_ids):
        if available_references:  # Make sure we have references available
            ref_audio = available_references[i % len(available_references)]
            speaker_voices[speaker] = ref_audio
            # print(f"Assigned reference audio to speaker: {speaker}")
    
    return speaker_voices

def generate_conversation_audio(dialog_data, output_dir, model, vocoder, mel_spec_type="vocos"):
    """Generate audio for each turn in the conversation"""
    print(f"Creating output directory at: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    speakers = dialog_data['Speaker'].unique()
    
    print("Loading BR-Speech dataset...")
    dataset = load_dataset("freds0/BRSpeech-TTS-Leni", split="test")
    
    print("Assigning speaker voices...")
    speaker_voices = assign_speaker_voices(speakers, dataset)
    
    generated_count = 0
    max_audios = 1000

    print("Starting audio generation for each dialog turn...")
    for idx, row in dialog_data.iterrows():
        if generated_count <= max_audios:           
            speaker = row['Speaker']
            text = row['Translated_Sentence']
            dialog_id = row['Dialog']
            turn = row['Turn']
            
            print(f"Generating audio for Dialog {dialog_id}, Turn {turn}, Speaker {speaker}")
            ref_audio = speaker_voices[speaker]

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_ref_file:
                temp_ref_path = temp_ref_file.name
                sf.write(temp_ref_path, ref_audio, 16000)  # Assuming 16kHz sample rate
        
            try:
                # Generate output filename
                output_file = f"dialog_{dialog_id}_turn_{turn}_{speaker}.wav"
                output_path = os.path.join(output_dir, output_file)
                
                # Generate audio
                audio, final_sample_rate, _ = infer_process(
                    temp_ref_path,  # Using the temporary file path
                    text,
                    text,
                    model,
                    vocoder,
                    mel_spec_type=mel_spec_type,
                    speed=1.0
                )
                
                # Save audio
                sf.write(output_path, audio, final_sample_rate)
                print(f"Generated audio for {output_file}")
                
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_ref_path):
                    os.unlink(temp_ref_path)

def load_config(config_path):
    """Load configuration from YAML file"""
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate conversation audio from dialog data")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated audio files")
    args = parser.parse_args()
    
    print("Parsing arguments...")
    config = load_config(args.config)
    
    print(f"Loading dialog data from: {config['dataset']['dialog_data_path']}")
    dialog_data = pd.read_parquet(config['dataset']['dialog_data_path'])
    
    print("Loading model and vocoder...")
    model_cls = DiT
    model_cfg = config['model']['config']
    repo_name = config['model']['repo']
    exp_name = config['model']['name']
    ckpt_step = config['model']['checkpoint_step']
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
    
    print(f"Loading model from checkpoint: {ckpt_file}")
    model = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=config['audio']['mel_spec_type'])
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False)
    
    print("Beginning conversation audio generation...")
    generate_conversation_audio(
        dialog_data, 
        args.output_dir, 
        model, 
        vocoder,
        mel_spec_type=config['audio']['mel_spec_type']
    )
    print("All audio generated successfully.")

if __name__ == "__main__":
    main() 