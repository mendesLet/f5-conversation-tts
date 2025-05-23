import os
import yaml
import tempfile
import numpy as np
import pandas as pd
import soundfile as sf
import argparse
import json
import logging
from pathlib import Path
from datasets import load_dataset
from cached_path import cached_path
from AgentF5TTSChunk import AgentF5TTS

# Default model configuration for Brazilian Portuguese
DEFAULT_TTS_MODEL = "F5-TTS-BR"
DEFAULT_TTS_MODEL_CFG = [
    "hf://ModelsLab/F5-tts-brazilian/Brazilian_Portuguese/model_2600000.pt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

logging.basicConfig(level=logging.INFO)

def get_reference_audio(speaker_id, dataset):
    """Get reference audio and text for a specific speaker"""
    for item in dataset:
        if item['filepath'] == speaker_id:
            return item['audio']['array'], item['text']
    return None, None

def assign_speaker_voices(speakers, dataset):
    """Assign a unique reference voice and text to each speaker"""
    speaker_voices = {}
    speaker_texts = {}
    speaker_ids = list(set(speakers))
    
    # Get all available reference audios and texts from the dataset
    available_references = []
    available_texts = []
    for item in dataset:
        available_references.append(item['audio']['array'])
        available_texts.append(item['text'])
    
    # Assign references in a round-robin fashion
    for i, speaker in enumerate(speaker_ids):
        if available_references:  # Make sure we have references available
            ref_audio = available_references[i % len(available_references)]
            ref_text = available_texts[i % len(available_texts)]
            speaker_voices[speaker] = ref_audio
            speaker_texts[speaker] = ref_text
    
    return speaker_voices, speaker_texts

def generate_conversation_audio(dialog_data, output_dir, agent):
    """Generate audio for each turn in the conversation using AgentF5TTS"""
    logging.info(f"Creating output directory at: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    speakers = dialog_data['Speaker'].unique()
    
    logging.info("Loading BR-Speech dataset...")
    dataset = load_dataset("freds0/BRSpeech-TTS-Leni", split="test")
    
    logging.info("Assigning speaker voices...")
    speaker_voices, speaker_texts = assign_speaker_voices(speakers, dataset)
    
    generated_count = 0
    max_audios = 1000

    logging.info("Starting audio generation for each dialog turn...")
    for idx, row in dialog_data.iterrows():
        if generated_count <= max_audios:           
            speaker = row['Speaker']
            gen_text = row['Translated_Sentence']  # This is the text we want to generate
            ref_text = speaker_texts[speaker]      # This is the reference text from the dataset
            dialog_id = row['Dialog']
            turn = row['Turn']
            
            logging.info(f"Generating audio for Dialog {dialog_id}, Turn {turn}, Speaker {speaker}")
            ref_audio = speaker_voices[speaker]

            # Create temporary files for reference audio and text
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_ref_file:
                temp_ref_path = temp_ref_file.name
                sf.write(temp_ref_path, ref_audio, 24000)  # Using target sample rate of 24000 Hz
            
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_text_file:
                temp_text_path = temp_text_file.name
                with open(temp_text_path, 'w', encoding='utf-8') as f:
                    f.write(gen_text)
        
            try:
                # Generate output filename
                output_file = f"dialog_{dialog_id}_turn_{turn}_{speaker}.wav"
                output_path = os.path.join(output_dir, output_file)
                
                # Generate audio using AgentF5TTS
                agent.generate_speech(
                    text_file=temp_text_path,  # Using the text file with the sentence to generate
                    output_audio_file=output_path,
                    ref_audio=temp_ref_path,  # Using the reference audio file
                    ref_text=ref_text,  # Using the reference text
                    convert_to_mp3=False
                )
                
                logging.info(f"Generated audio for {output_file}")
                generated_count += 1
                
            finally:
                # Clean up temporary files
                if os.path.exists(temp_ref_path):
                    os.unlink(temp_ref_path)
                if os.path.exists(temp_text_path):
                    os.unlink(temp_text_path)

def load_config(config_path):
    """Load configuration from YAML file"""
    logging.info(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate conversation audio from dialog data using AgentF5TTS")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated audio files")
    args = parser.parse_args()
    
    logging.info("Parsing arguments...")
    config = load_config(args.config)
    
    logging.info(f"Loading dialog data from: {config['dataset']['dialog_data_path']}")
    dialog_data = pd.read_parquet(config['dataset']['dialog_data_path'])
    
    logging.info("Loading model...")
    # Load model using the Brazilian Portuguese checkpoint
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    
    logging.info(f"Loading model from checkpoint: {ckpt_path}")
    agent = AgentF5TTS(
        ckpt_file=ckpt_path,
        vocoder_name="vocos",
        delay=0,
        device="cuda"  # You can change this to "cpu" or "mps" based on your system
    )
    
    logging.info("Beginning conversation audio generation...")
    generate_conversation_audio(
        dialog_data, 
        args.output_dir, 
        agent
    )
    logging.info("All audio generated successfully.")

if __name__ == "__main__":
    main()