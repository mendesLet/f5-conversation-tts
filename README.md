# F5-TTS Conversation Generator

## Installation

1. Clone the repository:
```bash
git clone git@github.com:mendesLet/f5-conversation-tts.git
cd f5-conversation-tts
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure your settings in `config.yaml`:
```yaml
# Example configuration
dataset:
  dialog_data_path: "hf://datasets/AKCIT-Audio/LIGHT_transcriptions/data/train-00000-of-00001.parquet"
  # ... other settings
```

2. Run the script:
```bash
python conversational_agent.py --config config.yaml --output_dir generated_audio
```
