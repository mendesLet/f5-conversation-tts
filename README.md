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
model:
  name: "F5TTS_Base"
  repo: "F5-TTS"
  checkpoint_step: 1200000
  # ... other settings
```

2. Run the script:
```bash
./run.sh --config config.yaml --output generated_audio
```
