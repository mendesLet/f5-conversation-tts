# F5-TTS Conversation Generator

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r f5/requirements.txt
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
./f5/run_tts.sh --config f5/config.yaml --output generated_audio
```