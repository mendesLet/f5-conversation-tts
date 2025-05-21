#!/bin/bash

# run_tts.sh

# Default values
CONFIG_FILE="config.yaml"
OUTPUT_DIR="generated_audio"

# Function to display help
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -c, --config FILE    Specify config file (default: config.yaml)"
    echo "  -o, --output DIR     Specify output directory (default: generated_audio)"
    echo "  -h, --help          Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script
echo "Starting TTS generation..."
echo "Using config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"

python create_conversation.py \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR"

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "TTS generation completed successfully!"
else
    echo "Error: TTS generation failed!"
    exit 1
fi