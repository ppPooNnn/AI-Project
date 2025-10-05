# Smart Camera Voice Assistant for the Blind

A real-time camera voice assistant that helps visually impaired users by describing what the camera sees in English.

## Features

- 🎥 **Real-time Camera Integration** - Connects to your camera automatically
- 🎤 **Voice Commands** - Control the system using voice commands
- 🔊 **English Speech Output** - Describes images in clear English
- 🤖 **AI-Powered** - Uses trained neural network for image description
- 📱 **Easy to Use** - Simple voice commands for operation

## Installation

1. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Make sure you have the required files:**
   - `input/features.pkl` - Pre-extracted image features
   - `input/captions.txt` - Image captions data
   - Trained model files (if available)

## Usage

### Quick Start

1. **Run the assistant:**
   ```bash
   python smart_camera_assistant.py
   ```

2. **Voice Commands:**
   - Say **"describe"** or **"take picture"** to describe current camera view
   - Say **"help"** to show available commands
   - Say **"quit"** or **"exit"** to exit the program

### Available Functions

```python
from smart_camera_assistant import SmartCameraAssistant

# Create assistant
assistant = SmartCameraAssistant()

# Test camera
assistant.test_camera()

# Quick test (capture and describe)
assistant.quick_test()

# Run full assistant
assistant.run_assistant()
```

## Camera Setup

The system will automatically try to connect to cameras in this order:
- Camera index 0 (default)
- Camera index 1
- Camera index 2

Make sure your camera is:
- Connected to your computer
- Not being used by other applications
- Working properly

## Voice Commands

| Command | Description |
|---------|-------------|
| "describe" | Describe current camera view |
| "take picture" | Same as describe |
| "help" | Show available commands |
| "quit" | Exit the program |
| "exit" | Exit the program |

## Troubleshooting

### Camera Issues
- **"No camera detected"**: Check camera connection and make sure no other apps are using it
- **"Unable to capture image"**: Try restarting the program or checking camera permissions

### Voice Issues
- **Speech not recognized**: Speak clearly and ensure microphone is working
- **No voice output**: Check speaker/headphone connection

### Model Issues
- **"Error loading model"**: Make sure all required files are in the correct location
- **"Unable to generate description"**: The AI model might need to be retrained

## Requirements

- Python 3.7+
- Webcam or external camera
- Microphone
- Speakers or headphones
- Windows/macOS/Linux

## File Structure

```
project/
├── smart_camera_assistant.py    # Main application
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── input/
    ├── features.pkl            # Image features
    └── captions.txt            # Image captions
```

## Support

If you encounter any issues:
1. Check that your camera and microphone are working
2. Ensure all dependencies are installed
3. Make sure required data files are present
4. Try running the test functions first

## License

This project is designed to help visually impaired users and is available for educational and assistive purposes.
