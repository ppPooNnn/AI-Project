# Quick Start Guide

## 🚀 How to Run the Smart Camera Voice Assistant

### Method 1: Using the Batch File (Windows)
1. Double-click `run_assistant.bat`
2. Wait for packages to install
3. Follow the voice prompts

### Method 2: Using the Shell Script (macOS/Linux)
1. Open terminal
2. Run: `./run_assistant.sh`
3. Follow the voice prompts

### Method 3: Manual Setup
1. Install packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Test the system:
   ```bash
   python test_camera.py
   ```

3. Run the assistant:
   ```bash
   python smart_camera_assistant.py
   ```

## 🎤 Voice Commands

| Say This | What It Does |
|----------|--------------|
| "describe" | Describes what the camera sees |
| "take picture" | Same as describe |
| "help" | Shows available commands |
| "quit" | Exits the program |

## 🔧 Troubleshooting

### Camera Not Working?
- Check if camera is connected
- Close other apps using the camera
- Try a different camera

### No Voice Output?
- Check speaker/headphone connection
- Make sure volume is up
- Test with: `python test_camera.py`

### Can't Hear Commands?
- Check microphone connection
- Speak clearly and wait for "Listening..." message
- Make sure no other apps are using the microphone

## 📁 Required Files

Make sure these files exist:
- `input/features.pkl`
- `input/captions.txt`

## 🎯 Quick Test

Run this to test everything:
```bash
python test_camera.py
```

If all tests pass, you're ready to use the assistant!
