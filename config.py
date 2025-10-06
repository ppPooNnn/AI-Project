"""
Configuration file for Smart Camera Voice Assistant
"""

# Camera settings
CAMERA_INDICES = [0, 1, 2]  # Try these camera indices in order
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Voice settings
SPEECH_RATE = 160  # Words per minute
SPEECH_VOLUME = 0.9  # Volume level (0.0 to 1.0)
VOICE_TIMEOUT = 5  # Seconds to wait for voice input
PHRASE_TIME_LIMIT = 10  # Maximum length of voice command

# Model settings
MODEL_INPUT_SIZE = (224, 224)  # Input image size for VGG16
FEATURE_DIMENSION = 512  # VGG16 feature dimension

# File paths
FEATURES_FILE = 'input/features.pkl'
CAPTIONS_FILE = 'input/captions.txt'
MODEL_FILE = 'best_model.h5'  # If you have a trained model

# Voice commands
VOICE_COMMANDS = {
    'describe': ['describe', 'take picture', 'picture', 'camera'],
    'help': ['help', 'commands', 'what can you do'],
    'quit': ['quit', 'exit', 'goodbye', 'bye', 'stop']
}

# English voice keywords to look for
ENGLISH_VOICE_KEYWORDS = ['english', 'us', 'en', 'zira', 'hazel']

# Debug settings
DEBUG_MODE = False  # Set to True for verbose output
SAVE_CAPTURED_IMAGES = False  # Set to True to save captured images
TEMP_IMAGE_DIR = 'temp_images'  # Directory for temporary images
