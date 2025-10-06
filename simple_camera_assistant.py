#!/usr/bin/env python3
"""
Simple Camera Assistant for the Blind
English-only system without speech recognition (to avoid FLAC issues)
"""

import os
import cv2
import pickle
import numpy as np
import tempfile
import time
from datetime import datetime

# Import TensorFlow and Keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Import text-to-speech only (no speech recognition)
import pyttsx3

class SimpleCameraAssistant:
    """
    Simple Camera Assistant - Text-to-Speech only
    Avoids FLAC issues by not using speech recognition
    """
    
    def __init__(self):
        print("Initializing Simple Camera Assistant...")
        
        # Initialize components
        self.camera = None
        self.model = None
        self.features = None
        self.tokenizer = None
        self.max_length = None
        self.mapping = None
        
        # Voice component (text-to-speech only)
        self.tts_engine = pyttsx3.init()
        
        # Setup voice
        self.setup_english_voice()
        
        # Load trained model and data
        self.load_model_and_data()
        
        print("Simple Camera Assistant initialized successfully!")
    
    def setup_english_voice(self):
        """Setup English text-to-speech voice"""
        try:
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Look for English voices
                for voice in voices:
                    if any(keyword in voice.name.lower() for keyword in ['english', 'us', 'en', 'zira']):
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                
                # Configure speech settings
                self.tts_engine.setProperty('rate', 160)  # Speech rate
                self.tts_engine.setProperty('volume', 0.9)  # Volume
                print("English voice configured successfully")
        except Exception as e:
            print(f"Voice setup error: {e}")
    
    def speak(self, text):
        """Convert text to speech"""
        print(f"Speaking: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def speak_with_delay(self, text, delay=0.5):
        """Speak text with a small delay"""
        self.speak(text)
        time.sleep(delay)
    
    def speak_multiple(self, texts):
        """Speak multiple texts with delays between them"""
        for text in texts:
            self.speak_with_delay(text, 0.3)
    
    def load_model_and_data(self):
        """Load the trained model and data"""
        try:
            # Load features
            with open('input/features.pkl', 'rb') as f:
                self.features = pickle.load(f)
            print(f"Loaded features for {len(self.features)} images")
            
            # Load feature extractor (VGG16) and apply global average pooling
            vgg = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
            self.model = Model(inputs=vgg.inputs, outputs=GlobalAveragePooling2D()(vgg.output))
            print("VGG16 feature extractor (GlobalAveragePooling) loaded successfully")

            # Try to load captioning model (.keras preferred, then .h5)
            self.caption_model = None
            try:
                if os.path.exists('best_model.keras'):
                    print("Found 'best_model.keras' — attempting to load caption model...")
                    self.caption_model = load_model('best_model.keras', compile=False)
                    print("Caption model loaded from best_model.keras")
                elif os.path.exists('best_model.h5'):
                    print("Found 'best_model.h5' — attempting to load caption model...")
                    self.caption_model = load_model('best_model.h5', compile=False)
                    print("Caption model loaded from best_model.h5")
                else:
                    print("No caption model file found (best_model.keras or best_model.h5). Using fallback descriptions")

                # Try to load tokenizer and max_length if saved
                try:
                    if os.path.exists('tokenizer.pkl'):
                        with open('tokenizer.pkl', 'rb') as f:
                            self.tokenizer = pickle.load(f)
                        print('Loaded tokenizer.pkl')
                    if os.path.exists('max_length.pkl'):
                        with open('max_length.pkl', 'rb') as f:
                            self.max_length = pickle.load(f)
                        print(f'Loaded max_length.pkl: {self.max_length}')
                except Exception as e:
                    print(f'Could not load tokenizer/max_length: {e}')

                if self.caption_model is not None:
                    try:
                        self.caption_model.summary()
                    except Exception:
                        pass

            except Exception as e:
                print(f"Could not load caption model: {e}")
                print("Caption model will be disabled and the assistant will use the demo fallback descriptions")
                self.caption_model = None
            
            # Load captions
            with open('input/captions.txt', 'r') as f:
                next(f)  # Skip header
                desc_doc = f.read()
            
            # Process captions
            self.mapping = {}
            for each_desc in desc_doc.split('\n'):
                tokens = each_desc.split(',')
                if len(each_desc) < 2:
                    continue
                image_id, desc_of = tokens[0], tokens[1:]
                image_id = image_id.split('.')[0]
                desc_of = " ".join(desc_of)
                if image_id not in self.mapping:
                    self.mapping[image_id] = []
                self.mapping[image_id].append(desc_of)
            
            # Create tokenizer
            img_desc = []
            for key in self.mapping:
                for caption in self.mapping[key]:
                    img_desc.append(caption)
            
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_texts(img_desc)
            self.max_length = max(len(text.split()) for text in img_desc)
            
            print(f"Loaded {len(self.mapping)} images with captions")
            
        except Exception as e:
            print(f"Error loading model and data: {e}")
            print("Please make sure the model files are in the correct location")
    
    def initialize_camera(self, camera_index=0):
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            if self.camera.isOpened():
                print(f"Camera initialized successfully (Index: {camera_index})")
                return True
            else:
                print(f"Failed to open camera {camera_index}")
                return False
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def capture_image(self):
        """Capture image from camera"""
        if self.camera is None:
            return None
        
        try:
            ret, frame = self.camera.read()
            if ret:
                return frame
            else:
                print("Failed to capture image")
                return None
        except Exception as e:
            print(f"Image capture error: {e}")
            return None
    
    def preprocess_image(self, image):
        """Preprocess image for model"""
        try:
            # Resize to 224x224
            image = cv2.resize(image, (224, 224))
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Preprocess for VGG16
            image = img_to_array(image)
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)
            
            return image
        except Exception as e:
            print(f"Image preprocessing error: {e}")
            return None
    
    def extract_features(self, image):
        """Extract features using VGG16"""
        try:
            features = self.model.predict(image, verbose=0)
            # The feature extractor now outputs a pooled vector (1,512).
            # Some caption models expect (1,512) vectors, others may expect maps — adapt if needed.
            return features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def predict_description(self, features):
        """Generate description from features"""
        try:
            # If a trained caption model is loaded, use it (greedy decoding)
            if hasattr(self, 'caption_model') and self.caption_model is not None:
                caption = self.generate_caption(features)
                return caption

            # Fallback: Simple demo descriptions (no trained caption model)
            descriptions = [
                "A person is looking at the camera",
                "There is an object in the view",
                "The scene contains various items",
                "A person or animal is visible",
                "Objects are arranged in the frame"
            ]
            import random
            return random.choice(descriptions)
            
        except Exception as e:
            print(f"Description prediction error: {e}")
            return "Unable to generate description"
    
    def clean_description(self, text):
        """Clean up generated description"""
        if not text:
            return "No description available"
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text

    def generate_caption(self, photo_features):
        """Generate a caption from photo features using the loaded caption model.

        This implements a simple greedy decode and supports a common
        architecture where the caption model expects two inputs:
        [photo_features, input_sequence]. If your trained model uses a
        different input signature you may need to adapt this function.
        """
        try:
            if self.caption_model is None:
                return "No caption model available"

            # Ensure photo_features is in batch form (1, ...)
            if len(photo_features.shape) == 3:
                # e.g., (7,7,512) -> expand to (1,7,7,512)
                photo = np.expand_dims(photo_features, axis=0)
            else:
                photo = photo_features

            # Use the tokens used during training: 'beginning' and 'ending'
            in_text = 'beginning'
            for i in range(self.max_length if self.max_length is not None else 34):
                sequence = self.tokenizer.texts_to_sequences([in_text])[0]
                sequence = pad_sequences([sequence], maxlen=self.max_length)

                # Decide how to call the model based on its inputs
                try:
                    n_inputs = len(self.caption_model.inputs)
                except Exception:
                    n_inputs = 1

                if n_inputs == 2:
                    # caption model expects [photo_vector, sequence]
                    # Ensure photo is shape (1, features)
                    try:
                        yhat = self.caption_model.predict([photo, sequence], verbose=0)
                    except Exception:
                        # Some models expect the photo vector flattened
                        yhat = self.caption_model.predict([np.reshape(photo, (photo.shape[0], -1)), sequence], verbose=0)
                else:
                    # Try calling with photo first, otherwise with sequence
                    try:
                        yhat = self.caption_model.predict(photo, verbose=0)
                    except Exception:
                        yhat = self.caption_model.predict(sequence, verbose=0)

                # yhat might be (1, vocab_size) or similar
                yhat_idx = np.argmax(yhat)
                word = self.tokenizer.index_word.get(yhat_idx)
                if word is None:
                    break
                in_text += ' ' + word
                if word == 'ending':
                    break

            # Clean 'beginning'/'ending' tokens
            return in_text.replace('beginning', '').replace('ending', '').strip()

        except Exception as e:
            print(f"Caption generation error: {e}")
            return "Unable to generate description"
    
    def describe_camera_view(self):
        """Describe current camera view"""
        try:
            # Capture image
            frame = self.capture_image()
            if frame is None:
                return "Unable to capture image"
            
            # Preprocess image
            processed_image = self.preprocess_image(frame)
            if processed_image is None:
                return "Unable to process image"
            
            # Extract features
            features = self.extract_features(processed_image)
            if features is None:
                return "Unable to extract features"
            
            # Generate description
            description = self.predict_description(features)
            cleaned_description = self.clean_description(description)
            
            return cleaned_description
            
        except Exception as e:
            return f"Error describing image: {str(e)}"
    
    def run_assistant(self):
        """Run the main assistant loop"""
        print("="*60)
        print("🎥 SIMPLE CAMERA VOICE ASSISTANT FOR THE BLIND")
        print("="*60)
        print("Commands:")
        print("• Press ENTER to describe current camera view")
        print("• Type 'help' for available commands")
        print("• Type 'quit' to exit")
        print("="*60)
        
        # Speak welcome message with more detail
        welcome_messages = [
            "Welcome to Smart Camera Voice Assistant for the Blind",
            "I am designed to help visually impaired users",
            "I will describe what the camera sees in clear English",
            "Initializing camera system now"
        ]
        self.speak_multiple(welcome_messages)
        
        # Initialize camera with detailed feedback
        camera_initialized = False
        for camera_index in [0, 1, 2]:
            self.speak(f"Trying to connect to camera number {camera_index + 1}")
            if self.initialize_camera(camera_index):
                camera_initialized = True
                self.speak(f"Camera number {camera_index + 1} connected successfully")
                break
            else:
                self.speak(f"Camera number {camera_index + 1} not available")
        
        if not camera_initialized:
            error_messages = [
                "No camera detected on your system",
                "Please check your camera connection",
                "Make sure no other programs are using the camera",
                "Then restart this program"
            ]
            self.speak_multiple(error_messages)
            print("No camera detected. Please check your camera connection.")
            return
        
        # Ready messages
        ready_messages = [
            "Camera system is now ready",
            "Press enter to describe the current view",
            "Type help for available commands",
            "Type quit to exit the program",
            "I will speak all responses to help you"
        ]
        self.speak_multiple(ready_messages)
        
        # Main loop
        while True:
            try:
                command = input("\nEnter command (or press ENTER to describe): ").strip().lower()
                
                # Exit commands
                if command in ['quit', 'exit', 'q']:
                    goodbye_messages = [
                        "Thank you for using the Smart Camera Voice Assistant",
                        "I hope I was able to help you today",
                        "Goodbye and have a great day!"
                    ]
                    self.speak_multiple(goodbye_messages)
                    break
                
                # Help commands
                elif command == 'help':
                    help_messages = [
                        "Here are the available commands:",
                        "Press enter to describe the current view",
                        "Type quit to exit the program",
                        "The system will speak all responses",
                        "This is designed to help visually impaired users",
                        "You can ask for descriptions as many times as you need"
                    ]
                    self.speak_multiple(help_messages)
                
                # Describe command (default when pressing enter)
                else:
                    processing_messages = [
                        "Taking picture now",
                        "Processing the image",
                        "Generating description"
                    ]
                    self.speak_multiple(processing_messages)
                    
                    description = self.describe_camera_view()
                    
                    if description and not description.startswith("Error"):
                        result_messages = [
                            "Here is the description of what I see:",
                            description,
                            "Press enter again for another description",
                            "Or type quit to exit the program"
                        ]
                        self.speak_multiple(result_messages)
                    else:
                        error_messages = [
                            "Unable to process the image",
                            "Please try again",
                            "Make sure the camera is not blocked",
                            "And there is enough light in the room",
                            "Press enter to try again"
                        ]
                        self.speak_multiple(error_messages)
                
            except KeyboardInterrupt:
                print("\nSystem stopped by user")
                self.speak("System stopped by user")
                break
            except Exception as e:
                print(f"System error: {e}")
                self.speak("Sorry, something went wrong. Please try again")
        
        # Release camera
        if self.camera is not None:
            self.camera.release()
            print("Camera released")
            self.speak("Camera released")
    
    def test_camera(self):
        """Test camera functionality"""
        print("Testing camera...")
        self.speak("Testing camera functionality")
        
        if self.initialize_camera():
            print("Camera working properly")
            self.speak("Camera is working properly and ready to use")
            
            # Test image capture
            frame = self.capture_image()
            if frame is not None:
                print("Image capture successful")
                self.speak("Image capture test successful")
                self.speak("The camera can take pictures correctly")
                return True
            else:
                print("Unable to capture image")
                self.speak("Unable to capture image from camera")
                self.speak("Please check if the camera is not blocked")
                return False
        else:
            print("Unable to access camera")
            self.speak("Unable to access camera")
            self.speak("Please check camera connection and try again")
            return False
    
    def quick_test(self):
        """Quick test - capture and describe"""
        print("Quick test...")
        self.speak("Starting quick test of the camera system")
        
        # Initialize camera
        if not self.initialize_camera():
            print("Unable to access camera")
            self.speak("Unable to access camera for quick test")
            return
        
        self.speak("Camera initialized successfully")
        
        # Capture and describe
        self.speak("Capturing image and generating description")
        description = self.describe_camera_view()
        print(f"Description: {description}")
        
        # Speak the description
        if description and not description.startswith("Error"):
            self.speak("Here is the description from the quick test:")
            self.speak(description)
            self.speak("Quick test completed successfully")
        else:
            self.speak("Quick test failed to generate description")
        
        # Release camera
        if self.camera is not None:
            self.camera.release()
            self.speak("Camera released after quick test")
        
        return description

def main():
    """Main function"""
    print("Starting Simple Camera Assistant...")
    
    try:
        # Create assistant
        assistant = SimpleCameraAssistant()
        
        # Test camera first
        if assistant.test_camera():
            print("Camera test successful. Starting assistant...")
            assistant.speak("Camera test successful. Starting the main assistant now.")
            assistant.run_assistant()
        else:
            print("Camera test failed. Please check your camera.")
            assistant.speak("Camera test failed. Please check your camera connection and restart the program.")
            
    except Exception as e:
        print(f"Error starting assistant: {e}")
        print("Please make sure all required files are in the correct location")
        try:
            # Try to speak error message if possible
            assistant = SimpleCameraAssistant()
            assistant.speak("Error starting assistant. Please check that all required files are present.")
        except:
            pass  # If we can't even create assistant, just show text error

if __name__ == "__main__":
    main()
