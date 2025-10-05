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
import subprocess
import platform

# Import TensorFlow and Keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
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
        self.tts_engine = None
        # Force pyttsx3 as primary TTS engine. If initialization fails, we will
        # not fall back to PowerShell automatically to avoid mixed backends.
        try:
            # On Windows, prefer sapi5 for best voices
            if platform.system().lower().startswith('win'):
                try:
                    self.tts_engine = pyttsx3.init('sapi5')
                except Exception:
                    # fallback to default init
                    self.tts_engine = pyttsx3.init()
            else:
                self.tts_engine = pyttsx3.init()

            # Setup voice and test quickly
            self.setup_english_voice()
            try:
                self.tts_engine.say('')
                self.tts_engine.runAndWait()
            except Exception:
                # If engine cannot run, mark as not available
                print('pyttsx3 engine initialized but runAndWait failed')
                self.tts_engine = None
        except Exception as e:
            print(f"pyttsx3 init error: {e}")
            self.tts_engine = None

        # Configure speak timing to avoid overlapping audio
        # small delay before/after each speak and longer pause between messages
        self.speak_pre_delay = 0.05
        self.speak_post_delay = 0.05
        self.speak_between_pause = 0.6
        
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

    def powershell_speak(self, text):
        """Fallback TTS using Windows PowerShell System.Speech.Synthesis.SpeechSynthesizer

        This uses PowerShell to invoke the .NET System.Speech API which is available
        on most Windows systems and works without extra Python packages.
        """
        try:
            # Use PowerShell to speak text using System.Speech
            # Escape double quotes in the text
            safe_text = text.replace('"', '`"')
            ps_script = (
                "Add-Type -AssemblyName System.speech;"
                f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                "$s.SelectVoiceAsync((Get-Culture).Name) > $null;"
                f"$s.Speak(\"{safe_text}\");"
            )
            subprocess.run(["powershell", "-Command", ps_script], check=False)
        except Exception as e:
            # If even PowerShell fails, just print the message
            print(f"Fallback TTS error: {e}")
    
    def speak(self, text):
        """Convert text to speech"""
        # Always print to terminal for debugging/visual feedback
        print(f"Speaking: {text}")
        # Add a tiny pre-delay to ensure audio subsystem is ready
        try:
            time.sleep(self.speak_pre_delay)
        except Exception:
            pass

        # If pyttsx3 engine is available, use it (preferred)
        if self.tts_engine is not None:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                try:
                    time.sleep(self.speak_post_delay)
                except Exception:
                    pass
                return
            except Exception as e:
                print(f"pyttsx3 speak error: {e}")

        # If pyttsx3 not available, inform user and fallback to printing
        print("TTS engine not available (pyttsx3). Please install pyttsx3 or check audio settings.")
        print(text)
    
    def speak_with_delay(self, text, delay=0.5):
        """Speak text with a small delay"""
        self.speak(text)
        # Use configured between-message pause if available
        try:
            time.sleep(self.speak_between_pause if delay is None else delay)
        except Exception:
            time.sleep(0.6)
    
    def speak_multiple(self, texts):
        """Speak multiple texts with delays between them"""
        for text in texts:
            self.speak_with_delay(text, self.speak_between_pause)
    
    def load_model_and_data(self):
        """Load the trained model and data"""
        try:
            # Load features
            with open('input/features.pkl', 'rb') as f:
                self.features = pickle.load(f)
            print(f"Loaded features for {len(self.features)} images")
            
            # Load VGG16 base and add Global Average Pooling so output is a (512,) vector
            base = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
            x = base.output
            x = GlobalAveragePooling2D()(x)  # -> (None, 512)
            self.model = Model(inputs=base.inputs, outputs=x)
            print("VGG16 + GAP model loaded successfully (output dim 512)")
            
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

            # Try to load a trained caption model if present
            try:
                if os.path.exists('best_model.h5'):
                    print('Found best_model.h5 — attempting to load caption model')
                    try:
                        self.caption_model = load_model('best_model.h5')
                        print('Caption model loaded successfully')
                    except Exception as e:
                        print(f'Could not load caption model (best_model.h5): {e}')
                        self.caption_model = None
                else:
                    self.caption_model = None
            except Exception as e:
                print(f'Error checking/loading caption model: {e}')
                self.caption_model = None
            
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
            # Ensure we return a 1-D vector (1,512) -> (512,) for nearest neighbor
            import numpy as _np
            arr = _np.array(features)
            if arr.ndim > 2:
                arr = arr.reshape((arr.shape[0], -1))
            return arr
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def predict_description(self, features):
        """Generate description from features"""
        try:
            # Prefer nearest-neighbor caption using precomputed features if available
            nn_caption = self.nearest_caption(features)
            if nn_caption:
                return nn_caption

            # If no nearest caption, fall back to a small set of generic sentences
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

    def nearest_caption(self, features_vec):
        """Return a caption from the nearest image in the precomputed feature set.

        features_vec: numpy array of shape (1, D) or (D,)
        """
        try:
            if self.features is None or not isinstance(self.features, dict):
                return None

            # Normalize input vector
            import numpy as _np
            v = _np.array(features_vec)
            if v.ndim == 2 and v.shape[0] == 1:
                v = v.reshape(-1)

            # Build matrix of features (N, D) — make sure all rows are 1D vectors
            keys = list(self.features.keys())
            rows = []
            for k in keys:
                fv = _np.array(self.features[k])
                fv = fv.reshape(-1)
                rows.append(fv)
            mats = _np.vstack(rows)

            # Normalize
            mats_norm = _np.linalg.norm(mats, axis=1)
            v_norm = _np.linalg.norm(v)
            if v_norm == 0 or mats_norm.sum() == 0:
                return None

            sims = mats.dot(v) / (mats_norm * (v_norm + 1e-10))
            best_idx = int(_np.argmax(sims))
            best_key = keys[best_idx]

            # Return one of the captions for this key (choose the first or random)
            caps = self.mapping.get(best_key, [])
            if not caps:
                return None
            # Prefer deterministic selection: choose the first caption
            caption = caps[0]
            return caption
        except Exception as e:
            print(f"Nearest caption error: {e}")
            return None
    
    def clean_description(self, text):
        """Clean up generated description"""
        if not text:
            return "No description available"
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
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
        
        # Let user probe and select which camera index to use (helps identify front/back cameras)
        chosen_index = self.select_camera_interactively(max_index=4)
        if chosen_index is None:
            self.speak("No camera selected. Exiting the assistant.")
            print("No camera selected. Exiting.")
            return

        if self.initialize_camera(chosen_index):
            self.speak(f"Using camera index {chosen_index}")
        else:
            self.speak(f"Failed to open camera index {chosen_index}. Exiting.")
            print(f"Failed to open camera index {chosen_index}")
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

    def probe_cameras(self, max_index=4, show_preview=False):
        """Probe available camera indices up to max_index.

        For each index that opens, capture one frame, report resolution and save
        a temporary snapshot so the user can visually identify front/back camera.
        Returns a list of available camera indices.
        """
        available = []
        tmpdir = tempfile.gettempdir()
        self.speak(f"Probing cameras from index zero to {max_index}")
        for idx in range(0, max_index + 1):
            try:
                cap = cv2.VideoCapture(idx)
                if not cap or not cap.isOpened():
                    print(f"Camera {idx}: not available")
                    cap.release()
                    continue

                # Try to grab a frame
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"Camera {idx}: opened but no frame")
                    self.speak(f"Camera {idx} opened but no image received")
                    cap.release()
                    continue

                h, w = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                info = f"Camera index {idx} available, resolution {w} by {h}, fps approx {int(fps) if fps>0 else 'unknown'}"
                print(info)
                self.speak(info)

                # Save a snapshot so user can inspect image files if needed
                snapshot_path = os.path.join(tmpdir, f"camera_{idx}_snapshot.jpg")
                try:
                    cv2.imwrite(snapshot_path, frame)
                    print(f"Saved snapshot: {snapshot_path}")
                except Exception as e:
                    print(f"Could not save snapshot for camera {idx}: {e}")

                # Optionally show a brief preview window so user can see which camera it is
                if show_preview:
                    winname = f"Camera {idx} preview"
                    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
                    cv2.imshow(winname, frame)
                    cv2.waitKey(1000)  # show for 1 second
                    cv2.destroyWindow(winname)

                available.append({'index': idx, 'width': w, 'height': h, 'fps': fps, 'snapshot': snapshot_path})
                cap.release()
            except Exception as e:
                print(f"Error probing camera {idx}: {e}")
        if not available:
            self.speak("No cameras found during probe")
        return available

    def select_camera_interactively(self, max_index=4):
        """Probe cameras and let the user choose which index to use.

        Returns the chosen camera index or None if none selected.
        """
        available = self.probe_cameras(max_index=max_index, show_preview=True)
        if not available:
            return None

        print("Available cameras:")
        for cam in available:
            print(f"- Index {cam['index']}: {cam['width']}x{cam['height']}, snapshot: {cam['snapshot']}")

        self.speak("Please enter the camera index you want to use, or press Enter to use the first available camera")
        choice = input("Select camera index (or press Enter for first): ").strip()
        if choice == "":
            return available[0]['index']

        try:
            chosen = int(choice)
            if any(cam['index'] == chosen for cam in available):
                return chosen
            else:
                print("Chosen index not in available list. Using first available.")
                return available[0]['index']
        except ValueError:
            print("Invalid input. Using first available camera.")
            return available[0]['index']
    
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
