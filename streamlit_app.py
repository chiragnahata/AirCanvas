import cvzone
import cv2
import numpy as np
from PIL import Image
import time
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
from google import genai
from cvzone.HandTrackingModule import HandDetector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit config
st.set_page_config(layout="wide")
st.title("AirCanvas Math Solver")

# Sidebar for API Key
api_key = None

# Try getting from environment variable or secrets first
try:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

if not api_key and os.getenv("GOOGLE_API_KEY"):
    api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    st.sidebar.success("API Key loaded from environment/secrets.")
    # Allow override
    override_key = st.sidebar.text_input("Override API Key (optional)", type="password")
    if override_key:
        api_key = override_key
else:
    api_key = st.sidebar.text_input("Google AI API Key", type="password")

if not api_key:
    st.warning("Please enter your Google AI API Key in the sidebar or set it in .env file.")
    st.stop()

# Configure GenAI
client = genai.Client(api_key=api_key)

# Colors
PURPLE = (255, 0, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# States
DRAWING = 0
SOLVING = 1
SHOWING_RESULT = 2

# Webrtc config
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]}
)

import threading

class VideoProcessor:
    def __init__(self):
        self.detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)
        self.canvas = None
        self.prev_pos = None
        self.state = DRAWING
        self.ai_result = ""
        self.full_equation = ""
        self.start_time = 0
        self.display_text = "Draw your math problem here"
        
        # Async processing
        self.is_processing = False
        self.processing_thread = None

    def get_hand_info(self, img):
        # Find hands
        try:
            hands, img = self.detector.findHands(img, draw=True, flipType=True)
            if hands:
                hand = hands[0]
                lmList = hand["lmList"]
                fingers = self.detector.fingersUp(hand)
                return fingers, lmList
        except Exception:
            pass # Handle potential mediapipe errors gracefully
        return None, None

    def draw(self, fingers, lmList, img_shape):
        current_pos = None
        if fingers == [0, 1, 0, 0, 0]:  # Index finger up
            current_pos = lmList[8][0:2]
            if self.prev_pos is None:
                self.prev_pos = current_pos
            if self.canvas is not None:
                cv2.line(self.canvas, tuple(self.prev_pos), tuple(current_pos), PURPLE, 5)
        elif fingers == [1, 0, 0, 0, 0]:  # Thumb up
            if self.canvas is not None:
                 self.canvas[:] = 0  # Clear canvas
        
        return current_pos

    def preprocess_image(self, canvas):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        return Image.fromarray(thresh)

    def _process_ai_request(self, canvas_copy):
        """Background thread function to handle AI request"""
        print("DEBUG: Starting AI request thread")
        try:
            pil_image = self.preprocess_image(canvas_copy)
            prompt = """Analyze the handwritten math problem in this image. Then:
            1. Solve the problem step by step.
            2. Provide the final numerical answer.
            3. Show your work by writing out the full equation and each step of the solution.
            Format your response as follows:
            ANSWER: [final numerical result]
            STEPS:
            [Step-by-step solution with equations]"""
            
            print("DEBUG: Sending request to Gemini...")
            response = client.models.generate_content(
                model='gemini-2.0-flash', 
                contents=[prompt, pil_image]
            )
            print("DEBUG: Received response from Gemini")
            if response and response.text:
                result_text = response.text.strip()
                self.ai_result, self.full_equation = self.parse_ai_response(result_text)
            else:
                self.ai_result = "Error: No response"
                self.full_equation = ""
        except Exception as e:
            print(f"ERROR in AI thread: {e}")
            import traceback
            traceback.print_exc()
            self.ai_result = "Error"
            self.full_equation = f"An error occurred: {e}"
        finally:
            print("DEBUG: AI thread finished")
            self.is_processing = False
            self.state = SHOWING_RESULT

    def parse_ai_response(self, response):
        lines = response.split('\n')
        answer = "No answer provided"
        steps = []
        current_section = None

        for line in lines:
            if line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").replace("$", "").strip()
                current_section = "answer"
            elif line.startswith("STEPS:"):
                current_section = "steps"
            elif current_section == "steps" and line.strip():
                steps.append(line.replace("$", "").strip())

        full_equation = "\n".join(steps) if steps else "No detailed steps provided"
        return answer, full_equation

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)

            if self.canvas is None:
                self.canvas = np.zeros_like(img)
                print("DEBUG: Canvas initialized")

            # Only process hand gestures if NOT currently processing AI request
            # This prevents starting multiple requests or drawing while waiting
            if not self.is_processing:
                fingers, lmList = self.get_hand_info(img)

                if fingers:
                    if self.state == DRAWING:
                        current_pos = self.draw(fingers, lmList, img.shape)
                        if current_pos:
                            self.prev_pos = current_pos
                        else:
                             self.prev_pos = None 

                        if fingers == [0, 0, 1, 1, 1]:
                            print("DEBUG: Solve gesture detected")
                            self.state = SOLVING
                            self.start_time = time.time()
                    
                    elif self.state == SOLVING:
                        # Start async processing
                        if not self.is_processing:
                            print("DEBUG: Triggering processing thread")
                            self.is_processing = True
                            canvas_copy = self.canvas.copy() # important to copy!
                            self.processing_thread = threading.Thread(target=self._process_ai_request, args=(canvas_copy,))
                            self.processing_thread.start()
                    
                    if fingers == [1, 0, 0, 0, 0]: # Reset
                         self.canvas[:] = 0
                         self.ai_result = ""
                         self.full_equation = ""
                         self.state = DRAWING
                         self.is_processing = False # Reset flag just in case

            # Combine
            try:
                if self.canvas.shape != img.shape:
                    print(f"DEBUG: Shape mismatch! Canvas: {self.canvas.shape}, Img: {img.shape}")
                    self.canvas = cv2.resize(self.canvas, (img.shape[1], img.shape[0]))
                image_combined = cv2.addWeighted(img, 0.7, self.canvas, 0.3, 0)
            except Exception as e:
                 print(f"ERROR in combine: {e}")
                 image_combined = img

            # UI Overlay
            if self.state == DRAWING:
                cv2.putText(image_combined, self.display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2, cv2.LINE_AA)
            
            elif self.state == SOLVING:
                 # This state persists while is_processing is True
                 elapsed_time = time.time() - self.start_time
                 # Create a simpler animation or text
                 dots = "." * (int(elapsed_time * 2) % 4)
                 cv2.putText(image_combined, f"Solving{dots}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2, cv2.LINE_AA)
            
            elif self.state == SHOWING_RESULT:
                cv2.putText(image_combined, f"Answer: {self.ai_result}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2, cv2.LINE_AA)
                y_offset = 100
                for line in self.full_equation.split('\n'):
                    cv2.putText(image_combined, line, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
                    y_offset += 30

             # Gestures
            h, w, c = image_combined.shape
            cv2.putText(image_combined, "Gestures:", (50, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
            cv2.putText(image_combined, "- Index finger: Draw", (70, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
            cv2.putText(image_combined, "- Thumb: Clear canvas", (70, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
            cv2.putText(image_combined, "- Last 3 fingers: Solve", (70, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(image_combined, format="bgr24")
        except Exception as e:
            print(f"CRITICAL ERROR in recv: {e}")
            import traceback
            traceback.print_exc()
            return frame # Return original frame to avoid freeze

webrtc_streamer(
    key="aircanvas",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
