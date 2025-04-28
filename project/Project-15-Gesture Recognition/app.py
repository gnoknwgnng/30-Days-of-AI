import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import threading

st.set_page_config(page_title="Gesture Recognition Control", layout="centered")
st.title("🤚 Gesture Recognition – Device Control Demo")

st.markdown("""
This demo uses your webcam to recognize hand gestures and show fun feedback or open File Explorer.\
**Allow webcam access when prompted.**
""")

# Placeholder for video stream and gesture
frame_placeholder = st.empty()
gesture_placeholder = st.empty()
action_placeholder = st.empty()
emoji_placeholder = st.empty()

# Instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
- Make sure your webcam is connected.
- Try these gestures in front of the camera:
    - Open palm (🖐️) — Open File Explorer
    - Thumbs up (👍) — Shows "Hi!" animation
    - Thumbs down (👎) — Shows "Bye!" animation
    - Fist (✊)
    - OK sign (👌)
    - Peace/Victory sign (✌️)
- The recognized gesture will show a message or animation on screen.
""")

# Simulated device state
device_state = st.session_state.get("device_state", "Idle")

# Start button
def start_camera():
    st.session_state["run_camera"] = True

def stop_camera():
    st.session_state["run_camera"] = False

start = st.button("Start Camera", on_click=start_camera)
stop = st.button("Stop Camera", on_click=stop_camera)

def threaded_open_file_explorer():
    threading.Thread(target=lambda: os.startfile("C:/")).start()

# --- Gesture to Action Mapping ---
def get_default_mapping():
    return {
        "Open Palm": "Open File Explorer",
        "Fist": "Show Fist",
        "Thumbs Up": "Show Hi Animation",
        "OK": "Show OK",
        "Peace/Victory": "Show Peace",
        "Thumbs Down": "Show Bye Animation"
    }

gesture_action_mapping = st.session_state.get("gesture_action_mapping", get_default_mapping())

# Sidebar: Gesture-Action Mapping
st.sidebar.subheader("Gesture to Action Mapping")
for gesture in gesture_action_mapping:
    action = st.sidebar.text_input(f"{gesture} action", value=gesture_action_mapping[gesture], key=f"action_{gesture}")
    gesture_action_mapping[gesture] = action
st.session_state["gesture_action_mapping"] = gesture_action_mapping

# Sidebar: Gesture History
if "gesture_history" not in st.session_state:
    st.session_state["gesture_history"] = []
st.sidebar.subheader("Gesture History")
st.sidebar.write(st.session_state["gesture_history"][-10:][::-1])

def recognize_gesture(hand_landmarks):
    if not hand_landmarks:
        return None, None
    tips_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    lm = hand_landmarks.landmark
    fingers = []
    # Thumb: check if thumb tip is to the left (for right hand) or right (for left hand) of its MCP
    if lm[tips_ids[0]].x < lm[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers: check if tip is above PIP joint (y decreases upwards)
    for id in range(1, 5):
        if lm[tips_ids[id]].y < lm[tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    # Open Palm
    if sum(fingers) == 5:
        return "Open Palm", gesture_action_mapping["Open Palm"]
    # Fist
    elif sum(fingers) == 0:
        return "Fist", gesture_action_mapping["Fist"]
    # Thumbs Up: only thumb up, others down, thumb tip above MCP
    elif fingers == [1, 0, 0, 0, 0] and lm[4].y < lm[3].y:
        return "Thumbs Up", gesture_action_mapping["Thumbs Up"]
    # Thumbs Down: only thumb up, others down, thumb tip below MCP
    elif fingers == [1, 0, 0, 0, 0] and lm[4].y > lm[3].y:
        return "Thumbs Down", gesture_action_mapping["Thumbs Down"]
    # OK sign (index and thumb touching, others folded)
    elif fingers[0] == 1 and fingers[1] == 1 and sum(fingers[2:]) == 0:
        dist = np.linalg.norm(np.array([lm[4].x, lm[4].y]) - np.array([lm[8].x, lm[8].y]))
        if dist < 0.07:
            return "OK", gesture_action_mapping["OK"]
    # Peace/Victory (index and middle up, others down)
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace/Victory", gesture_action_mapping["Peace/Victory"]
    else:
        return "Unknown", None

if st.session_state.get("run_camera", False):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    while st.session_state.get("run_camera", False):
        ret, frame = cap.read()
        if not ret:
            frame_placeholder.warning("Webcam not found.")
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        gesture, action = None, None
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                gesture, action = recognize_gesture(handLms)
                # --- Blur background, keep hand sharp ---
                h, w, _ = frame.shape
                points = np.array([
                    [int(lm.x * w), int(lm.y * h)] for lm in handLms.landmark
                ], dtype=np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                if len(points) > 0:
                    hull = cv2.convexHull(points)
                    cv2.fillConvexPoly(mask, hull, 255)
                blurred = cv2.GaussianBlur(frame, (55, 55), 0)
                hand_region = cv2.bitwise_and(frame, frame, mask=mask)
                background = cv2.bitwise_and(blurred, blurred, mask=cv2.bitwise_not(mask))
                frame = cv2.add(hand_region, background)
        # Show video
        frame_placeholder.image(frame, channels="BGR")
        # Show gesture
        if gesture:
            gesture_placeholder.success(f"Gesture: {gesture}")
            # Add to gesture history
            if gesture != "Unknown":
                st.session_state["gesture_history"].append(gesture)
        else:
            gesture_placeholder.info("Show your hand to the camera...")
        # Show animation or perform action
        if action == "Open File Explorer":
            device_state = "File Explorer Opened"
            threaded_open_file_explorer()
            emoji_placeholder.markdown(":open_file_folder: <b>File Explorer Opened!</b>", unsafe_allow_html=True)
        elif action == "Show Hi Animation":
            device_state = "Hi!"
            emoji_placeholder.markdown("<h1 style='color:green;'>👍 Hi!</h1>", unsafe_allow_html=True)
        elif action == "Show Bye Animation":
            device_state = "Bye!"
            emoji_placeholder.markdown("<h1 style='color:red;'>👎 Bye!</h1>", unsafe_allow_html=True)
        elif action == "Show Fist":
            device_state = "Fist"
            emoji_placeholder.markdown("<h1>✊ Fist!</h1>", unsafe_allow_html=True)
        elif action == "Show OK":
            device_state = "OK"
            emoji_placeholder.markdown("<h1>👌 OK!</h1>", unsafe_allow_html=True)
        elif action == "Show Peace":
            device_state = "Peace"
            emoji_placeholder.markdown("<h1>✌️ Peace!</h1>", unsafe_allow_html=True)
        else:
            emoji_placeholder.markdown("")
        action_placeholder.markdown(f"**Device State:** {device_state}")
        # Break loop if stop button pressed
        if not st.session_state.get("run_camera", False):
            break
    cap.release()
    hands.close()
    frame_placeholder.empty()
    gesture_placeholder.empty()
    action_placeholder.empty()
    emoji_placeholder.empty()
    st.success("Camera stopped.")
