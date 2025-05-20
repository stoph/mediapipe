# https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md

import cv2
import mediapipe as mp
import subprocess

def num_to_range(num, inMin, inMax, outMin, outMax):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))

# Initialize MediaPipe Hand components
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles # Adds colors to digits
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Initialize webcam
cap = cv2.VideoCapture(0)

def is_peace_sign(hand_landmarks, mp_hands):
    # Get the y-coordinates of the finger tips and MCPs
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
    
    # Check if index and middle fingers are extended (tip is above MCP)
    index_extended = index_tip < index_mcp
    middle_extended = middle_tip < middle_mcp
    # Check if ring and pinky fingers are closed (tip is below MCP)
    ring_closed = ring_tip > ring_mcp
    pinky_closed = pinky_tip > pinky_mcp
    return index_extended and middle_extended and ring_closed and pinky_closed

# Initialize MediaPipe Hands with default parameters
with mp_hands.Hands(
  static_image_mode=False, 
  #model_complexity=0,
  max_num_hands=2, 
  min_detection_confidence=0.5, 
  min_tracking_confidence=0.5) as hands, \
     mp_face_detection.FaceDetection(
         model_selection=0,
         min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh(
         static_image_mode=False,
         max_num_faces=1,
         refine_landmarks=True,
         min_detection_confidence=0.5,
         min_tracking_confidence=0.5) as face_mesh:

    gaze_focused = False
    face_present = False
    gaze_focus_counter = 0
    gaze_unfocus_counter = 0
    DEBOUNCE_FRAMES = 5

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to grab image")
            break
        
        # Face detection (before color conversion for hands)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_results = face_detection.process(image_rgb)
        if face_results.detections:
            if not face_present:
                print("Face detected (focus ON)")
                face_present = True
            # Draw bounding boxes for each detected face
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            if face_present:
                print("Face lost (focus OFF)")
                face_present = False

        # Gaze detection using Face Mesh
        face_mesh_results = face_mesh.process(image_rgb)
        if face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            # Left and right iris center indices (468 and 473)
            left_iris = face_landmarks.landmark[468]
            right_iris = face_landmarks.landmark[473]
            # Left and right eye corner indices (33, 133 for left; 362, 263 for right)
            left_eye_left = face_landmarks.landmark[33]
            left_eye_right = face_landmarks.landmark[133]
            right_eye_left = face_landmarks.landmark[362]
            right_eye_right = face_landmarks.landmark[263]
            # Calculate horizontal iris position as a ratio within the eye
            left_eye_ratio = (left_iris.x - left_eye_left.x) / (left_eye_right.x - left_eye_left.x)
            right_eye_ratio = (right_iris.x - right_eye_left.x) / (right_eye_right.x - right_eye_left.x)
            # If both ratios are near 0.5, the person is looking forward
            if 0.4 < left_eye_ratio < 0.6 and 0.4 < right_eye_ratio < 0.6:
                gaze_focus_counter += 1
                gaze_unfocus_counter = 0
                if not gaze_focused and gaze_focus_counter >= DEBOUNCE_FRAMES:
                    print("Eyes focused")
                    gaze_focused = True
            else:
                gaze_unfocus_counter += 1
                gaze_focus_counter = 0
                if gaze_focused and gaze_unfocus_counter >= DEBOUNCE_FRAMES:
                    print("Eyes unfocused")
                    gaze_focused = False
        else:
            gaze_unfocus_counter += 1
            gaze_focus_counter = 0
            if gaze_focused and gaze_unfocus_counter >= DEBOUNCE_FRAMES:
                print("Eyes unfocused")
                gaze_focused = False

        # Process the image and get hand landmarks
        #image = cv2.resize(image, (800, 600))
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        image_height, image_width, _ = image.shape

        # Draw hand landmarks
        if results.multi_hand_landmarks:
          #for hand_landmarks in results.multi_hand_landmarks:
          for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
            print(f'HAND NUMBER: {hand_no+1}')
            print('-----------------------')
            
            # Peace sign detection for hand 1
            if hand_no == 0 and is_peace_sign(hand_landmarks, mp_hands):
                print("PEACE SIGN DETECTED! ✌️")
                subprocess.run(["osascript", "-e", 'display notification "Peace sign detected" with title "Action"'])
            
            finger_point  = 8
            print(f'{mp_hands.HandLandmark(finger_point).name}:')
            x = hand_landmarks.landmark[mp_hands.HandLandmark(finger_point).value].x
            x = num_to_range(x, 0, 1, 0, 255)
            print(f'x: {x}')

            # for i in range(5):
            #   print(f'{mp_hands.HandLandmark(i).name}:')
            #   print(f'{hand_landmarks.landmark[mp_hands.HandLandmark(i).value]}')
            
            mp_drawing.draw_landmarks(
              image, 
              hand_landmarks, 
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
            #print(hand_landmarks)

        # Display the image (flipped horizontally for selfie view)
        cv2.imshow('Hands', cv2.flip(image, 1))

        # Exit on ESC
        if cv2.waitKey(5) & 0xFF == 27:
          break

cap.release()
cv2.destroyAllWindows()
