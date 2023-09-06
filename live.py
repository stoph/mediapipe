# https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md

import cv2
import mediapipe as mp

def num_to_range(num, inMin, inMax, outMin, outMax):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))

# Initialize MediaPipe Hand components
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles # Adds colors to digits
mp_hands = mp.solutions.hands

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands with default parameters
with mp_hands.Hands(
  static_image_mode=False, 
  #model_complexity=0,
  max_num_hands=2, 
  min_detection_confidence=0.5, 
  min_tracking_confidence=0.5) as hands:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Failed to grab image")
      break
    
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
