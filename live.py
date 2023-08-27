import cv2
import mediapipe as mp

def num_to_range(num, inMin, inMax, outMin, outMax):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))

# Initialize MediaPipe Hand components
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands with default parameters
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("Failed to grab frame")
      break
    frame = cv2.flip(frame, 1)
    #frame = cv2.resize(frame, (800, 600))

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get hand landmarks
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

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
          
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        #print(hand_landmarks)

    # Display the frame
    cv2.imshow('Hand', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
