import cv2

# Initialize the webcam (0 refers to the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    # Loop to continuously capture frames
    while True:
        ret, frame = cap.read() # Capture frame-by-frame
        if ret: # Check if frame is successfully read
            cv2.imshow('Live Video Stream', frame) # Display the frame
        else:
            print("Error: Could not read frame.")

        if cv2.waitKey(1) & 0xFF == ord('q'): # Break the loop on pressing 'q'
            break

    # Release the capture and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
