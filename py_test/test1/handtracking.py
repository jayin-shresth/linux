import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)  # track only one hand for simplicity

# Open webcam
cap = cv2.VideoCapture(0)

def is_thumb_up(hand_landmarks):
    """
    Returns True if thumb is up and all other fingers are folded
    """
    # Landmark indices for fingertips and thumb
    tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

    # Thumb: compare tip to IP joint
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]

    # Other fingers: compare tip to PIP joint
    fingers_folded = []
    for tip_id in tip_ids[1:]:
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[tip_id - 2]
        fingers_folded.append(tip.y > pip.y)  # y increases downward

    thumb_up = thumb_tip.y < thumb_ip.y  # higher in frame means smaller y
    return thumb_up and all(fingers_folded)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip for mirror view
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check thumbs-up
            if is_thumb_up(hand_landmarks):
                cv2.putText(image, "👍 Thumbs Up!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Thumbs Up Detector", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

