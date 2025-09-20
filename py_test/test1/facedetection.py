import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Read the image
image = cv2.imread("/home/jayin/Pictures/oscar.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale (needed for Haar cascades)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,   # How much the image size is reduced at each scale
    minNeighbors=5,    # Higher value = fewer detections, but better quality
    minSize=(30, 30)   # Minimum face size
)

print(f"Found {len(faces)} face(s).")

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Show the result
cv2.imshow("Faces Found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

