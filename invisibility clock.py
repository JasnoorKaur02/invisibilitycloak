import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

time.sleep(3)

# Capture background
for i in range(60):
    ret, background = cap.read()
background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mauve cloak HSV range
    lower = np.array([160, 60, 60])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to apply only on largest contour
    cloak_mask = np.zeros_like(mask)

    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Only apply effect if cloak area is large enough
        if cv2.contourArea(largest_contour) > 5000:  
            cv2.drawContours(cloak_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Apply cloak effect only in cloak_mask area
    res1 = cv2.bitwise_and(background, background, mask=cloak_mask)
    res2 = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(cloak_mask))
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow("Invisibility Cloak", final_output)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
