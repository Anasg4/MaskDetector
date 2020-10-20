import cv2, numpy as np, ctypes

face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
mask = cv2.CascadeClassifier("cascade_mask.xml")
data = cv2.VideoCapture(0)

while True:
    success, img = data.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face1 = face.detectMultiScale(grey, 1.3, 10)
    mask1 = mask.detectMultiScale(grey, 1.3, 10)
    #No mask
    for (x, y, w, h) in face1:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, "Tanpa Masker", (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 2)
        # winsound.Beep(700, 500) #If someone not using mask this sound will triggred, but your fps becomse slow
    # Use Mask
    for (x, y, w, h) in mask1:
        cv2.rectangle(img, (x, y), (x + w, y + h), (180, 200, 150), 2)
        cv2.putText(img, "Pake Masker", (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 2)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xff ==ord('q'): #pencet q mematikan toools
        break
data.release()
cv2.destroyAllWindows()

