#Number Plate Detection
import cv2
################################################################
framewidth = 640
frameheight = 480
nplateCascade = cv2.CascadeClassifier("haarcascade_license_plate_rus_16stages.xml")
################################################################
cap = cv2.VideoCapture("vid1.mp4")
cap.set(3, framewidth)
cap.set(4, frameheight)
cap.set(10, 150)
count = 0
while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    num_plates = nplateCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in num_plates:
        area = w*h
        if area>500:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255),2)
            imgRoi = img[y:y+h, x:x+w]
            cv2.imshow("ROI", imgRoi)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        cv2.imwrite("dataset/vid_created/NumPlate_"+str(count)+".jpeg", imgRoi)
        cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1
        break
