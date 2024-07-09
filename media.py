import cv2
import mediapipe as mp 

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()
#utilizacion de las camaras 
cap=cv2.VideoCapture(0)

#capturrar los fotogramas
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    #/cambiar tipo de formato de color 
    rgb_frame  = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    RESULT = face_detection.process(rgb_frame)
    RESULT = face_detection.process(rgb_frame)
    
    if RESULT.detections:
        for detection in RESULT.detections:
            bboxC= detection.location_data.relative_bounding_box
            ih, iw,_ = frame.shape
            #la diagonal se usa para seguir escribiendo en la pagina siguiente 
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height*ih)
                
            cv2.rectangle(frame, bbox ,(0,255,0), 2)
            cv2.imshow('facil detection', frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
cap.realise()
cv2.destroyAllWindows()            
                
            


