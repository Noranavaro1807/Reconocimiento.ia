import cv2
cap=cv2.VideoCapture(0)
net= cv2.dnn.readNetFromCaffe("opencv_face_detector.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

#parametros del modulo 
#tamaño de la pantalla para mostrar
anchonet = 300
altonet = 300

media =[184, 117, 123]
umbral = 0.7

while True:
    #leemos los frames 
    ret, frame = cap.read()
    
    #si hay error 
    if not ret:
        break
    
    #realizamos la conversion de formas 
    frame = cv2.flip(frame, 1)
    #extraemos info de los frames
    altoframe = frame.shape[0]
    anchoframe = frame.shape[1]
    
    #procesamos la imagen 
    blob = cv2.dnn.blobFromImage(frame,1.0,(anchonet, altonet), media, swapRB = False, crop = False)

    #corremos el modelo
    
    net.setInput(blob)
    detecciones = net.forward()
    
    #iteramos 
    for i in range(detecciones.shape[2]):
        #extraemos la confianza de esa deteccion 
        conf_detect = detecciones[0,0,i,2]
        # si superamos el umbral (70% deprobabilidad de que sea un rostro )
        if conf_detect > umbral:
            #extraeemos las cordenadas
            xmin = int(detecciones[0,0,i,3] * anchoframe)
            ymin = int(detecciones[0,0,i,4] * altoframe)
            xmax = int(detecciones[0,0,i,5]* anchoframe)
            ymax = int(detecciones[0,0,i,6]* altoframe)
            
            #dibujamos el rectangulo
            cv2.rectangle(frame,(xmin,ymin), (xmax,ymax), (0,0,255),2)
            #texto que vamos a mostrar
            label = "Confianza de deteccion: $.4f" % conf_detect
            #tamaño de fondo del label
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            #aqui colocamos fondo al texto
            cv2.rectangle(frame, (xmin, ymin - label_size[1]), (xmin + label_size[0], ymin + base_line),
                    (0,0,0), cv2.FILLED)
            #colocamos el texto 
            cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
                
    cv2.imshow("Deteccion de rostros", frame)
    
    t = cv2.waitKey(1)
    if t == 27:
        break
cv2.destroyAllWindows()
cap.release()
#mediaPipe