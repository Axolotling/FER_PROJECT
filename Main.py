import csv
import numpy as np
import cv2 as cv
import os
import linecache
import tensorflow as tf
import h5py
#import model
########################################################################################################################


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
########################################################################################################################


def create_image(pixels, sizex, sizey):
    blank_image = np.zeros((sizex, sizey), np.uint8)
    for y in range(0, sizey):
        for x in range(0, sizex):
            blank_image[y][x] = pixels[sizex * y + x]
    return blank_image
########################################################################################################################


def save_image(image, path, name):
    cv.imwrite(path + name, image)
########################################################################################################################


def cut_frames(path_video, freq):
    create_directory('.\\Frames\\')
    frames = 0
    counter = 0
    film = cv.VideoCapture(path_video)
    while film.isOpened():
        ret, frame = film.read()
        if frame is None:
            break
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #####################################################
        rows, cols = img.shape
        mat = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        img = cv.warpAffine(img, mat, (cols, rows))
        #####################################################
        if frames % freq == 0:
            save_image(img, '.\\Frames\\', str(counter) + ".jpg")
            counter = counter + 1
        frames = frames + 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    film.release()
    cv.destroyAllWindows()
    return counter
########################################################################################################################


def cut_faces(frames, sizex, sizey):
    create_directory('.\\Faces\\')
    #face_cascade = cv.CascadeClassifier('.\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    face_cascade = cv.CascadeClassifier('.\\haarcascades\\haarcascade_frontalface_default.xml')
    number = 0
    while number < frames:
        img = cv.imread('.\\Frames\\' + str(number)+".jpg")
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = img[y:y + h, x:x + w]
            scaled_img = cv.resize(crop_img, (sizex, sizey), cv.INTER_CUBIC)
            save_image(scaled_img, '.\\Faces\\', str(number) + ".jpg")
        number = number + 1
    cv.waitKey(0)
    cv.destroyAllWindows()
########################################################################################################################


def tests(path, create_dataset):
    create_directory('.\\DataSet\\')
    create_directory('.\\DataSet\\0\\')
    create_directory('.\\DataSet\\1\\')
    create_directory('.\\DataSet\\2\\')
    create_directory('.\\DataSet\\3\\')
    create_directory('.\\DataSet\\4\\')
    create_directory('.\\DataSet\\5\\')
    create_directory('.\\DataSet\\6\\')

    lines_in_file = len(open(path, 'rU').readlines())
    number = 1
    while number <= lines_in_file:
        row = linecache.getline(path, number)
        desired = row[0]
        row = row[row.find(",") + 1:]
        row = row[:row.find(",")]
        pixels = row.split(" ")
        val = list(map(int, pixels))
        img = create_image(val, 48, 48)
        save_image(img, '.\\DataSet\\' + str(desired) + "\\", str(number) + ".jpg")
        if create_dataset:
            img2 = create_dataset_canny(img, desired, number)
            save_image(img2, '.\\DataSet\\' + str(desired) + "\\", str(number) + "canny.jpg")
        number = number + 1
    cv.waitKey(0)
    cv.destroyAllWindows()
########################################################################################################################


def create_dataset_canny(image, desired, number):
    img = cv.Canny(image, 100, 200)
    save_image(img, '.\\DataSet\\' + str(desired) + "\\", str(number) + "canny.jpg")
    line = str(desired) + ','
    for y in range(48):
        for x in range(48):
            if x == 47 and y == 47:
                line = line + str(img[y][x]) + ',\n'
            else:
                line = line + str(img[y][x]) + ' '
    open(".\\dataset_canny.csv", 'a').write(line)
    return img
########################################################################################################################


def extract_data_from_file(path):
    n_epochs = 547
    batch_size = 7
    with open(path) as csv_file:
        n_rows = n_epochs * batch_size
        row_counter = 0
        readcsv = csv.reader(csv_file, delimiter=' ')
        for row in readcsv:
            row_counter += 1
            tmp_list = [0, 0, 0, 0, 0, 0, 0]
            tmp_list[int(row[0])] = 1
            Y.append(tmp_list)
            X.append(list(map(int, row[1].split())))
            if row_counter >= n_rows:
                return
########################################################################################################################


def kamerka(model, n_frames = 20):
    try:
        cap = cv.VideoCapture(0)
        
        cap.set(3, 640)
        cap.set(4, 480)
        face_cascade = cv.CascadeClassifier('.\\haarcascades\\haarcascade_frontalface_alt2.xml')
        #face_cascade = cv.CascadeClassifier('.\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
        i = 0

    except:
        print("problem opening input stream")
        exit(1)
   # w, h = 20, 8;
    tlumacz = ['wscieklosc','obrzydzenie','strach','radosc','smutek','zaskoczenie','neutralnosc']
    
    mark = [0,0,0,0,0,0,0]
    
  #  for i in range(0,19):
    #    ostatnie_20_predykcji[i] = [0,0,0,0,0,0,0]
    text = ''
    texts = ['','','','','','','']
    while i < n_frames:
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        if not ret:
            exit(0)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #print(len(faces))

        font                   = cv.FONT_HERSHEY_SIMPLEX       
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
     
       


        face_detected = False
        face_img = frame
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h,x:x+w]
            if (len(face_img) != len(frame)):
                face_detected = True
            cv.rectangle(frame, (x - 30, y + h + 5), (x + w + 30, y + h + 30 * len(texts)), (0, 0, 0),25)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for i in range(len(texts)):
                cv.putText(frame,texts[i], (x - 20,y+h +30 + 30 * i), font, 1/200 * w, fontColor, int(1/100 * w))#lineType)
            
        resized_face = cv.resize(face_img, (54, 54)) 
        canny_face = cv.Canny(resized_face, 100, 200)
        canny_face = canny_face[3:51,2:50]
        x = np.reshape(canny_face, (1, 48*48))
        x = np.array([[(number + 1) / 256.0 for number in numbers] for numbers in x])
        
        
        print(np.shape(x))
        print(x)
        #print(X[0])
        
       
        
        if(face_detected):
            #return
            pred = model.predict(x)
            predykcja = model.predict_classes(x)
            
            for i in range(0, len(mark)):
                mark[i] = (mark[i]*9+pred[0][i])/10
            
            text = tlumacz[predykcja[0]]
           
            for i in range(0, len(mark)):
                texts[i] = (tlumacz[i]+' '+ str(int(mark[i]*100))+'%')
            #print(tlumacz[predykcja[0]])
            #print(model.predict(x))
           
           # ostatnie_20_predykcji =  ostatnie_20_predykcji[1:]
           # ostatnie_20_predykcji.insert(19,cala_predykcja[0])
            
           # avg = []
          #  for j in range(0,len(cala_predykcja[0]) - 1):
           #     for i in range(0,20):                
            #        avg[j] += ostatnie_20_predykcji[i][j]
           #     avg[j] /= 20
           # print(cala_predykcja)
            
             
           # for i in range(0,len(cala_predykcja[0]) - 1):            
           #     print(tlumacz[i],': ', int(avg[i]*100),'%')
        cv.imshow('podglad', frame)
        cv.imshow('canny_resized', canny_face)

    
        #cv.imwrite("image%04i.jpg" % i, frame)
        i += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
#####################################################################################################
model_save_path = "model/saved_model.h5py"
model = tf.keras.models.load_model(model_save_path, custom_objects=None, compile=False)

kamerka(model, 1000)

path_film = '.\\Filmy\\MOV.mp4'
path_dataset = '.\\dataset.csv'
path_dataset_canny = '.\\dataset_canny.csv'

# frames_number = cut_frames(path_film, 5)
# cut_faces(frames_number, 48, 48)
#tests(path_dataset, 1)
# tests(path_dataset_canny, 0)
#Y = []
#X = []
# extract_data_from_file(path_dataset)
