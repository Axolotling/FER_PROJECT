import csv
import numpy as np
import cv2 as cv
import os
import linecache
import tensorflow as tf

#######################Wycinanie klatek z filmu######################################################
def Wycinanie_klatek_z_filmu(path_video,freq):
    if not os.path.exists('.\\Klatki\\'):
        os.makedirs('.\\Klatki\\')
    klatki = 0
    licznik = 1
    film = cv.VideoCapture(path_video)
    while(film.isOpened()):
        ret, frame = film.read()
        if frame is None:
            break
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #3 nastepne linijki sa potrzebne tylko dla filmow nagrywanych pionowo, a kamerki są poziomo (do testow)
        rows, cols = img.shape
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        img = cv.warpAffine(img, M, (cols, rows))
        if klatki % freq == 0:
            cv.imwrite('.\\Klatki\\' + str(licznik)+".jpg",img)
            licznik = licznik + 1
        klatki=klatki+1
        if cv.waitKey(1)&0xFF == ord('q'):
            break
    print("ukonczono wycinanie klatek z filmu")
    film.release()
    cv.destroyAllWindows()
    return licznik
#######################Wycinanie twarzy z klatek######################################################
def Wycinanie_twarzy_z_klatek(licznik):
    path2='.\\Mordka\\'
    if not os.path.exists(path2):
        os.makedirs(path2)
    face_cascade = cv.CascadeClassifier('.\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    number = 1
    while(number<licznik):
        img = cv.imread('.\\Klatki\\' + str(number)+".jpg")
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in faces:
            crop_img = img[y:y + h, x:x + w]
            scaled_img = cv.resize(crop_img, (48,48), cv.INTER_CUBIC)
            cv.imwrite(path2 + str(number) + ".jpg", scaled_img)
        number=number+1
    print("ukonczono wycinanie twarzy z klatek")
    cv.waitKey(0)
    cv.destroyAllWindows()

############################Test_klasyfikacji########################################################
def Test_klasyfikacji(path):
    if not os.path.exists('.\\DataSet\\'):
        os.makedirs('.\\DataSet\\')
    if not os.path.exists('.\\DataSet\\0\\'):
        os.makedirs('.\\DataSet\\0\\')
    if not os.path.exists('.\\DataSet\\1\\'):
        os.makedirs('.\\DataSet\\1\\')
    if not os.path.exists('.\\DataSet\\2\\'):
        os.makedirs('.\\DataSet\\2\\')
    if not os.path.exists('.\\DataSet\\3\\'):
        os.makedirs('.\\DataSet\\3\\')
    if not os.path.exists('.\\DataSet\\4\\'):
        os.makedirs('.\\DataSet\\4\\')
    if not os.path.exists('.\\DataSet\\5\\'):
        os.makedirs('.\\DataSet\\5\\')
    if not os.path.exists('.\\DataSet\\6\\'):
        os.makedirs('.\\DataSet\\6\\')
    count = len(open(path, 'rU').readlines())
    number = 1
    licznik = 547*7
    blank_image = np.zeros((48, 48), np.uint8)
    while (number <= licznik):
        wiersz = linecache.getline(path, number)
        desired = wiersz[0]
        wiersz = wiersz[wiersz.find(",") + 1:]
        wiersz = wiersz[:wiersz.find(",")]
        pixels = wiersz.split(" ")
        val = list(map(int, pixels))
        for y in range(0, 48):
            for x in range(0, 48):
                blank_image[y][x] = val[48 * y + x]
        cv.imwrite('.\\DataSet\\' + str(desired) + "\\"+ str(number) + ".jpg", blank_image)
        number=number+1
    print("przetworzono wszystkie obrazy")
    cv.waitKey(0)
    cv.destroyAllWindows()
###############################Przygotowanie_danych##################################################
def extract_data_from_file(path):
    N_EPOCHS = 1  #byle co tu wpisalem zeby sie kompilowalo
    BATCH_SIZE = 10 #byle co tu wpisalem zeby sie kompilowalo
    with open(path) as csv_file:
        n_rows = N_EPOCHS * BATCH_SIZE
        row_counter = 0
        readcsv = csv.reader(csv_file, delimiter=',')
        for row in readcsv:
            row_counter += 1
            tmp_list = [0, 0, 0, 0, 0, 0, 0]
            tmp_list[int(row[0])] = 1
            Y.append(tmp_list)
            X.append(list(map(int, row[1].split())))
            if (row_counter >= n_rows):
                print("przetworzono dane")
                return
###############Obraz z kamerki i zapisywanie zdjec z niej############################################
def kamerka():
    try:
        cap = cv.VideoCapture(0)
        cap.set(3, 640)  # WIDTH
        cap.set(4, 480)  # HEIGHT
        face_cascade = cv.CascadeClassifier('.\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
        i = 0

    except:
        print("problem opening input stream")
        exit(1)

    while (i < 20):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:  # if return code is bad, abort.
            exit(0)
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(len(faces))

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv.imshow('frame', frame)
        cv.imwrite("image%04i.jpg" % i, frame)
        i += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
#####################################################################################################
path_film = '.\\Filmy\\MOV.mp4'
path_dataset = '.\\dataset.csv'

liczba_klatek = Wycinanie_klatek_z_filmu(path_film,5)
Wycinanie_twarzy_z_klatek(liczba_klatek)
Test_klasyfikacji(path_dataset)
Y = []
X = []
extract_data_from_file(path_dataset)
