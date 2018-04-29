import numpy as np
import cv2 as cv
import os
import linecache
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
        #3 nastepne linijki sa potrzebne tylko dla filmow nagrywanych pionowo
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
    Face = []
    face_cascade = cv.CascadeClassifier('.\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    number = 1
    while(number<licznik):
        img = cv.imread('.\\Klatki\\' + str(number)+".jpg")
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in faces:
            crop_img = img[y:y + h, x:x + w]
            scaled_img = cv.resize(crop_img, (48,48), cv.INTER_CUBIC)
            cv.imwrite(path2 + str(number) + ".jpg", scaled_img)
        for i in range(48):
            for j in range(48):
                Face.append(scaled_img[i][j][0])
        number=number+1
    print("ukonczono wycinanie twarzy z klatek")
    cv.waitKey(0)
    cv.destroyAllWindows()

#############################################################################
def Wczytaj_Do_Nauki(path):
    if not os.path.exists('.\\DataSet\\'):
        os.makedirs('.\\DataSet\\')
    count = len(open(path, 'rU').readlines())
    number = 1
    licznik = 35887
    blank_image = np.zeros((48, 48), np.uint8)
    while (number <= licznik):
        wiersz = linecache.getline(path, number)

        wiersz = wiersz[wiersz.find(",") + 1:]
        wiersz = wiersz[:wiersz.find(",")]

        pixels = wiersz.split(" ")
        val = list(map(int, pixels))
        for y in range(0, 48):
            for x in range(0, 48):
                blank_image[y][x] = val[48 * y + x]
        cv.imwrite('.\\DataSet\\' + str(number) + ".jpg", blank_image)
        number=number+1
    print("przetworzono wszystkie obrazy")
    cv.waitKey(0)
#############################################################################
path_film = '.\\Filmy\\MOV.mp4'
path_dataset = '.\\dataset.csv'

liczba_klatek = Wycinanie_klatek_z_filmu(path_film,5)
Wycinanie_twarzy_z_klatek(liczba_klatek)
Wczytaj_Do_Nauki(path_dataset)
