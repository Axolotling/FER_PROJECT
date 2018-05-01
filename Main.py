import csv
import numpy as np
import cv2 as cv
import os
import linecache
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
    face_cascade = cv.CascadeClassifier('.\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
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


def kamerka():
    try:
        cap = cv.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        face_cascade = cv.CascadeClassifier('.\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
        i = 0

    except:
        print("problem opening input stream")
        exit(1)

    while i < 20:
        ret, frame = cap.read()
        if not ret:
            exit(0)
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
path_dataset_canny = '.\\dataset_canny.csv'

# frames_number = cut_frames(path_film, 5)
# cut_faces(frames_number, 48, 48)
tests(path_dataset, 1)
# tests(path_dataset_canny, 0)
Y = []
X = []
# extract_data_from_file(path_dataset)
