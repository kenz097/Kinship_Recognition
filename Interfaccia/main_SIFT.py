#Classe per effettuare i test sui volti tramite il modello generato con il metodo SIFT.

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import csv

all_result = []
dati = []
all_good_poits = []
all_distance = []

#Classe per creare gli accoppiamenti delle immagini con i punti in comune con valore di distanza sotto lo 0.7
def SIFT(path1, path2):
    img1 = cv2.imread('volti_conosciuti/' + path1) # queryImage
    img2 = cv2.imread('volti_conosciuti/' + path2) # trainImage

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #sift
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    kp_1, desc_1 = sift.detectAndCompute(img1,None)
    kp_2, desc_2 = sift.detectAndCompute(img2,None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)
    good_points = []
    ratio = 0.7

    for m, n in matches:
        if m.distance < ratio*n.distance:
            good_points.append(m)
            all_good_poits.append((m.distance/n.distance))

    result = cv2.drawMatches(img1, kp_1, img2, kp_2, good_points, None)
    all_result.append(result)

    dati.append(str(len(kp_1)))
    dati.append(str(len(kp_2)))
    dati.append(str(len(good_points)))

#Funzione grafico
def display_multiple_img():
    fig = plt.figure()
    print(all_distance)
    for i in range(0, len(all_result)):
        fig.add_subplot(4, 2, (i + 1))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Valore punti in comune: ' + all_distance[i])
        plt.imshow(all_result[i])

    plt.show()


#Funzione per richiamare la costruzione della tabella 
def functionSIFT():
    try:
        test_df = pd.read_csv("submission.csv")

        for index in range (0,len(test_df)):
            img0_path = test_df.iloc[index].img_pair.split(",")[0]
            img1_path = test_df.iloc[index].img_pair.split(",")[1]
            SIFT(img0_path, img1_path)

        with open("video_SIFT.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Punti totali img1", "Punti totali img2", "Punti in comune con ratio 0.7:", "Valore_punti"])
            m = 0
            for i in range(0,8):
                distance = "["
                j = dati[(i*3)+2]
                for x in range(m,int(j)+m):
                    if x < int(j)+m-1:
                        distance = distance + str(round(all_good_poits[x],2)) + ","
                    else:
                        distance = distance + str(round(all_good_poits[x],2))
                m = m + int(j)
                distance = distance + "]"
                all_distance.append(distance)

                writer.writerow([dati[(i*3)], dati[((i*3)+1)], dati[(i*3)+2],distance])
        display_multiple_img()

    except:
        ("No file")