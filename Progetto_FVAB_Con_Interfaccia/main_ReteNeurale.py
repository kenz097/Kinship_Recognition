#Classe per effettuare i test sui volti tramite il modello generato con la rete neurale Siamese.

import cv2
import matplotlib.pyplot as plt
import pandas as pd


all_result = []
dati = []

def display_multiple_img(images):
    fig = plt.figure()

    j = 0
    for i in range (0,len(all_result)):
        if i%2 == 0:
            fig.add_subplot(2, 8, (i+1))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(all_result[i])


            plt.xlabel('Risultato di parentela: ' + str(dati[j]))
            j = j + 1
        else:
            fig.add_subplot(2, 8, (i+1))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(all_result[i])

    plt.axis("off")
    plt.show()


def functionNeural():
    try:
        test_df = pd.read_csv("submission.csv")
        for index in range (0,len(test_df)):

            img0_path = test_df.iloc[index].img_pair.split(",")[0]
            img1_path = test_df.iloc[index].img_pair.split(",")[1]
            img1 = cv2.imread("volti_conosciuti/"+img0_path)  # queryImage
            img2 = cv2.imread("volti_conosciuti/"+img1_path)  # trainImage

            all_result.append(img1)
            all_result.append(img2)

            dati.append(test_df.iloc[index].is_related)

        display_multiple_img(all_result)
    except:
        ("No file")

