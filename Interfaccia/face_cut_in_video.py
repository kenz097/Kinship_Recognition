#Classe intaglio volti immagine nelle puntate per effettuare successivamente i test attraverso le funzioni
#di SNN e SIFT.

import os
import dlib
from PIL import Image
from skimage import io
from os import listdir
from os.path import isfile, join
import face_recognition
import cv2
import csv
import matplotlib.pyplot as plt

# Data manipulation
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader,Dataset
from torchvision.models import *

def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

def cut_initial():
    if not os.path.exists('volti_conosciuti'):
            os.makedirs('volti_conosciuti')

    # Load image
    onlyfiles_simili = [f for f in listdir("data_initial") if isfile(join("data_initial", f))]
    onlyfiles_simili.sort()

    for i in range(0, len(onlyfiles_simili)):
        print("Analisi..." + onlyfiles_simili[i])
        control = True
        
        onlyfiles_volti_conosciuti = [f for f in listdir("volti_conosciuti")
                            if isfile(join("volti_conosciuti", f))]

        try:
            image = io.imread("data_initial/" + onlyfiles_simili[i])
            detected_faces = detect_faces(image)
            
            for n, face_rect in enumerate(detected_faces):
                face = Image.fromarray(image).crop(face_rect)
                face = face.resize((100, 100))
                face.save("test.jpg")

                unknown_image = face_recognition.load_image_file("test.jpg")
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
           
                for j in range(0, len(onlyfiles_volti_conosciuti)):
                    known_image = face_recognition.load_image_file("volti_conosciuti/"
                                                                   + onlyfiles_volti_conosciuti[j])
                    known_encoding = face_recognition.face_encodings(known_image)[0]
                    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
                    if results[0] == True: 
                      control = False
              
                if control == True:
                  try:
                    known_image = face_recognition.load_image_file("test.jpg")
                    known_encoding = face_recognition.face_encodings(known_image)[0]
                    face.save("volti_conosciuti/" + onlyfiles_simili[i] + "_" + str(n) + ".jpg")
                  except:
                    print("Volto non riconoscibile")
        except:
          print("ERRORE")


def cut_game():
    # Load image
    onlyfiles_simili = [f for f in listdir("data_game/")
                        if isfile(join("data_game/", f))]

    onlyfiles_simili.sort()
    for i in range(0, len(onlyfiles_simili)):
        print("Analisi..." + onlyfiles_simili[i])
        control = True
        onlyfiles_volti_conosciuti = [f for f in listdir("volti_conosciuti")
                                      if isfile(join("volti_conosciuti", f))]

        try:
            image = io.imread("data_game/" + onlyfiles_simili[i])
            detected_faces = detect_faces(image)

            for n, face_rect in enumerate(detected_faces):
                face = Image.fromarray(image).crop(face_rect)
                face = face.resize((100, 100))
                face.save("test.jpg")

                unknown_image = face_recognition.load_image_file("test.jpg")
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

                for j in range(0, len(onlyfiles_volti_conosciuti)):
                    known_image = face_recognition.load_image_file("volti_conosciuti/"
                        + onlyfiles_volti_conosciuti[j])
                    known_encoding = face_recognition.face_encodings(known_image)[0]
                    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
                    if results[0] == True: 
                      control = False
            
                if control == True:
                    try:
                        known_image = face_recognition.load_image_file(
                            "test.jpg")
                        known_encoding = face_recognition.face_encodings(known_image)[0]
                        face.save("volti_conosciuti/"
                                  + onlyfiles_simili[i] + "_" + str(n) + ".jpg")
                    except:
                        print("Volto non riconoscibile")

        except:
            print("ERRORE")

def convert_game_parents(file,time):
    try:
        if not os.path.exists('data_game'):
            os.makedirs('data_game')
    except OSError:
        print ('Error: Creating directory of data')
    
    
    cap = cv2.VideoCapture(file)
    currentFrame = 0
    while(currentFrame < (time+2)*25*60):
        ret, frame = cap.read()

        # Saves image of the current frame in jpg file
        if currentFrame > time*25*60:
            if currentFrame%10 == 0:
              name = 'data_game/frame' + str(currentFrame) + '.jpg'
              print ('Creating...' + name)
              try:
                detected_faces = detect_faces(frame)
                cv2.imwrite(name, frame)
              except:
                print("ERRORE")
                break
        # To stop duplicate images
        currentFrame += 1

    cap.release()
    cv2.destroyAllWindows()

def convert_initial(file):
    # Playing video from file:
    cap = cv2.VideoCapture(file)

    try:
        if not os.path.exists('data_initial'):
            os.makedirs('data_initial')
    except OSError:
        print ('Error: Creating directory of data')

    currentFrame = 0
    while(currentFrame < 300):
        ret, frame = cap.read()

        # Saves image of the current frame in jpg file
        if currentFrame > 0:
            if currentFrame%3 == 0:
              name = 'data_initial/frame' + str(currentFrame) + '.jpg'
              print ('Creating...' + name)
              try:
                detected_faces = detect_faces(frame)
                if len(detected_faces) > 0:
                    cv2.imwrite(name, frame)
              except:
                print("ERRORE")
                break
        # To stop duplicate images
        currentFrame += 1

    cap.release()
    cv2.destroyAllWindows()


class SiameseNetwork(
    nn.Module):  # A simple implementation of siamese network, ResNet50 is used, and then connected by three fc layer.
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # self.cnn1 = models.resnet50(pretrained=True)#resnet50 doesn't work, might because pretrained model recognize all faces as the same.
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=.2),
        )
        self.fc1 = nn.Linear(2 * 32 * 100 * 100, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 50)
        self.fc4 = nn.Linear(500, 4)

    def forward(self, input1, input2):  # did not know how to let two resnet share the same param.
        output1 = self.cnn1(input1)
        output1 = output1.view(output1.size()[0], -1)  # make it suitable for fc layer.
        output2 = self.cnn1(input2)
        output2 = output2.view(output2.size()[0], -1)

        output = torch.cat((output1, output2), 1)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output



class testDataset(
    Dataset):  # different from train dataset, because the data organized in submission.csv is different from train.csv

    def __init__(self, transform=None):
        self.test_df = pd.read_csv("sample_submission.csv")  # pandas用来读取csv文件
        self.transform = transform

    def __getitem__(self, index):
        img0_path = self.test_df.iloc[index].img_pair.split(",")[0]
        img1_path = self.test_df.iloc[index].img_pair.split(",")[1]

        img0 = Image.open("volti_conosciuti" + "/" + img0_path)
        img1 = Image.open("volti_conosciuti" + "/" + img1_path)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1

    def __len__(self):
        return len(self.test_df)


def imshow(img, text=None, should_save=False):  # for showing the data you loaded to dataloader
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):  # for showing loss value changed with iter
    plt.plot(iteration, loss)
    plt.show()

def main(video,min):
    convert_game_parents(video,min)
    convert_initial(video)
    cut_initial()
    cut_game()

    with open("sample_submission.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["img_pair","is_related"])
        face = [f for f in listdir("volti_conosciuti/") if isfile(join("volti_conosciuti/", f))]
        parente_pos = 0
        nome_parente = 100000
        for i in range(0, len(face)):
            for j in range(i,len(face)):
                if int(face[i].split("frame")[1].split(".")[0]) > int(face[j].split("frame")[1].split(".")[0]):
                    temp = face[i]
                    face[i] = face[j]
                    face[j] =  temp
        for i in range(0, len(face)):
            if int(face[i].split("frame")[1].split(".")[0]) > 300 and int(face[i].split("frame")[1].split(".")[0]) < int(nome_parente):
                nome_parente = int(face[i].split("frame")[1].split(".")[0])
                parente_pos = i
        for i in range(parente_pos+1, len(face)):
            writer.writerow([face[parente_pos]+","+face[i],"0"])

    salvataggio="fvab2.pth"
    model=torch.load(salvataggio)
    net = model

    testset = testDataset(transform=transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()]))
    testloader = DataLoader(testset,shuffle=False,num_workers=0,batch_size=1)
    vis_dataloader = DataLoader(testset, shuffle=False, num_workers=2, batch_size=8)
    dataiter = iter(vis_dataloader)

    test_df = pd.read_csv("sample_submission.csv")#pandas用来读取csv文件
    predictions=[]
    with torch.no_grad():
        for data in testloader:
            img0, img1 = data
            img0, img1 = img0.cuda(), img1.cuda()
            outputs = net(img0,img1)
            _, predicted = torch.max(outputs, 1)
            predictions = np.concatenate((predictions,predicted.cpu().numpy()),0)#taking care of here, the output data format is important for transfer

    test_df['is_related'] = predictions
    test_df.to_csv("submission.csv", index=False)#submission.csv should be placed directly in current fold.

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    imshow(torchvision.utils.make_grid(concatenated))

    relation = []
    test_df = pd.read_csv("submission.csv")#pandas用来读取csv文件
    for i in range(0,len(test_df)):
        relation.append(int(test_df["is_related"][i]))
    print(relation)