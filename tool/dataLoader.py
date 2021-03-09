import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import glob
import csv
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import hdf5storage
from random import randint


class DatasetFuse(Dataset):

    def __init__(self, allClip, mode, idStart):
        self.allClip = allClip
        self.action = {'walk': 0}
        self.mode = mode
        self.idStart = idStart

    def __len__(self):
        return len(self.allClip)

    def __getitem__(self, index):
        if self.mode == 'training':
            csiData1 = []
            for frame in self.allClip[index]:
                # print(frame)
                # exit()
                data = hdf5storage.loadmat(frame, variable_names={'csi_serial_AmpPhase'})
                csiData1.append(np.array(torch.from_numpy(data['csi_serial_AmpPhase'])))
            personId1 = int(self.allClip[index][0].split("/")[-4]) - 1
            # print(personId1)
            csiData1 = torch.FloatTensor(csiData1)
            # print(csiData1.shape)
            return csiData1, personId1

        elif self.mode == 'testing':
            
            csiData = []
            for frame in self.allClip[index]:
                # print(frame)
                data = hdf5storage.loadmat(frame, variable_names={'csi_serial_AmpPhase'})
                csiData.append(np.array(torch.from_numpy(data['csi_serial_AmpPhase'])))
            personId = int(self.allClip[index][0].split("/")[-4]) - 1
            csiData = torch.FloatTensor(csiData)
            return csiData, personId

        else:
            print('Error: mode should be training or testing')


def preprocess(dataPath, clipLen, mode):
    folder = glob.glob(dataPath)
    folder.sort(key=takeFolderIndex)
    actions = ['walk']
    # cnt = 0
    if mode == 'training':
        clip = []
        for folderIndex in range(40):
            for action in actions:
                filenames = glob.glob('{}/{}/train_AmpPhase/*.mat'.format(folder[folderIndex], action))
                filenames.sort(key=takeCsiIndex)
                filenames = filenames[1000:]
                for i in range(0, int(len(filenames)/clipLen), 1):
                    clip.append(filenames[i*clipLen: (i+1)*clipLen])
        return clip

    elif mode == 'testing':
        queryClip, galleryClip = [], []
        for folderIndex in range(40):
            for action in actions:
                filenames = glob.glob('{}/{}/test_AmpPhase/*.mat'.format(folder[folderIndex], action))
                filenames.sort(key=takeCsiIndex)
                queryFilenames = filenames[1000:]
                for i in range(0, int(len(queryFilenames)/clipLen), 20):
                    queryClip.append(queryFilenames[i*clipLen: (i+1)*clipLen])

                filenames = glob.glob('{}/{}/test_AmpPhase/*.mat'.format(folder[folderIndex], action))
                filenames.sort(key=takeCsiIndex)
                galleryFilenames = filenames[1000:]
                for i in range(0, int(len(galleryFilenames)/clipLen), 6):
                    galleryClip.append(galleryFilenames[i*clipLen: (i+1)*clipLen])

        return queryClip, galleryClip

    else:
        print('Error: mode should be training or testing')



def takeFolderIndex(elem):
    return int(elem.split("/")[3])


def takeCsiIndex(elem):
    return int(elem.split("/")[-1].split(".")[0])


