import torch
import numpy as np
import os
# Implement the data loader to read the dataset

class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        # implement the getFeatures method
        """
        read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        features = np.array([])
        for fileName in os.listdir(filePath):
            fullPath = filePath+"/"+fileName
            loadFeature = np.load(fullPath)
            if (features.size == 0):
                features = loadFeature
            else:
                features = np.concatenate((features,loadFeature),axis=0)

        return features
    def _getLabels(self, filePath):
        # implement the getLabels method
        """
        read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        labels = np.array([])
        for fileName in os.listdir(filePath):
            fullPath = filePath+"/"+fileName
            loadLabel = np.load(fullPath)
            if (labels.size == 0):
                labels = loadLabel
            else:
                labels = np.concatenate((labels,loadLabel),axis=0)

        return labels

    def __init__(self, mode):
        # remember to change the file path according to different experiments
        assert mode in ['SD-train','LOSO-train', 'SD-test', 'LOSO-test', 'finetune']
        if mode == 'SD-train':
            # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
            # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
            self.features = self._getFeatures(filePath='./dataset/SD_train/features/')
            self.labels = self._getLabels(filePath='./dataset/SD_train/labels/')
        if mode == 'LOSO-train':
            # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
            # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
            self.features = self._getFeatures(filePath='./dataset/LOSO_train/features/')
            self.labels = self._getLabels(filePath='./dataset/LOSO_train/labels/')
        if mode == 'finetune':
            # finetune: ./dataset/FT/features/ and ./dataset/FT/labels/
            self.features = self._getFeatures(filePath='./dataset/FT/features/')
            self.labels = self._getLabels(filePath='./dataset/FT/labels/')
        if mode == 'SD-test':
            # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
            # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
            self.features = self._getFeatures(filePath='./dataset/SD_test/features/')
            self.labels = self._getLabels(filePath='./dataset/SD_test/labels/')
        if mode == 'LOSO-test':
            # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
            # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
            self.features = self._getFeatures(filePath='./dataset/LOSO_test/features/')
            self.labels = self._getLabels(filePath='./dataset/LOSO_test/labels/')

    def __len__(self):
        # implement the len method
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # implement the getitem method
        return torch.Tensor(self.features[idx]),self.labels[idx]