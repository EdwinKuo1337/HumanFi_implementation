import numpy as np
import torch
import torch.nn as nn
import time
import os
from tool.dataLoader import *
from torch.utils.data import DataLoader
from models.models import *
import sys
from tool.averageMeter import AverageMeter, getAverageMeter, lossesUpdate
from tool.losses import CrossEntropyLabelSmooth
from tool.evalMetrics import evaluate
from random import sample
from colorama import Style, Fore, Back
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=False, default="./data/cifar10", help="Data directory")
parser.add_argument('--batchSize', default=32, type=int, help='training batch size')
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--decayGamma', default=0.8, type=float)
parser.add_argument('--decayStepSize', default=20, type=int)
parser.add_argument('--mode', type=str, default="training")
parser.add_argument('--trainingCheckpoint', type=str, default="./checkpoint/model.pth")
parser.add_argument('--testingCheckpoint', type=str, default="./checkpoint/model.pth")
parser.add_argument('--dataPath', type=str, default="/data2/lab50_dataset/*")
parser.add_argument('--maxEpoch', default=50, type=int)
parser.add_argument('--testFreq', default=1, type=int)
parser.add_argument('--csiChannel', default=30, type=int)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--clipLen', default=1, type=int)
parser.add_argument('--numAction', default=1, type=int)
parser.add_argument('--trainIdsRange', type=int, nargs='+')
parser.add_argument('--testIdsRange', type=int, nargs='+')
parser.add_argument('--numIds', default=40, type=int)

args = parser.parse_args()

useGpu = torch.cuda.is_available()

def saveCheckpoint(model, optimizer, **kwargs):
    if not os.path.exists('./checkpoint'): os.makedirs('./checkpoint')
    torch.save({
        'epoch': kwargs['epoch'] + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best': kwargs['best']
    }, args.trainingCheckpoint)

def trainHumanFi(model, trainingLoader, optimizer, scheduler, epoch, **loss):
    start = time.time()
    print(Fore.CYAN + "==> Training")
    print("==> Epoch {}/{}".format(epoch, args.maxEpoch))
    print("==> Learning Rate = {}".format(optimizer.param_groups[0]['lr']))
    losses = getAverageMeter(numLosses=6)
    model.train()


    pBar = tqdm(trainingLoader, desc='Training')
    for batchIndex, (csiDataDataData1, personId1) in enumerate(pBar):
        b, s, h, w = csiDataDataData1.shape
        if useGpu:
            csiDataDataData1 = csiDataDataData1.cuda()
            personId1 = personId1.cuda()
                    
        x1Id = model(csiDataDataData1)
        # print(x1Id)
        # exit()
        personId1 = personId1.cuda()
        x1Id = x1Id.cuda()
        # print(personId1.shape)
        # print(x1Id.shape)
        # loss['id'] = loss['id'].cuda(useGpu)
        idLoss = loss['id'](x1Id, personId1)
        totalLoss = idLoss
        losses = lossesUpdate(losses, args.batchSize, [idLoss])


        pBar.set_postfix({'id': '{:.3f}'.format(losses[0].avg)})

        optimizer.zero_grad()
        
        totalLoss.backward()
        optimizer.step()
        

    scheduler.step()

    endl = time.time()
    print('Costing time:', (endl-start)/60)
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print('Current time:', current_time)
    print(Style.RESET_ALL, end='')
    print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


def loadCheckpoint(model, mode, optimizer=None):
    if mode == 'training': checkpointPath = args.trainingCheckpoint
    else: checkpointPath = args.testingCheckpoint

    load = True
    if os.path.isfile(checkpointPath):
        checkpoint = torch.load(checkpointPath)
        startEpoch = checkpoint['epoch']
        bestScore = checkpoint['best']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer: optimizer.load_state_dict(checkpoint['optimizer'])
        print(Fore.RED + "=> loaded checkpoint '{}' (epoch {}, bestScore {:.4f})"\
        .format(checkpointPath, checkpoint['epoch'], checkpoint['best']) + Style.RESET_ALL)
    else:
        bestScore = 0
        startEpoch = 0
        print(Fore.RED + "=> no checkpoint found at '{}'".format(checkpointPath) + Style.RESET_ALL)
        load = False

    return model, optimizer, startEpoch, bestScore, load

def test(model, queryLoader, galleryLoader, ranks=[1, 3, 5, 7]):
    start = time.time()
    print(Fore.GREEN + "==> Testing")
    model.eval()
    queryPersonId = [], []
    correct = 0
    total = 0
    for batchIndex, (csiDataData, pids) in enumerate(tqdm(queryLoader, desc='Extracted features for query set')):
        total = total + 1
        if useGpu: csiDataData = csiDataData.cuda()
        b, s, h, w = csiDataData.size()

        outputs = model(csiDataData)
        perdictedId = outputs.argmax(1) + 1
        Gt = pids[0].cuda()
        if Gt == perdictedId:
            correct = correct + 1
        # print(Id)
        # print(pids.shape)
        # exit()
        # outputs = torch.mean(outputs, 0)
        # outputs = outputs.unsqueeze(0)

        # queryPersonId.extend(pids)
    accuracy = correct/total
    # queryPersonId = np.asarray(queryPersonId)
    # print("Extracted features for query set, obtained {}-by-{} matrix".format(queryFeat.size(0), queryFeat.size(1)))

    # galleryPersonId = [], []
    # for batchIndex, (csiData, pids) in enumerate(tqdm(galleryLoader, desc='Extracted features for gallery set')):
    #     if useGpu: csiData = csiData.cuda(useGpu)
    #     b, s, h, w = csiData.size()
        
    #     outputs = model(csiData)

    #     outputs = torch.mean(outputs, 0)
    #     outputs = outputs.unsqueeze(0)

    #     galleryPersonId.extend(pids)
    # galleryPersonId = np.asarray(galleryPersonId)

    # print("Extracted features for gallery set, obtained {}-by-{} matrix".format(galleryFeat.size(0), galleryFeat.size(1)))
    
    # m, n = queryFeat.size(0), galleryFeat.size(0)  
    # distanceMatrix = torch.pow(queryFeat, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #                 torch.pow(galleryFeat, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distanceMatrix.addmm_(1, -2, queryFeat, galleryFeat.t())
    # distanceMatrix = distanceMatrix.numpy()
    # print("Computing distance matrix, obtained {}-by-{} matrix".format(distanceMatrix.shape[0], distanceMatrix.shape[1]))

    # cmc, mAP = evaluate(distanceMatrix, queryPersonId, galleryPersonId)
    
    print("\nResults")
    print("------------------")
    print(accuracy)
    # print("mAP: {:.1%}".format(mAP))
    # print("CMC curve")
    # for r in ranks:
    #     print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")
    print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

    return accuracy

def main():
    if args.mode == 'training':
        trainIdsRange = [i-1 for i in range(args.trainIdsRange[0], args.trainIdsRange[1]+1)]
        # load amplitude and phase
        trainingData = preprocess(args.dataPath, args.clipLen, 'training')
        trainingLoader = DataLoader(dataset = DatasetFuse(trainingData, 'training', idStart=args.trainIdsRange[0]),
                                    batch_size = args.batchSize, num_workers = args.workers, drop_last = True, shuffle = True)

        testIdsRange = [i-1 for i in range(args.testIdsRange[0], args.testIdsRange[1]+1)]
        # load amplitude and phase
        queryData, galleryData = preprocess(args.dataPath, args.clipLen, 'testing')
        queryLoader = DataLoader(dataset = DatasetFuse(queryData, 'testing', idStart=args.testIdsRange[0]), batch_size = 1, num_workers = 0,
                                drop_last = True, shuffle = True)
        galleryLoader = DataLoader(dataset = DatasetFuse(galleryData, 'testing', idStart=args.testIdsRange[0]), batch_size = 1, num_workers = 0,
                                drop_last = True, shuffle = True)

        model = humanFi(args.csiChannel, len(trainIdsRange))
        model.weights_init()
        # cross entropy loss
        # idEntropyLoss = CrossEntropyLabelSmooth(args.numIds, use_gpu=useGpu)
        idEntropyLoss = nn.CrossEntropyLoss()
        
        if useGpu:
            model = model.cuda()
            idEntropyLoss = idEntropyLoss.cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decayStepSize, gamma=args.decayGamma)


        model, optimizer, startEpoch, bestScore, _ = loadCheckpoint(model, args.mode, optimizer)
        print(Fore.RED + '=> Training Ids\t: {} ~ {}'.format(args.trainIdsRange[0], args.trainIdsRange[1]))
        print('=> Testing Ids\t: {} ~ {}'.format(args.testIdsRange[0], args.testIdsRange[1]) + Style.RESET_ALL)

        for epoch in range(startEpoch, args.maxEpoch):
            start = time.time()
            trainHumanFi(model, trainingLoader, optimizer, scheduler, epoch, id=idEntropyLoss)
            if epoch % args.testFreq == 0:
                accuracy = test(model, queryLoader, galleryLoader)
                if accuracy > bestScore:
                    if not os.path.exists('./checkpoint'): os.makedirs('./checkpoint')
                    print(Fore.RED + 'Rank1: {:.3f}  >=  Best Score {:.3f}'.format(accuracy, bestScore))
                    print('Update model!!!')
                    print("---------------------------------------------------------------" + Style.RESET_ALL)
                    bestScore = accuracy
                    saveCheckpoint(model, optimizer, epoch=epoch, best=accuracy)

    elif args.mode == 'testing':
        testIdsRange = [i-1 for i in range(args.testIdsRange[0], args.testIdsRange[1]+1)]
        # load data
        queryData, galleryData = preprocess(args.dataPath, args.clipLen, 'testing')
        queryLoader = DataLoader(dataset = DatasetFuse(queryData, 'testing', idStart=args.testIdsRange[0]), batch_size = 1, num_workers = 0,
                                drop_last = True, shuffle = True)
        # galleryData = preprocess(args.dataPath, args.clipLen, 'training')
        galleryLoader = DataLoader(dataset = DatasetFuse(galleryData, 'testing', idStart=args.testIdsRange[0]), batch_size = 1, num_workers = 0,
                                drop_last = True, shuffle = True)

        # load the model
        model = humanFi(args.csiChannel, len(testIdsRange))
        model, optimizer, startEpoch, bestScore, load = loadCheckpoint(model, args.mode)
        if not load: exit()
        

        if useGpu: model = model.cuda()
        test(model, queryLoader, galleryLoader)
    else:
        print('Error: mode should be training or testing')
        return 0


if __name__ == '__main__':
    main()
