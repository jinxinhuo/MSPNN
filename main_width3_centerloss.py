from torch.utils.data import DataLoader
from dataset_cv import Mydata
from torchvision import transforms
from torch import nn,optim
import argparse
import torch
from utility.step_lr import StepLR
from utility.initialize import initialize
import sys;
from test.smooth_cross_entropy import smooth_crossentropy
from my.width3_centerloss import multnet
import matplotlib.pyplot as plt
from centerloss import CenterLoss
from xception import Xception
sys.path.append("..")


def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(2):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    plt.xlim(xmin=-8,xmax=8)
    plt.ylim(ymin=-8,ymax=8)
    plt.text(-7.8,7.3,"epoch=%d" % epoch)
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=100, type=int, help="Total number of epochs.")
    parser.add_argument("--learning_rate", default=0.0001, type=float,               #batchsize12时学习率0.1，损失能到几万
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    args = parser.parse_args()

#=================================随机种子-cuda-Dataloader========================================
    initialize(args, seed=43)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data = Mydata(r'G:\WN\code\FaceForensics-master\crop\FF++\c40\face2face\train',
                        transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]))
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

    val_data = Mydata(r'G:\WN\code\FaceForensics-master\crop\FF++\c40\face2face\val',
                       transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)

    test_data = Mydata(r'G:\WN\code\FaceForensics-master\crop\FF++\c40\face2face\test',
                       transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    trainval_loaders = {'train': train_loader, 'valid': val_loader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'valid']}
    test_size = len(test_loader.dataset)
#=================================模型-优化器-训练策略========================================
    model = multnet().to(device)
    optimizer =  optim.Adam(model.parameters(),lr=args.learning_rate )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)

#=================================训练========================================、
    best_acc = 0.0
    nllloss = nn.NLLLoss().cuda(0)  # CrossEntropyLoss = log_softmax + NLLLoss
    centerloss = CenterLoss(10, 2).cuda(0)
    optimzer4center = optim.SGD(centerloss.parameters(), lr=0.5)
    for epoch in range(args.epochs):

        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_corrects = 0.0

            train_loss = 0.0
            train_corrects = 0.0
            val_loss = 0.0
            val_corrects = 0.0
            ip1_loader = []
            idx_loader = []


            if phase == 'train':
                model.train()
            else:
                model.eval()


            for batch in trainval_loaders[phase]:
                inputs, targets = (b.to(device) for b in batch)
                optimizer.zero_grad()
                if phase == 'train':
                    ip1, pred = model(inputs)
                    _, preds = torch.max(pred.data, 1)
                else:
                    with torch.no_grad():
                        ip1, pred = model(inputs)
                        _, preds = torch.max(pred.data, 1)

                loss = nllloss(pred, targets) + 1 * centerloss(targets, ip1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimzer4center.step()
                    ip1_loader.append(ip1)
                    idx_loader.append((targets))

                # with torch.no_grad():  #计算结果不带梯度，值是没有区别的
                #     correct = torch.argmax(predictions.data, 1) == targets
                #     log(model,args.batch_size, loss.cpu(), correct.cpu(), scheduler.get_last_lr()[0])

                correct = preds == targets

                # running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(predictions == targets.data)
                running_loss += loss.sum().item()
                running_corrects += correct.sum().item()

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects / trainval_sizes[phase]

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1,args.epochs, epoch_loss, epoch_acc))

        with torch.no_grad():
            running_loss = 0
            running_corrects = 0
            for batch in test_loader:
                inputs, targets = (b.to(device) for b in batch)

                ip1, pred = model(inputs)
                _, preds = torch.max(pred.data, 1)
                loss = nllloss(pred, targets) + 1 * centerloss(targets, ip1)
                correct = preds == targets
                running_loss += loss.sum().item()
                running_corrects += correct.sum().item()
            loss = running_loss / test_size
            correct = running_corrects / test_size
            if correct > best_acc:
                best_acc = correct
                torch.save(model.state_dict(), 'width3_val_{}.pth'.format(epoch))
        print("[testt] Epoch: {}/{} Loss: {} Acc: {} best: {}".format(epoch + 1, args.epochs, loss, correct,best_acc))
        scheduler.step()

