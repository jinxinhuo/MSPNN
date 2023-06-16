import torch
import torch.nn as nn
import torch.nn.functional as F
#from my.cbam import *
class multnet(nn.Module):
    def __init__(self):
        super(multnet, self).__init__()

        #============conv========#
        self.conv1 = nn.Conv2d(3,32,3,1,1,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, 3,1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,3,(2,2),1,bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.res1 = nn.Conv2d(32,64,1,(2,2),0,bias=False)
        self.bnres1 = nn.BatchNorm2d(64)

        #do relue here
        self.dropout = nn.Dropout(p=0.5)

        #===========short1=======#
        self.conv_s1 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn_s1 = nn.BatchNorm2d(64)
        self.pool_s1 = nn.MaxPool2d(3, (2, 2), 1)
        self.conv_s2 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn_s2 = nn.BatchNorm2d(128)
        self.pool_s2 = nn.MaxPool2d(3, (2, 2), 1)
        self.conv_s3 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn_s3 = nn.BatchNorm2d(256)
        self.pool_s3 = nn.MaxPool2d(4, (4, 4), 1)



        # #===========midle1=======#
        self.conv_m1 = nn.Conv2d(32,64,3,1,1,bias=False)
        self.bn_m1 = nn.BatchNorm2d(64)
        self.conv_m2 = nn.Conv2d(64,64,3,1,1, bias=False)
        self.bn_m2 = nn.BatchNorm2d(64)
        self.pool_m1 = nn.MaxPool2d(3,(2,2),1)
        self.res_m1 = nn.Conv2d(64, 128, 1, (2, 2), 0, bias=False)
        self.bnres_m1 = nn.BatchNorm2d(128)

        # ===========midle2=======#
        self.conv_m3 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn_m3 = nn.BatchNorm2d(128)
        self.conv_m4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.bn_m4 = nn.BatchNorm2d(128)
        self.pool_m2 = nn.MaxPool2d(3, (2, 2), 1)
        # self.res_m2 = nn.Conv2d(128, 256, 1, (4, 4), 0, bias=False)
        # self.bnres_m2 = nn.BatchNorm2d(256)

        self.conv_m5 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn_m5 = nn.BatchNorm2d(256)
        self.conv_m6 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn_m6 = nn.BatchNorm2d(256)
        self.pool_m3 = nn.MaxPool2d(3, (2, 2), 1)
        self.res_m3 = nn.Conv2d(256, 256, 1, (2, 2), 0, bias=False)
        self.bnres_m3 = nn.BatchNorm2d(256)
        #
        #
        #
        # #===========long1=======#
        self.conv4 = nn.Conv2d(32,64,5,1,2,bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64,5, 1, 2, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.pool5 = nn.MaxPool2d(3,(2,2),1)
        self.res2 = nn.Conv2d(64, 128, 1, (2, 2), 0, bias=False)
        self.bnres2 = nn.BatchNorm2d(128)
        #
        # # =============long2========#
        self.conv6 = nn.Conv2d(64,128,5,1,2,bias=True)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128,128,5,1,2,bias=True)
        self.bn7 = nn.BatchNorm2d(128)
        self.pool7 = nn.MaxPool2d(3,(2,2),1)
        self.res3 = nn.Conv2d(128,256,1,(2,2),0,bias=False)
        self.bnres3 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(128,256,5,1,2,bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256,256,5,1,2,bias=False)
        self.bn9 = nn.BatchNorm2d(256)
        self.pool9 = nn.MaxPool2d(3,(2,2),1)
        self.res4 = nn.Conv2d(256,256,2,(2,2),0,bias=False)
        self.bnres4 = nn.BatchNorm2d(256)
        #
        # #=============long3========#
        self.conv10 = nn.Conv2d(256,512,5,1,2,bias=False)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512,512,5,1,2,bias=False)
        self.bn11 = nn.BatchNorm2d(512)
        self.pool10 = nn.MaxPool2d(3,(2,2),1)
        self.conv12 = nn.Conv2d(512,256,1,1,0,bias=False)
        self.bn12 = nn.BatchNorm2d(256)
        #self.cbam = CBAM(768)
        #
        #
        #  #===============cat =========#
        self.conv13 = nn.Conv2d(768,1024,3,1,1,bias=False)
        self.bn13 = nn.BatchNorm2d(1024)
        self.conv14 = nn.Conv2d(1024, 1024,3, 1, 1, bias=False)
        self.bn14 = nn.BatchNorm2d(1024)
        self.fc = nn.Linear(1024,2)
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(2, 2)
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x) #[1, 32, 128, 128]
        x1 = self.res1(x) #[1, 64, 64, 64]
        x1 = self.bnres1(x1)

        #===========short1==========#
        x_s = self.conv_s1(x)
        x_s = self.bn_s1(x_s)
        x_s = self.pool_s1(x_s)
        x_s = self.conv_s2(x_s)
        x_s = self.bn_s2(x_s)
        x_s = self.pool_s2(x_s)
        x_s = self.conv_s3(x_s)
        x_s = self.bn_s3(x_s)
        x_s = self.pool_s3(x_s)
        x_s = self.relu(x_s)


        #==============mid1==================
        x_m = self.conv_m1(x)
        x_m = self.bn_m1(x_m)
        x_m = self.relu(x_m)
        x_m = self.conv_m2(x_m)
        x_m = self.bn_m2(x_m)
        x_m = self.pool_m1(x_m)  # [1, 64, 64,64]
        # x_m_res = self.res_m1(x_m)
        # x_m_res = self.bnres_m1(x_m_res)

        # ==============mid2==================
        x_m = self.relu(x_m + x1)
        x_m = self.conv_m3(x_m)
        x_m = self.bn_m3(x_m)
        x_m = self.relu(x_m)
        x_m = self.conv_m4(x_m)
        x_m = self.bn_m4(x_m)
        x_m = self.pool_m2(x_m)
        x_m = self.relu(x_m)


        x_m = self.conv_m5(x_m) #[1, 128, 32, 32]
        x_m = self.bn_m5(x_m)
        x_m = self.relu(x_m)
        x_m = self.conv_m6(x_m)
        x_m = self.bn_m6(x_m)
        x_m = self.pool_m3(x_m)
        x_m = self.relu(x_m)
        x_m = self.res_m3(x_m)
        x_m = self.bnres_m3(x_m)  # [1, 256, 8, 8]


        #===============long===================

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.pool5(x) #[1, 64, 64,64]
        x2 = self.res2(x)
        x2 = self.bnres2(x2) #[1, 256, 8, 8]


        x = self.relu(x1+x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.pool7(x)
        x = self.relu(x)  #[1, 128, 32, 32]
        x3 = self.res3(x)
        x3 = self.bnres3(x3) #[1, 256, 8, 8]

        x = self.conv8(x+x2)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.pool9(x)
        x = self.relu(x) #[1, 256, 16, 16]
        # x4 = self.res4(x)
        # x4 = self.bnres4(x4) #[1, 256, 8, 8]

        x = self.conv10(x+x3)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.pool10(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.bn12(x) #[1, 256, 8, 8]

        #=============concat========#
        x = torch.cat([x_s,x_m,x],1) #[1, 768, 8, 8]
        #x = self.cbam(x)
        x = self.conv13(x )
        x = self.bn13(x)
        x = self.relu(x)
        x = self.conv14(x)
        x = self.bn14(x)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x,(1, 1))
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        #x = self.dropout(x)
        x = self.preluip1(self.fc(x)) ## 1, 2
        ip2 = self.ip1(x)  # 1, 2
        return x, F.log_softmax(ip2, dim=1)



# a = torch.ones(1,3,256,256)
# model = multnet()
# c = model(a)
# print(c)
# print(f'num params: {sum(p.numel() for p in model.parameters())}')


