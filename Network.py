import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

CH_FOLD2 = 1

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x




class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x



class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(32*CH_FOLD2))
        self.Conv2 = conv_block(ch_in=int(32*CH_FOLD2),ch_out=int(64*CH_FOLD2))
        self.Conv3 = conv_block(ch_in=int(64*CH_FOLD2),ch_out=int(128*CH_FOLD2))
        self.Conv4 = conv_block(ch_in=int(128*CH_FOLD2),ch_out=int(256*CH_FOLD2))
        self.Conv5 = conv_block(ch_in=int(256*CH_FOLD2),ch_out=int(512*CH_FOLD2))

        self.Up5 = up_conv(ch_in=int(512*CH_FOLD2),ch_out=int(256*CH_FOLD2))
        self.Up_conv5 = conv_block(ch_in=int(512*CH_FOLD2), ch_out=int(256*CH_FOLD2))

        self.Up4 = up_conv(ch_in=int(256*CH_FOLD2),ch_out=int(128*CH_FOLD2))
        self.Up_conv4 = conv_block(ch_in=int(256*CH_FOLD2), ch_out=int(128*CH_FOLD2))
        
        self.Up3 = up_conv(ch_in=int(128*CH_FOLD2),ch_out=int(64*CH_FOLD2))
        self.Up_conv3 = conv_block(ch_in=int(128*CH_FOLD2), ch_out=int(64*CH_FOLD2))
        
        self.Up2 = up_conv(ch_in=int(64*CH_FOLD2),ch_out=int(32*CH_FOLD2))
        self.Up_conv2 = conv_block(ch_in=int(64*CH_FOLD2), ch_out=int(32*CH_FOLD2))

        self.Conv_1x1 = nn.Conv2d(int(32*CH_FOLD2),output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = d1.squeeze(1)

        return torch.transpose(d1, -1, -2) * d1



class U_Net_FP(nn.Module):
    def __init__(self,img_ch=17,output_ch=1):
        super(U_Net_FP, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        FPNch = [8, 16, 32, 64, 128]
        self.fpn = FP(output_ch=FPNch)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(32), size=3)
        self.Conv2 = conv_block(ch_in=int(32)+FPNch[0],ch_out=int(64*CH_FOLD), size=3)
        self.Conv3 = conv_block(ch_in=int(64*CH_FOLD)+FPNch[1],ch_out=int(128*CH_FOLD))
        self.Conv4 = conv_block(ch_in=int(128*CH_FOLD)+FPNch[2],ch_out=int(256*CH_FOLD))
        self.Conv5 = conv_block(ch_in=int(256*CH_FOLD)+FPNch[3],ch_out=int(512*CH_FOLD))

        self.Up5 = tp_conv(ch_in=int(512*CH_FOLD)+FPNch[4],ch_out=int(256*CH_FOLD))
        self.Up_conv5 = conv_block(ch_in=int(512*CH_FOLD)+FPNch[3], ch_out=int(256*CH_FOLD))

        self.Up4 = tp_conv(ch_in=int(256*CH_FOLD),ch_out=int(128*CH_FOLD))
        self.Up_conv4 = conv_block(ch_in=int(256*CH_FOLD)+FPNch[2], ch_out=int(128*CH_FOLD))

        self.Up3 = tp_conv(ch_in=int(128*CH_FOLD),ch_out=int(64*CH_FOLD))
        self.Up_conv3 = conv_block(ch_in=int(128*CH_FOLD)+FPNch[1], ch_out=int(64*CH_FOLD))

        self.Up2 = tp_conv(ch_in=int(64*CH_FOLD),ch_out=int(32))
        self.Up_conv2 = conv_block(ch_in=int(64)+FPNch[0], ch_out=int(32))

        self.Conv_1x1 = nn.Conv2d(int(32),output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x, m):
        # encoding path
        fp1, fp2, fp3, fp4, fp5 = self.fpn(m)

        x1 = self.Conv1(x)
        x1 = torch.cat((x1, fp1), dim=1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = torch.cat((x2, fp2), dim=1)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = torch.cat((x3, fp3), dim=1)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = torch.cat((x4, fp4), dim=1)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = torch.cat((x5, fp5), dim=1)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = d1.squeeze(1)
        return torch.transpose(d1, -1, -2) * d1
