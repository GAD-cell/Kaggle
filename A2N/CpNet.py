import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContrBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn=nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.dropout=nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv1_out = self.conv1(x)
        #conv1_out = self.gn(conv1_out)
        conv1_out = F.relu(conv1_out)
        #conv1_out = self.dropout(conv1_out)

        conv2_out = self.conv2(conv1_out)
        #conv2_out = self.gn(conv2_out)
        conv2_out = F.relu(conv2_out)
        #conv2_out = self.dropout(conv2_out)

        depthwise_concat = torch.cat([x, x], dim=1)
        aggregated = conv2_out + depthwise_concat
        pooled_out = self.maxpool(aggregated)
        return pooled_out

class ExpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExpBlock, self).__init__()
        self.gn=nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.dropout=nn.Dropout2d(p=0.2)
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x,Contr_out):
        convtrans_out = self.convT(x)
        #convtrans_out = self.gn(convtrans_out)
        convtrans_out = F.relu(convtrans_out)
        #convtrans_out = self.dropout(convtrans_out)

        concat = torch.cat([convtrans_out, Contr_out],dim=1)
        conv1_out = self.conv1(concat)
        #conv1_out = self.gn(conv1_out)
        conv1_out = F.relu(conv1_out)
        #conv1_out = self.dropout(conv1_out)

        conv2_out = self.conv2(conv1_out)
        #conv2_out = self.gn(conv2_out)
        conv2_out = F.relu(conv2_out)
        #conv2_out = self.dropout(conv2_out)
        
        aggregated = conv2_out + convtrans_out + Contr_out
        return aggregated

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckBlock, self).__init__()
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv1(conv1_out))
        aggregated = conv2_out + x
        concat = torch.cat([aggregated, x],dim=1)
        conv12_out = F.relu(self.conv2(concat))
        conv3_out = F.relu(self.conv3(conv12_out)) 
        output = conv3_out + aggregated + x
        return output
    

class CPNet(nn.Module):
    def __init__(self,in_channels):
        super(CPNet, self).__init__()
        self.first_conv = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.contr1 = ContrBlock(16,32)
        self.contr2 = ContrBlock(32, 64)
        self.contr3 = ContrBlock(64, 128)
        self.contr4 = ContrBlock(128, 256)
        self.contr5 = ContrBlock(256, 512)
        #self.contr6 = ContrBlock(512, 1024)
        
        self.bottleneck = BottleneckBlock(512,1024)
        
        #self.exp1 = ExpBlock(1024, 512)
        self.exp2 = ExpBlock(512, 256)
        self.exp3 = ExpBlock(256, 128)
        self.exp4 = ExpBlock(128, 64)
        self.exp5 = ExpBlock(64, 32)
        self.convT = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 5, kernel_size=1)

    def forward(self, x):
        # Contracting path

        first_conv=self.first_conv(x)

        contr1=self.contr1(first_conv)
        #print(contr1.shape)
        contr2=self.contr2(contr1)
        contr3=self.contr3(contr2)
        contr4=self.contr4(contr3)
        contr5=self.contr5(contr4)
        #contr6=self.contr6(contr5)
        exp1=self.bottleneck(contr5)
        #exp1=self.exp1(contr6,contr5)
        exp2=self.exp2(exp1,contr4)
        exp3=self.exp3(exp2,contr3)
        exp4=self.exp4(exp3,contr2)
        exp5=self.exp5(exp4,contr1)
        convT=F.relu(self.convT(exp5))
        output=self.final_conv(convT)
        return output
    

if __name__ == "__main__":
    input_tensor = torch.randn(1, 1, 64, 128)
    model = CPNet(in_channels=1)
    output = model(input_tensor)
    print(output)
    print("Taille de sortie :", output.shape)