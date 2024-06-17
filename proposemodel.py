import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##########################Generator###############################

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class AttentionBlock(nn.Module):
    """Attention Block module."""
    def __init__(self, in_channels, num_styles):
        super(AttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.num_styles = num_styles
        self.gram_matrices = []

    def forward(self, x, style_idx):
        batch_size, channels, height, width = x.size()
        # Calculate the queries, keys, and values
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        # Calculate the attention weights based on style information
        attention_weights = F.softmax(torch.bmm(query, key), dim=2)
        style_weights = self.num_styles * torch.ones(batch_size, width * height, self.num_styles, device=x.device)
        style_weights = style_weights.scatter_(2, style_idx.view(-1, 1, 1), 0)
        attention_weights = attention_weights * style_weights

        # Calculate the attended values 
        attended_values = torch.bmm(value, attention_weights.transpose(1, 2))
        attended_values = attended_values.view(batch_size, channels, height, width)
        
        # Apply the gamma factor and add the attended values to the input
        x = self.gamma * attended_values + x
        return x

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)  # Create the main network up to this point
        self.attention_blocks = nn.ModuleList()

        # Initialize AttentionBlock and ResidualBlock layers in an interleaved manner
        attention_residual_layers = []
        for i in range(repeat_num):
            attention_residual_layers.append(AttentionBlock(in_channels=curr_dim, num_styles=c_dim))
            attention_residual_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.attention_residual_blocks = nn.ModuleList(attention_residual_layers)
            
        # for i in range(repeat_num):
        #     layers.append(AttentionBlock(in_channels=curr_dim, num_styles=c_dim))
        #     layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    
    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x_fake = self.main(x)  # Generate x_fake using the main network

        for i, att_block in enumerate(self.attention_blocks):
            x_fake = att_block(x_fake, style_idx=i)  # Apply AttentionBlock for each style
        return x_fake


##########################Discriminator###############################
class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
