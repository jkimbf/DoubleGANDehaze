from libraries import *
from Unet import UNet
from Discriminator import Discriminator
from Fusion_Discriminator import Fusion_Discriminator
from Losses import *


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F3.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F3.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F3.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F3.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F3.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


class DoubleGANNet(nn.Module):

    def __init__(self, unet_input, unet_output, discriminator_input):
        super().__init__()


        unet = UNet(in_channels=unet_input ,out_channels=unet_output)
        unet = nn.DataParallel(unet, device_ids=[0,1])
        unet = unet.cuda()

        # discriminator = Discriminator(in_channels=discriminator_input , use_sigmoid=True)
        discriminator = Fusion_Discriminator(input_nc=discriminator_input)
        discriminator = nn.DataParallel(discriminator, device_ids=[0,1])
        discriminator = discriminator.cuda()

        criterion = nn.MSELoss()
        adversarial_loss = AdversarialLoss(type='hinge')
        l1_loss = nn.L1Loss()
        content_loss = ContentLoss()
        ssim = SSIM(window_size = 11)
        bce = nn.BCELoss()

        self.add_module('unet', unet)
        self.add_module('discriminator', discriminator)

        self.add_module('criterion', criterion)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('l1_loss', l1_loss)
        self.add_module('content_loss', content_loss)
        self.add_module('ssim_loss', ssim)
        self.add_module('bce_loss', bce)
        

        self.unet_optimizer = optim.Adam(
            unet.parameters(), 
            lr = float(0.002),
            betas=(0.9, 0.999)
            )

        self.dis_optimizer = optim.Adam(
             params=discriminator.parameters(),
             lr=float(0.002),
             betas=(0.9, 0.999)
             )

        self.unet_input = unet_input
        self.unet_output = unet_output
        self.discriminator_input = discriminator_input


    def load(self, path_unet, path_discriminator):
        weight_unet = torch.load(path_unet)
        weight_discriminator = torch.load(path_discriminator)
        self.unet.load_state_dict(weight_unet)
        self.discriminator.load_state_dict(weight_discriminator)

    def save_weight(self, path_unet, path_dis):
        torch.save(self.unet.state_dict(), path_unet)
        torch.save(self.discriminator.state_dict(), path_dis)

    def process(self, haze_images, dehaze_images): 

        # zero optimizers
        self.unet_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # find output and initialize loss to zero
        unet_loss = 0
        dis_loss = 0

        outputs = self.unet(haze_images.cuda())
        print("G(I) shape:", outputs.shape)

        # unet loss
        unet_fake = self.discriminator(outputs.cuda())        
        unet_gan_loss = self.adversarial_loss(unet_fake, True, False) * 0.7
        unet_loss += unet_gan_loss

        unet_criterion = self.criterion(outputs.cuda(), dehaze_images.cuda())
        unet_loss += unet_criterion




        gen_content_loss = self.content_loss(outputs.cuda(), dehaze_images.cuda())
        gen_content_loss = (gen_content_loss * 0.7).cuda()
        unet_loss += gen_content_loss.cuda()
        
        
        ssim_loss =  self.ssim_loss(outputs.cuda(), dehaze_images.cuda())
        ssim_loss = (1-ssim_loss)*2
        unet_loss += ssim_loss.cuda()

        # Original Discriminator
        # discriminator loss
        dis_real = self.discriminator(dehaze_images.cuda())        
        dis_fake = self.discriminator(outputs.detach().cuda())       
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        return unet_loss, dis_loss, unet_criterion, 1-ssim_loss/2


    # def backward_dis(self, dis_loss):
    #     dis_loss.backward(retain_graph = True)
    #     self.dis_optimizer.step()
        

    # def backward_unet(self, unet_loss):
    #     unet_loss.backward()
    #     self.unet_optimizer.step()
    
    def backward(self, unet_loss, dis_loss):
        # """ This resolved the large black hole issue """
        # loss = unet_loss + dis_loss
        # loss.backward()

        # dis_loss.backward(retain_graph = True)
        # unet_loss.backward()
        
        # self.dis_optimizer.step()
        # self.unet_optimizer.step()

        unet_loss.backward()
        self.unet_optimizer.step()
        
        dis_loss.backward(retain_graph = True)
        self.dis_optimizer.step()
        

    def predict(self, haze_images):
        predict_mask = self.unet(haze_images.cuda())
        return predict_mask