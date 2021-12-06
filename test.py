from libraries import *
from DU_Net import DU_Net
from DoubleGANNet import DoubleGANNet

graph_gloss = []
input_unet_channel = 3
output_unet_channel = 3
input_dis_channel = 3
max_epochs = 100
# DUNet = DU_Net(input_unet_channel ,output_unet_channel ,input_dis_channel).cuda()
DUNet = DoubleGANNet(input_unet_channel, output_unet_channel, 7, 0, 0).cuda()

""" Pre-trained weights """
# path_of_generator_weight = 'pretrained_weights/outdoor/generator.pth'  #path where the weights of genertaor are stored
# path_of_discriminator_weight = 'pretrained_weights/outdoor/discriminator.pth'  #path where the weights of discriminator are stored

""" BPPNet trained by us """
# path_of_generator_weight = 'weights_211201/generator_300.pth'  #path where the weights of genertaor are stored
# path_of_discriminator_weight = 'weights_211201/discriminator_300.pth'  #path where the weights of discriminator are stored

""" First hyper-parameter version of our method """
path_of_generator_weight = 'ours_weights_211204/generator_500.pth'  #path where the weights of genertaor are stored
path_of_discriminator_weight = 'ours_weights_211204/discriminator_500.pth'  #path where the weights of discriminator are stored

""" Second hyper-parameter version of our method """
# path_of_generator_weight = 'rand_lr_ours_weights/generator_250.pth'  #path where the weights of genertaor are stored
# path_of_discriminator_weight = 'rand_lr_ours_weights/discriminator_250.pth'  #path where the weights of discriminator are stored

# path_of_generator_weight = 'pretrained_weights/outdoor/generator.pth'  #path where the weights of genertaor are stored
# path_of_discriminator_weight = 'pretrained_weights/outdoor/discriminator.pth'  #path where the weights of discriminator are stored

DUNet.load(path_of_generator_weight,path_of_discriminator_weight)


def to_tensor(img):
    img_t = F6.to_tensor(img).float()
    return img_t

def postprocess(img):
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()


path_of_test_hazy_images = 'O-Haze/train/haze/*'
path_for_resultant_dehaze_images = 'results/train_ours_211204_500/'

# path_of_test_hazy_images = 'NTIRE2021_NH-Haze2/haze/*'
# path_for_resultant_dehaze_images = 'NTIRE2021_results/ours_500/'

# path_of_test_hazy_images = 'Dense-Haze/haze/*'
# path_for_resultant_dehaze_images = 'Dense-Haze_results/ours_300/'

# path_for_resultant_dehaze_images = 'results/pretrained/'
os.mkdir(path_for_resultant_dehaze_images)
image_paths_test_hazy=glob.glob(path_of_test_hazy_images)
print(image_paths_test_hazy)

for i in range(len(image_paths_test_hazy)):
    haze_image = cv2.imread(image_paths_test_hazy[i])
    haze_image = Img.fromarray(haze_image)
    haze_image = haze_image.resize((512,512), resample=PIL.Image.BICUBIC)
    haze_image = np.array(haze_image)
    haze_image = cv2.cvtColor(haze_image, cv2.COLOR_BGR2YCrCb)
    haze_image = to_tensor(haze_image).cuda()
    haze_image = haze_image.reshape(1,3,512,512)

    dehaze_image = DUNet.predict(haze_image) 
    
    dehaze_image = postprocess(dehaze_image)[0]
    dehaze_image = dehaze_image.cpu().detach().numpy()
    dehaze_image = dehaze_image.astype('uint8')
    dehaze_image = dehaze_image.reshape(512,512,3)
    dehaze_image = cv2.cvtColor(dehaze_image, cv2.COLOR_YCrCb2BGR)
    print(path_for_resultant_dehaze_images+image_paths_test_hazy[i].split('_')[0][-2:]+'.png')
    cv2.imwrite(path_for_resultant_dehaze_images+image_paths_test_hazy[i].split('_')[0][-2:]+'.png', dehaze_image)

    # print(path_for_resultant_dehaze_images+image_paths_test_hazy[i].split('.png')[0][-2:]+'.png')
    # cv2.imwrite(path_for_resultant_dehaze_images+image_paths_test_hazy[i].split('.png')[0][-2:]+'.png', dehaze_image)

