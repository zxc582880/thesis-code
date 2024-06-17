import subprocess
from PIL import Image
import torch
from torchvision import transforms
from pytorch_fid import fid_score
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from lpips import LPIPS
import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import re
import os


def load_image(image_path):
    return Image.open(image_path).convert('RGB')


def calculate_psnr_ssim(image1, image2):
    image1 = np.array(image1)
    image2 = np.array(image2)

    psnr = peak_signal_noise_ratio(image1, image2)
    ssim = structural_similarity(image1, image2, multichannel=True)

    return psnr, ssim

def calculate_lpips_kl(image_path1, image_path2):
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)

    lpips = calculate_lpips(image1, image2)
    kl_divergence = calculate_kl_divergence(image1, image2)

    return lpips, kl_divergence

def calculate_lpips(image1, image2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = LPIPS(net='alex', verbose=0).to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image1 = transform(image1).unsqueeze(0).to(device)
    image2 = transform(image2).unsqueeze(0).to(device)

    lpips_value = lpips_model(image1, image2).item()

    return lpips_value

def calculate_kl_divergence(image1, image2):
    histogram1 = calculate_histogram(image1)
    histogram2 = calculate_histogram(image2)

    kl_divergence_value = entropy(histogram1, histogram2)

    return kl_divergence_value

def calculate_histogram(image):
    image_array = np.array(image)
    histogram, _ = np.histogram(image_array.flatten(), bins=256, range=[0, 256])
    histogram = histogram / float(np.sum(histogram))

    return histogram
def gram_matrix(feature_map):
    # Ensure the feature map has 3 dimensions
    assert len(feature_map.shape) == 3, "Input tensor should have 3 dimensions (H, W, C)"

    # Reshape the feature map to (H*W, C)
    shape = tf.shape(feature_map)
    reshaped_map = tf.reshape(feature_map, (shape[0] * shape[1], shape[2]))

    # Calculate the Gram matrix
    gram = tf.matmul(reshaped_map, reshaped_map, transpose_a=True)

    # Normalize the Gram matrix by the number of elements
    num_elements = tf.cast(shape[0] * shape[1] * shape[2], tf.float32)
    gram /= num_elements

    return gram

for j in range(ord('H'), ord('X')+1):
    # results_path = "D:\\stargan-master\\ABSTUDY_2\\evaluation.txt"
    # with open(results_path, 'a') as file:
    #     file.write(f'ABSTUDY Group AB{chr(j)}:\n')
    os.system('cls')
    for i in range(3):
        numbers=['2','3','4']

        image1_path='D:\\stargan-master\\filter_Batch2\\filtertest\\1\\000000062805_resized.jpg'##原始內容/不用改
        dataset1_path='D:\\stargan-master\\filter_Batch2\\filtertest\\1'##原始內容/不用改


        image2_path='D:\\stargan-master\\filter_Batch2\\filtertest\\'+numbers[i]+'\\t.jpg'##風格 2>3>4
        dataset2_path='D:\\stargan-master\\filter_Batch2\\filtertest\\' +numbers[i] ##風格 2>3>4

        # path=f'D:\\stargan-master\\ABSTUDY_2\\stargan_filterAB{chr(j)}\\'
        path = 'D:\\stargan-master\\stargan_filter\\'

        image3_path = path +'eva\\1-'+numbers[i]+'_image.jpg'##生成內容 PRO>ORI>SA/1-2>1-3>1-4
        dataset3_path = path +'eva\\fid\\'+ numbers[i] ##生成內容 PRO>ORI>SA/2>3>4

        # print(image2_path+'\n'+image3_path+'\n'+dataset2_path+'\n'+dataset3_path)
        def evaluation(image1,dataset1,image2,image3,dataset3):

            # 計算1跟2之間的 FID、PSNR、SSIM
            psnr_31, ssim_31 = calculate_psnr_ssim(load_image(image3), load_image(image1))


            # 計算2跟3之間的 LPIPS、KL 散度
            lpips_32 = calculate_lpips_kl(image3, image2)

            def run_fid_calculation(x, y):
                command = f"python -m pytorch_fid {x} {y}"
                process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
                fid, error = process.communicate()

                if error:
                    print(f"An error occurred: {error}")
                else:
                    fid_score=fid.decode('utf-8')
                    numbers = re.findall(r'\d+\.\d+', fid_score)
                    fid_score_numeric = float(''.join(numbers))
                return fid_score_numeric





            # print('FID:')
            fid_scores=run_fid_calculation(dataset3,dataset1)


            # 輸出結果
            # print(f"PSNR (3, 1): {psnr_31}, SSIM (3, 1): {ssim_31}")

            # print(f"LPIPS (3, 2): {lpips_32}")
            # Load the VGG16 model pre-trained on ImageNet data
            base_model = VGG16(weights='imagenet', include_top=False)

            # Choose the layer from which you want to extract features
            selected_layer = base_model.get_layer('block4_conv2')

            # Create a new model that outputs the features from the selected layer
            feature_extractor_model = Model(inputs=base_model.input, outputs=selected_layer.output)

            # Load and preprocess the style image
            style_img = image.load_img(image2, target_size=(224, 224))
            style_img = image.img_to_array(style_img)
            style_img = np.expand_dims(style_img, axis=0)
            style_img = preprocess_input(style_img)

            # Load and preprocess the generated image
            generated_img = image.load_img(image3, target_size=(224, 224))
            generated_img = image.img_to_array(generated_img)
            generated_img = np.expand_dims(generated_img, axis=0)
            generated_img = preprocess_input(generated_img)

            # Extract features from the selected layer for both style and generated images
            style_features = feature_extractor_model.predict(style_img)
            generated_features = feature_extractor_model.predict(generated_img)

            # Calculate Gram matrices for style features of the content and style images
            Gram_style = gram_matrix(style_features[0])  # Assuming style_features is a list, use the first element
            Gram_generated = gram_matrix(generated_features[0])  # Assuming generated_features is a list, use the first element

            # Now you can use the Gram matrices for further analysis, such as evaluating style difference
            style_difference = (tf.reduce_sum(tf.square(Gram_style - Gram_generated)))/1000000
            euclidean_distance = ((tf.sqrt(tf.reduce_sum(tf.square(Gram_style - Gram_generated)))))/1000


            # Print the style difference
            # print("Style Difference:", style_difference.numpy())


            
            print(f"PSNR{i+1}: {psnr_31}, SSIM{i+1}: {ssim_31}, FID{i+1}:{fid_scores},LPIPS{i+1}:{lpips_32}, GRAM{i+1}:{style_difference.numpy()}, Gram-Eu{i+1}:{euclidean_distance.numpy()}")
            # results_path = "D:\\stargan-master\\ABSTUDY_2\\evaluation.txt"
            # with open(results_path, 'a') as file:
            #     file.write(f'PSNR{i+1}: {psnr_31}, SSIM{i+1}: {ssim_31}, FID{i+1}:{fid_scores},LPIPS{i+1}:{lpips_32}, GRAM{i+1}:{style_difference.numpy()}, Gram-Eu{i+1}:{euclidean_distance.numpy()}\n')
        evaluation(image1_path,dataset1_path,image2_path,image3_path,dataset3_path) 



# 輸出結果