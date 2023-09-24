import os
import gdown
import tensorflow as tf

# !mkdir checkpoints
# !mkdir masks

tf.io.gfile.makedirs('/content/PDS/checkpoints')
tf.io.gfile.makedirs('/content/PDS/masks')

cwd = os.getcwd()
url_24 = 'https://drive.google.com/uc?id=1JInV8bPGy18QiIzZcS1iECGHCuXL6_Nz'
url_8 = 'https://drive.google.com/uc?id=1le6lIosRGjnraM-HBOsuA1zcqDV8DB76'
url_12 = 'https://drive.google.com/uc?id=1yS8QZb_6tCeZkY7DK4_RI-Crc6LQLILN' # deep CIFAR10

url_mask_cifar10 = 'https://drive.google.com/uc?id=1tz-ObG-WRFGNFDxnJW6YgtzhKYhDWBS9'

gdown.download(url_24, '/content/PDS/checkpoints/checkpoint_24.pth', quiet=False)
gdown.download(url_8, '/content/PDS/checkpoints/checkpoint_8.pth', quiet=False)
gdown.download(url_12, '/content/PDS/checkpoints/checkpoint_12.pth', quiet=False)

gdown.download(url_mask_cifar10, '/content/PDS/masks/cifar10_freq', quiet=False)


tf.io.gfile.makedirs('/content/PDS/assets/stats')
url_stats_cifar10 = 'https://drive.google.com/uc?id=14UB27-Spi8VjZYKST3ZcT8YVhAluiFWI'
gdown.download(url_stats_cifar10, '/content/PDS/assets/stats/cifar10_stats.npz', quiet=False)