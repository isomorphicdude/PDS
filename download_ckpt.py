import os
import gdown
import tensorflow as tf
from absl import flags
import sys

# !mkdir checkpoints
# !mkdir masks

flags.DEFINE_string("workdir", None, "Work directory.")
FLAGS = flags.FLAGS
FLAGS(sys.argv)

tf.io.gfile.makedirs(os.path.join(FLAGS.workdir, 'checkpoints'))
tf.io.gfile.makedirs(os.path.join(FLAGS.workdir, 'masks'))

cwd = os.getcwd()
url_24 = 'https://drive.google.com/uc?id=1JInV8bPGy18QiIzZcS1iECGHCuXL6_Nz'
url_8 = 'https://drive.google.com/uc?id=1le6lIosRGjnraM-HBOsuA1zcqDV8DB76'
url_12 = 'https://drive.google.com/uc?id=1yS8QZb_6tCeZkY7DK4_RI-Crc6LQLILN' # deep CIFAR10

url_mask_cifar10 = 'https://drive.google.com/uc?id=1tz-ObG-WRFGNFDxnJW6YgtzhKYhDWBS9'

gdown.download(url_24, os.path.join(FLAGS.workdir, 'checkpoints', 'checkpoint_24.pth'), quiet=False)
gdown.download(url_8, os.path.join(FLAGS.workdir, 'checkpoints', 'checkpoint_8.pth'), quiet=False)
gdown.download(url_12, os.path.join(FLAGS.workdir, 'checkpoints', 'checkpoint_12.pth'), quiet=False)
gdown.download(url_mask_cifar10, os.path.join(FLAGS.workdir, 'masks', 'cifar10_freq'), quiet=False)

tf.io.gfile.makedirs(os.path.join(FLAGS.workdir, 'assets', 'stats'))
url_stats_cifar10 = 'https://drive.google.com/uc?id=14UB27-Spi8VjZYKST3ZcT8YVhAluiFWI'
gdown.download(url_stats_cifar10, os.path.join(FLAGS.workdir, 'assets', 'stats', 'cifar10_stats.npz'), quiet=False)