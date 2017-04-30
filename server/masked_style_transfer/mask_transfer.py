from scipy.misc import imread, imresize, imsave, fromimage, toimage
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import time
import argparse
import warnings

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
import keras.backend.tensorflow_backend as K
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model

TF_19_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

parser = argparse.ArgumentParser(description='Neural style transfer')
parser.add_argument('base_image_path', metavar='base', type=str, help='Path to image to transform')
parser.add_argument('style_image_paths', metavar='ref', nargs='+', type=str, help='Path to the reference images')
parser.add_argument('result_prefix', metavar='res_prefix', type=str, help='Prefix for the output')
parser.add_argument('--style_masks', type=str, default=None, nargs='+', help='Masks for style images')
parser.add_argument('--color_mask', type=str, default=None, help='Mask for color preservation')
parser.add_argument('--image_size', dest="img_size", default=400, type=int, help="Minimum image size")
parser.add_argument('--content_weight', dest='content_weight', default=0.025, type=float, help="Weight of content")
parser.add_argument('--style_weight', dest="style_weight", nargs='+', default=[1], type=float, help="Weight of style, can be multiple for multiple styles")
parser.add_argument('--total_variation_weight', dest='tv_weight', default=8.5e-5, type=float, help="Total Variation weight")
parser.add_argument('--num_iter', dest="num_iter", default=10, type=int, help="Number of iterations")
parser.add_argument('--model', default="vgg19", type=str, help="Choices are 'vgg16' and 'vgg19'")
parser.add_argument('--content_loss_type', default=0, type=int, help='Can be one of 0, 1, or 2. See Readme')
parser.add_argument('--rescale_method', dest='rescale_method', default='bilinear', type=str, help='Rescale image algorithm')
parser.add_argument('--maintain_aspect_ratio', dest='maintain_aspect_ratio', default="True", type=str, help="Maintain aspect ratio of loaded images")
parser.add_argument('--content_layer', dest="content_layer", default="conv4_2", type=str, help="Content layer used for content loss.")
parser.add_argument('--init_image', dest='init_image', default='noise', type=str, help="Initial image used to generate the final image. Options are 'content', 'noise', or 'gray'")
parser.add_argument('--pool_type', dest='pool', default='max', type=str, help="Pooling type. Can be 'ave' for average pooling or 'max' for max pooling")
parser.add_argument('--preserve_color', dest='color', default='False', type=str, help="Preserve original color in image")
parser.add_argument('--min_improvement', default=0.0, type=float, help='Defines minimum improvment required to continue script')

def str_to_bool(v):
    return v.lower() in ("true", "yes", "t", "1")

args = parser.parse_args()
base_image_path             = args.base_image_path
style_reference_image_paths = args.style_image_paths
result_prefix               = args.result_prefix

style_image_paths = []
for style_image_path in style_reference_image_paths:
    style_image_paths.append(style_image_path)
