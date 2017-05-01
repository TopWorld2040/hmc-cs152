from scipy.misc import imread, imresize, imsave, fromimage, toimage
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import time
import argparse
import warnings

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
import keras.backend.tensorflow_backend as K
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model

THEANO_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

TH_19_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels_notop.h5'
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
parser.add_argument('--rescale_image', dest='rescale_image', default="False", type=str, help="Rescale image after execution to original dimensions")
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

style_mask_present = args.style_masks is not None
mask_paths = []
if style_mask_present:
    for mask_path in args.style_masks:
        mask_paths.append(mask_path)
    assert len(style_image_paths) == len(mask_paths), "Wrong number of style mask provided."

color_mask_present      = args.color_mask is not None
rescale_image           = str_to_bool(args.rescale_image)
maintain_aspect_ratio   = str_to_bool(args.maintain_aspect_ratio)
preserve_color          = str_to_bool(args.color)

content_weight          = args.content_weight
total_variation_weight  = args.tv_weight
style_weights           = []
for style_weight in args.style_weight:
    style_weights.append(style_weight)

# Define pooling type
pooltype                = str(args.pool).lower()
assert pooltype in ['ave', 'max'], 'Pooling argument is wrong.'
pooltype = 1 if pooltype == 'ave' else 0

# Read mode
read_mode               = 'gray' if args.init_image == 'gray' else 'color'

# Dimension of the generated picture
img_width               = 0
img_height              = 0
img_WIDTH               = 0
img_HEIGHT              = 0
aspect_ratio            = 0

# content loss type
assert args.content_loss_type in [0, 1, 2], "Content loss type error"

# util function to open, resize, and format pictures into appropriate tensors
def preprocess_image(image_path, load_dims=False, read_mode="color"):
    global img_width, img_height, img_WIDTH, img_HEIGHT, aspect_ratio

    mode = "RGB" if read_mode == 'color' else "L"
    img = imread(image_path, mode=mode)

    if mode == 'L':
        # expand the 1 channel grayscale to 3 channel grayscale image 
        temp = np.zeros(img.shape + (3,), dtype=np.uint8)
        temp[:, :, 0] = img
        temp[:, :, 1] = img
        temp[:, :, 2] = img
        img = temp

    if load_dims:
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = float(img_HEIGHT) / img_WIDTH
        
        img_width = args.img_size
        if maintain_aspect_ratio:
            img_height = int(img_width * aspect_ratio)
        else:
            img_height = args.img_size

    img = imresize(img, (img_width, img_height)).astype('float32')

    # RGB -> BGR
    img = img[:, :, ::-1]
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    if K.image_dim_ordering() == 'th':
        img = img.transpose((2, 0, 1)).astype('float32')
    img = np.expand_dims(img, axis=0)
    return img

# util function to convert a tensor back to an image
def deprocess_image(x):
    """
    Convert a tensor back to a valid image.
    """
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, img_width, img_height))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_width, img_height, 3))
    
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # BGR => RGB
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


# util function to preserve an image color
def original_color_transform(content, generated, mask=None):
    generated = fromimage(toimage(generated, mode="RGB"), mode="YCbCr")

    if mask is None:
        generated[:, :, 1:] = content[:, :, 1:] # Generated CbCr =  Content CbCr
    else:
        width, height, channels = generated.shape

        for i in range(width):
            for j in range(height):
                if mask[i, j] == 1:
                    generated[i, j, 1:] = content[i, j, 1:]
    
    generated = fromimage(toimage(generated, mode="YCbCr"), mode="RGB")
    return generated

# load mask
def load_mask(mask_path, shape, return_mask_img=False):
    if K.image_dim_ordering() == 'th':
        _, channels, width, height = shape
    else:
        _, width, height, channels = shape

    mask = imread(mask_path, mode="L") # Grayscale mask load
    mask = imresize(mask, (width, height)).astype('float32')

    # Perform binarization of mask
    mask[mask <= 127] = 0
    mask[mask > 127 ] = 255

    max_val = np.amax(mask)
    mask /= max_val

    if return_mask_img:
        return mask

    mask_shape  = shape[1:]
    mask_tensor = np.empty(mask_shape)

    for i in range(channels):
        if K.image_dim_ordering() == "th":
            mask_tensor[i, :, :] = mask
        else:
            mask_tensor[:, :, i] = mask
    
    return mask_tensor

# pooling function
def pooling_func(x):
    if pooltype == 1:
        return AveragePooling2D((2, 2), strides=(2, 2))(x)
    else:
        return MaxPooling2D((2, 2), strides=(2, 2))(x)

# get tensor representation of base image
base_image = K.variable(preprocess_image(base_image_path, True, read_mode=read_mode))
style_reference_images = []
for style_path in style_image_paths:
    style_reference_images.append(K.variable(preprocess_image(style_path)))

# this will contain our generated image
if K.image_dim_ordering() == 'th':
    combination_image = K.placeholder((1, 3, img_width, img_height))
else:
    combination_image = K.placeholder((1, img_width, img_height, 3))

image_tensor = [base_image]
for style_image_tensor in style_reference_images:
    image_tensor.append(style_image_tensor)
image_tensor.append(combination_image)

nb_tensors = len(image_tensor)
nb_style_images = nb_tensors - 2 # Content andoutput image not considered

# combine the various image into a single Keras tensor
input_tensor = K.concatenate(image_tensor, axis=0)

if K.image_dim_ordering() == 'th':
    shape = (nb_tensors, 3, img_width, img_height)
else:
    shape = (nb_tensors, img_width, img_height, 3)

ip = Input(tensor=input_tensor, shape=shape)

# build a network
x = Conv2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(ip)
x = Conv2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(x)
x = pooling_func(x)

x = Conv2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(x)
x = pooling_func(x)

x = Conv2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(x)
if args.model == 'vgg19':
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_4', padding="same")(x)
x = pooling_func(x)

x = Conv2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(x)
if args.model == 'vgg19':
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_4', padding="same")(x)
x = pooling_func(x)

x = Conv2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(x)
if args.model == 'vgg19':
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_4', padding="same")(x)
x = pooling_func(x)

model = Model(ip, x)

# Get the weights 
if K.image_dim_ordering() == "th":
    if args.model == "vgg19":
        weights = get_file('vgg19_weights_th_dim_ordering_th_kernels_notop.h5', TH_19_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    else:
        weights = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5', THEANO_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
else:
    if args.model == "vgg19":
        weights = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_19_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    else:
        weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')

model.load_weights(weights)

# Convert kernels to tensorflow
if K.image_dim_ordering() == 'th':
    warnings.warn('You are using the TensorFlow backend, yet you are using Theano ordering')
    convert_all_kernels_in_model(model)

print('Model loaded.')

# get the symbolic outputs of each 'key' layer (we gave them unique names).
output_dicts = dict([(layer.name, layer.output) for layer in model.layers])
shape_dicts = dict([(layer.name, layer.output_shape) for layer in model.layers])

# compute the neural style loss
# first we need to define 4 util functions

# Improvement 1
# the gram matrix of an image tensor (feature-wise outer product) using shifted activations
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_dim_ordering() == 'th':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features - 1, K.transpose(features - 1))
    return gram

# the 'style loss' is designed to maintain the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of feature maps from the style reference image
# and from the generated image
def style_loss(style, combination, mask_path=None, nb_channels=None):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3

    if mask_path is not None:
        style_mask = K.variable(load_mask(mask_path, nb_channels))

        style = style * K.stop_gradient(style_mask)
        combination = combination * K.stop_gradient(style_mask)

        del style_mask
    
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_width * img_height
    return K.sum( K.square(S - C) ) / (4.0 * (channels ** 2) * (size ** 2))

# an auxiliary loss function designed to main the 'content' of the base image in the generated image
def content_loss(base, combination):
    channel_dim = 0 if K.image_dim_ordering() == 'th' else -1

    channels = K.shape(base)[channel_dim]
    size = img_width * img_height

    if args.content_loss_type == 1:
        multiplier = 1 / (2.0 * (channels ** 0.5) * (size ** 0.5))
    elif args.content_loss_type == 2:
        multiplier = 1 / (channels * size)
    else:
        multiplier = 1.0

    return multiplier * K.sum(K.square(combination - base)) 

# the 3rd loss function, total variation loss, is designed to keep the generated image 
# locally coherent
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_dim_ordering() == 'th':
        a = K.square(x[:, :, :img_width-1, :img_height-1] - x[:, :, 1:, :img_height-1])
        b = K.square(x[:, :, :img_width-1, :img_height-1] - x[:, :, :img_width-1, 1:])
    else:
        a = K.square(x[:, :img_width-1, :img_height-1, :] - x[:, 1:, :img_height-1, :])
        b = K.square(x[:, :img_width-1, :img_height-1, :] - x[:, :img_width-1, 1:, :])
    return K.sum(K.pow(a+b, 1.25))

if args.model == 'vgg19':
    feature_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                      'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
else:
    feature_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                      'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']

# combine these loss functions into a single scalar
loss = K.variable(0.)
layer_features = output_dicts[args.content_layer]
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[nb_tensors - 1, :, :, :]
loss += content_weight * content_loss(base_image_features, combination_features)

# Improvement 2
# Use all layers for style feature extraction and reconstruction
nb_layers = len(feature_layers) - 1

style_masks = []
if style_mask_present:
    style_masks = mask_paths # If mask present, pass dictionary of masks to style loss
else:
    style_masks = [None for _ in range(nb_style_images)]

channel_index = 1 if K.image_dim_ordering() == 'th' else -1
             
# Improvement 3
# Chained Inference without blurring
for i in range(len(feature_layers) - 1):
    layer_features = output_dicts[feature_layers[i]]
    shape = shape_dicts[feature_layers[i]]
    combination_features = layer_features[nb_tensors - 1, :, :, :]
    style_reference_features = layer_features[1:nb_tensors - 1, :, :, :]
    sl1 = []
    for j in range(nb_style_images):
        sl1.append(style_loss(style_reference_features[j], combination_features, style_masks[j], shape))

    layer_features = output_dicts[feature_layers[i + 1]]
    shape = shape_dicts[feature_layers[i + 1]]
    combination_features = layer_features[nb_tensors - 1, :, :, :]
    style_reference_features = layer_features[1:nb_tensors-1, :, :, :]
    sl2 = []
    for j in range(nb_style_images):
        sl2.append(style_loss(style_reference_features[j], combination_features, style_masks[j], shape))
    
    for j in range(nb_style_images):
        sl = sl1[j] - sl2[j]

        # Improvement 4
        # Geometric weighted scaling of style loss
        loss += (style_weights[j] / ( 2 ** (nb_layers - (i + 1)))) * sl

loss += total_variation_weight * total_variation_loss(combination_image)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)

outputs = [loss]
if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)

def eval_loss_and_grads(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((1, 3, img_width, img_height))
    else:
        x = x.reshape((1, img_width, img_height, 3))
    
    outs = f_outputs([x])
    loss_value = outs[0] 
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grads_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grads_values)
        self.loss_value = None
        self.grads_values = None
        return grad_values

evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image so as to minimize the neural style loss
if "content" in args.init_image or "gray" in args.init_image:
    x = preprocess_image(base_image_path, True, read_mode=read_mode)
elif "noise" in args.init_image:
    x = np.random.uniform(0, 255, (1, img_width, img_height, 3)) - 128.0
    if K.image_dim_ordering() == 'th':
        x = x.transpose((0, 3, 1, 2))
else:
    print("Using initial image : ", args.init_image)
    x = preprocess_image(args.init_image, read_mode=read_mode)

# We require original image if we are to receive color in YCbCr mode
if preserve_color:
    content = imread(base_image_path, mode="YCbCr")
    content = imresize(content, (img_width, img_height))

    if color_mask_present:
        if K.image_dim_ordering() == 'th':
            color_mask_shape = (None, None, img_width, img_height)
        else:
            color_mask_shape = (None, img_width, img_height, None)
        color_mask = load_mask(args.color_mask, color_mask_shape, return_mask_img=True)
    else:
        color_mask = None
else:
    color_mask = None

num_iter = args.num_iter 
prev_min_val = -1

improvement_threshold = float(args.min_improvement)
list_loss = []

for i in range(num_iter):
    print("Starting iteration {} of {}".format(i+1, num_iter))
    start_time = time.time()

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
    list_loss.append(min_val)

    with open('cost_trace.txt','w') as output_cost_trace:
        for each in list_loss:
            output_cost_trace.write(str(format(each, '.20f')) + '\n')

    if prev_min_val == -1:
        prev_min_val = min_val
    
    improvement = (prev_min_val - min_val) / prev_min_val * 100

    print("Current loss value: ", min_val, " Improvement : %0.3f" % improvement, "%")
    prev_min_val = min_val

    # save current generated image
    img = deprocess_image(x.copy())

    if preserve_color and content is not None:
        img = original_color_transform(content, img, mask=color_mask)
    
    if not rescale_image:
        img_ht = int(img_width * aspect_ratio)
        print("Rescaling Image to ({}, {})".format(img_width, img_ht))
        img = imresize(img, (img_width, img_ht), interp=args.rescale_method)

    if rescale_image:
        print("Rescaling Image to ({}, {})".format(img_WIDTH, img_HEIGHT))
        img = imresize(img, (img_WIDTH, img_HEIGHT), interp=args.rescale_method)

    fname = result_prefix + "_at_iteration_{}.png".format(i + 1)

    imsave(fname, img)

    end_time = time.time()
    print("Image saved as", fname)
    print("Iteration %d completed in %ds " % (i+1, end_time - start_time))

    if improvement_threshold is not 0.0:
        if improvement < improvement_threshold and improvement is not 0.0:
            print("Improvement is less than threshold.")
            exit()
            
