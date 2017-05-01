from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

import tensorflow as tf
from keras.applications import vgg19
import keras.backend.tensorflow_backend as K


base_image_path = 'Tuebingen_Neckarfront.jpg'
combination_image_path = 'everfilter.jpg'

result_prefix = 'attempt2-result'
iterations = 15

# these are the weights of the different loss components
total_variation_weight = 10
style_weight = 1000
content_weight = 1

# dimensions of the generated picture.
width, height = load_img(base_image_path).size
img_nrows = height
img_ncols = int(width * img_nrows / height)

# util function to open, resize and format pictures into appropriate tensors


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# util function to convert a tensor into a valid image


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


####################################################################

base_image = K.variable(preprocess_image(base_image_path))
# style_image = K.variable(preprocess_image(base_image_path))
combination_image_ref = K.variable(preprocess_image(combination_image_path))

# NOTE: check K.image_data_format() == channels_first
if K.image_data_format() == 'channels_first':
    style_image = K.placeholder((1, 3, img_nrows, img_ncols))
    combination_image_gen = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    style_image = K.placeholder((1, img_nrows, img_ncols, 3))
    combination_image_gen = K.placeholder((1, img_nrows, img_ncols, 3))


input_tensor = K.concatenate([base_image,
                              style_image,
                              combination_image_gen], axis=0)

print("Create [INPUT TENSOR] Successful!")

model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

######################################################################

# loss_on_combination
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def loss_combination():
    layer_features = outputs_dict['block1_conv2']
    base_image_features   = layer_features[0, :, :, :]
    style_image_features  = layer_features[1, :, :, :]
    comb_gen_features     = layer_features[2, :, :, :] 
    # comb_ref_features     = layer_features[3, :, :, :]

    # Original loss function
    loss = K.variable(0.)
    loss += content_weight * content_loss(base_image_features, comb_gen_features)

    feature_layers = ['block1_conv1', 'block2_conv1',
                     'block3_conv1', 'block4_conv1',
                     'block5_conv1']
    for layer_name in feature_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(feature_layers)) * sl

    loss += total_variation_weight * total_variation_loss(combination_image_gen)

    #loss += content_weight * content_loss(comb_ref_features, comb_gen_features)
    return loss

loss_variable = loss_combination()
outputs = [loss_variable]

grads = K.gradients(loss_variable, combination_image_gen)
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)


grads = K.gradients(loss_variable, style_image)
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image_gen, style_image], outputs)


###################################################################

def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((2, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((2, img_nrows, img_ncols, 3))

    x1 = x(1, :, ;, :)
    x2 = x(2, :, :, :)
    
    outs = f_outputs([x1, x2])
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
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

if K.image_data_format() == 'channels_first':
    x = np.random.uniform(0, 255, (2, 3, img_nrows, img_ncols)) - 128.
else:
    x = np.random.uniform(0, 255, (2, img_nrows, img_ncols, 3)) - 128.

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x_style.flatten(),
                                           fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val, info)
    # save current generated image
    # img = deprocess_image(x.copy())
    # fname = result_prefix + '_at_iteration_%d.png' % i
    # imsave(fname, img)
    # end_time = time.time()
    # print('Image saved as', fname)
    # print('Iteration %d completed in %ds' % (i, end_time - start_time))




# content loss = L2-norm(base, comb_generated)
# style loss.  = gram(style_gen, comb_generated)
# generated_loss = L2-norm(comb_generated, comb_ref)
