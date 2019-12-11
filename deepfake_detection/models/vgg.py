import keras
from keras.applications import VGG16, VGG19

from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.models import Model

def vgg(backbone='vgg16', inputs=None, modifier=None, **kwargs):
    """ Constructs a m2det model using a vgg backbone.
    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
        inputs: The inputs to the network (defaults to a Tensor of shape (256, 256, 3)).
        modifier: A function handler which can modify the backbone before using it in m2det (this can be used to freeze backbone layers for example).
    Returns
        RetinaNet model with a VGG backbone.
    """
    # choose default input
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, 256, 256))
        else:
            inputs = keras.layers.Input(shape=(256, 256, 3))
    # create the vgg backbone
    if backbone == 'vgg16':
        vgg = VGG16(input_tensor=inputs, include_top=False, weights='imagenet', classes=1)
    elif backbone == 'vgg19':
        vgg = VGG19(input_tensor=inputs, include_top=False, weights='imagenet', classes=1)
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    if modifier:
        vgg = modifier(vgg)
    x = vgg.output

    vgg = AveragePooling2D(pool_size=8)(x)
    vgg = Flatten()(vgg)

    outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(vgg)
    model = Model(inputs=inputs, outputs=outputs)

    # create the full model
    return model
