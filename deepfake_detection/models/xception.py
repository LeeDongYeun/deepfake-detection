import keras
from keras.applications.xception import Xception

from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import GlobalAveragePooling2D, Input
from keras.models import Model


def xception(backbone='xception', inputs=None, modifier=None, **kwargs):
    """ Constructs a m2det model using a Xception backbone.
    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (Xception)
        inputs: The inputs to the network (defaults to a Tensor of shape (256, 256, 3)).
        modifier: A function handler which can modify the backbone.
    Returns
        XceptionNet.
    """
    # choose default input
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, 256, 256))
        else:
            inputs = keras.layers.Input(shape=(256, 256, 3))

    # create the resnet backbone
    if backbone == 'xception':
        xceptionNet = Xception(include_top=False, weights='imagenet', input_tensor=inputs, classes=1)
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    # invoke modifier if given
    if modifier:
        xceptionNet = modifier(xceptionNet)
    x = xceptionNet.output
    print(x.shape)
    xceptionNet = GlobalAveragePooling2D(name='avg_pool')(x)

    outputs = Dense(1, activation='sigmoid', name='predictions')(xceptionNet)
    model = Model(inputs=inputs, outputs=outputs, name='xception')

    # create the full model
    return model
