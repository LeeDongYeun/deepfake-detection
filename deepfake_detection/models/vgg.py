import keras
from keras.applications import VGG16, VGG19


def vgg(backbone='vgg16', inputs=None, modifier=None, **kwargs):
    """ Constructs a m2det model using a vgg backbone.
    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
        inputs: The inputs to the network (defaults to a Tensor of shape (320, 320, 3)).
        modifier: A function handler which can modify the backbone before using it in m2det (this can be used to freeze backbone layers for example).
    Returns
        RetinaNet model with a VGG backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # create the vgg backbone
    if backbone == 'vgg16':
        vgg = VGG16(input_tensor=inputs, include_top=False, weights=None)
    elif backbone == 'vgg19':
        vgg = VGG19(input_tensor=inputs, include_top=False, weights=None)
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    if modifier:
        vgg = modifier(vgg)

    # create the full model

    return vgg