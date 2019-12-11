import keras
from keras.applications.resnet50 import ResNet50

from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.models import Model

# from keras.applications.resnet101 import ResNet101
# from keras.applications.resnet152 import ResNet152
# import keras_resnet
# import keras_resnet.models




def resnet(backbone='resnet50', inputs=None, modifier=None, **kwargs):
    """ Constructs a m2det model using a resnet backbone.
    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('resnet50', 'resnet101', 'resnet152')).
        inputs: The inputs to the network (defaults to a Tensor of shape (640, 640, 3)).
        modifier: A function handler which can modify the backbone before using it in m2det (this can be used to freeze backbone layers for example).
    Returns
        m2det model with a ResNet backbone.
    """
    # choose default input
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, 256, 256))
        else:
            inputs = keras.layers.Input(shape=(256, 256, 3))

    # create the resnet backbone
    if backbone == 'resnet50':
        resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs, classes=1)
        print(resnet.output)
        # resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
    # elif backbone == 'resnet101':
        # resnet = ResNet101(include_top=False, weights='imagenet', input_tensor=inputs, classes=1)
        # resnet = keras_resnet.models.ResNet101(inputs, include_top=False, freeze_bn=True)
    # elif backbone == 'resnet152':
        # resnet = ResNet152(include_top=False, weights='imagenet', input_tensor=inputs, classes=1)
        # resnet = keras_resnet.models.ResNet152(inputs, include_top=False, freeze_bn=True)
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)
    x = resnet.output
    print(x.shape)
    resnet = AveragePooling2D(pool_size=8)(x)
    resnet = Flatten()(resnet)

    outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(resnet)
    model = Model(inputs=inputs, outputs=outputs)

    # create the full model
    return model
