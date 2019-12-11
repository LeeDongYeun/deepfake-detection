def create_model(model_name):
    if 'xception' in model_name:
        from .xception import xception
        model = xception(model_name)
    if 'resnet' in model_name:
        # from .resnet2 import resnet_v2
        # model = resnet_v2((256, 256, 3), 110, num_classes=1)
        from .resnet import resnet
        model = resnet(model_name)
    if 'vgg' in model_name:
        from .vgg import vgg
        model = vgg(model_name)
    return model


def load_model(filepath, backbone_name='resnet50'):
    """ Loads a retinanet model using the correct custom objects.
    Args
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        backbone_name         : Backbone with which the model was trained.
    Returns
        A keras.models.Model object.
    Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    import keras.models
    print("Loading the model from ", filepath, "...")
    return keras.models.load_model(filepath)
