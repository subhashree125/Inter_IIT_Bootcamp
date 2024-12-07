from keras.models import Model
from keras.layers import Reshape, Permute, Activation
from types import MethodType
# from .train import train
# from .predict import predict, predict_multiple, evaluate

def get_segmentation_model(input, output):

    img_input = input
    o = output

    # Get the input and output shapes from the model
    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    # For 'channels_last' (default in TensorFlow)
    output_height = o_shape[1]
    output_width = o_shape[2]
    input_height = i_shape[1]
    input_width = i_shape[2]
    n_classes = o_shape[3]

    # Reshape and apply softmax activation
    o = Reshape((output_height * output_width, -1))(o)
    o = Activation('softmax')(o)

    # Define the model
    model = Model(img_input, o)

    # Set attributes to store model dimensions and number of classes
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.model_name = ""

    # Attach custom methods for training, prediction, and evaluation
    # #model.train = MethodType(train, model)
    # model.predict_segmentation = MethodType(predict, model)
    # model.predict_multiple = MethodType(predict_multiple, model)
    # model.evaluate_segmentation = MethodType(evaluate, model)

    return model
