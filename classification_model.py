from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, BatchNormalization, UpSampling2D, concatenate
from models_utils import get_segmentation_model
from vgg16 import get_vgg_encoder


def _unet(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608, channels=3):

    # Get the input image and levels from the encoder
    img_input, levels = encoder(
        input_height=input_height, input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    # Decoder begins here
    o = f4

    # Block 1
    o = ZeroPadding2D((1, 1))(o)
    o = Conv2D(512, (3, 3), padding='valid', activation='relu')(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([o, f3], axis=-1)  # axis=-1 is for "channels_last"
    o = ZeroPadding2D((1, 1))(o)
    o = Conv2D(256, (3, 3), padding='valid', activation='relu')(o)
    o = BatchNormalization()(o)

    # Block 2
    o = UpSampling2D((2, 2))(o)
    o = concatenate([o, f2], axis=-1)
    o = ZeroPadding2D((1, 1))(o)
    o = Conv2D(128, (3, 3), padding='valid', activation='relu')(o)
    o = BatchNormalization()(o)

    # Block 3
    o = UpSampling2D((2, 2))(o)

    if l1_skip_conn:
        o = concatenate([o, f1], axis=-1)

    o = ZeroPadding2D((1, 1))(o)
    o = Conv2D(64, (3, 3), padding='valid', activation='relu', name="seg_feats")(o)
    o = BatchNormalization()(o)

    # Final segmentation output
    o = Conv2D(n_classes, (3, 3), padding='same')(o)

    # Create the model
    model = get_segmentation_model(img_input, o)

    return model

def vgg_unet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3):

    model = _unet(n_classes, get_vgg_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "vgg_unet"
    return model
