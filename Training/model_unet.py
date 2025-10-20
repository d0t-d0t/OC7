
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout, Reshape, Permute
# from tensorflow.python.keras.optimizers import Adadelta, Nadam
from keras.models import Model
from keras.applications import VGG16
# from tensorflow.python.keras.utils import multi_gpu_model, plot_model



def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y


def Unet(img_height, img_width, nclasses=3, filters=64):
# down
    input_layer = Input(shape=(img_height, img_width, 3), 
                        name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters*4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters*8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*16)
    conv5 = Dropout(0.5)(conv5)
# up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)
    deconv7 = Dropout(0.5)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
# output
    output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Reshape((img_height*img_width, nclasses), 
                            input_shape=(img_height, img_width, nclasses)
                            )(output_layer)
    output_layer = Activation('softmax')(output_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model    

def VGG16_Unet(img_height, img_width, nclasses=3):
    # Input
    input_layer = Input(shape=(img_height, img_width, 3), name='image_input')
    
    # Load pre-trained VGG16 as encoder
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)
    
    # Make encoder layers non-trainable
    for layer in vgg16.layers:
        layer.trainable = False
    
    # Get skip connections from VGG16
    # VGG16 has 5 blocks with the following output names:
    skip1 = vgg16.get_layer('block1_conv2').output  # 64 filters
    skip2 = vgg16.get_layer('block2_conv2').output  # 128 filters  
    skip3 = vgg16.get_layer('block3_conv3').output  # 256 filters
    skip4 = vgg16.get_layer('block4_conv3').output  # 512 filters
    
    # Bottom of U-Net (last VGG16 block)
    bottom = vgg16.get_layer('block5_conv3').output  # 512 filters
    bottom = Dropout(0.5)(bottom)
    
    # Acending part - Trainable
    # Up sampling block 1
    deconv6 = deconv_block(bottom, residual=skip4, nfilters=512)
    deconv6 = Dropout(0.5)(deconv6)
    
    # Up sampling block 2
    deconv7 = deconv_block(deconv6, residual=skip3, nfilters=256)
    deconv7 = Dropout(0.5)(deconv7)
    
    # Up sampling block 3
    deconv8 = deconv_block(deconv7, residual=skip2, nfilters=128)
    
    # Up sampling block 4
    deconv9 = deconv_block(deconv8, residual=skip1, nfilters=64)
    
    # Output layer
    output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Reshape((img_height*img_width, nclasses), input_shape=(img_height, img_width, nclasses))(output_layer)
    output_layer = Activation('softmax')(output_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer, name='VGG16_Unet')
    return model