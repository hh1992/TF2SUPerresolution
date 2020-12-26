import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.layers import Conv2D, ReLU, Concatenate, GlobalAveragePooling2D, Multiply, Dense, Reshape

def SRB(X,filter):
    X1 = ReLU()(Conv2D(filter, kernel_size=(3, 3), padding='same')(X))

    return X + X1
def ca(input_tensor, filters, reduce=16):
    x = GlobalAveragePooling2D()(input_tensor)
    x = Reshape((1, 1, filters))(x)
    x = Dense(filters/reduce,  activation='relu', kernel_initializer='he_normal', use_bias=False)(x)
    x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)
    x = Multiply()([x, input_tensor])
    return x


def RFDB(X):
     filter_left = int(list(X.shape)[-1] / 2)
     filter_right = int(list(X.shape)[-1])
     left_1 = Conv2D(filter_left, kernel_size=(1, 1))(X)
     right_1 = SRB(X, filter_right)

     left_2 = Conv2D(filter_left, kernel_size=(1, 1))(right_1)
     right_2 = SRB(right_1, filter_right)

     left_3 = Conv2D(filter_left, kernel_size=(1, 1))(right_2)
     right_3 = SRB(right_2, filter_right)

     right_final = Conv2D(filter_left, kernel_size=(3, 3), padding='same')(right_3)

     concat = Concatenate(axis=-1)([left_1, left_2, left_3, right_final])

     concate_1 = Conv2D(filter_right, kernel_size=(1, 1))(concat)
     concate_1 = ca(concate_1, filter_right)

     return concate_1 + X




def rfdn(scale, num_filters=64):
    x_in = Input(shape=(None, None, 3))
    X1 = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(x_in)

    out_B1 = RFDB(X1)
    out_B2 = RFDB(out_B1)
    out_B3 = RFDB(out_B2)
    out_B4 = RFDB(out_B3)

    concat = Concatenate(axis=-1)([out_B1, out_B3, out_B3, out_B4])

    concat_1 = Conv2D(num_filters, kernel_size=(1, 1), activation='relu')(concat)

    LR = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(concat_1) + X1

    X_up = Conv2D(num_filters * (scale ** 2), 3, padding='same')(LR)
    out = tf.nn.depth_to_space(X_up, scale)
    out = Conv2D(3, kernel_size=(1, 1))(out)
    return Model(x_in, out, name="rfdn")










