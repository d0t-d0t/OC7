from keras import ops
from keras.losses import binary_crossentropy

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Convert y_true to float32 to match y_pred's type
    y_true = ops.cast(y_true, dtype='float32')
    y_pred = ops.cast(y_pred, dtype='float32')
    
    # Flatten the tensors using keras.ops
    y_true_f = ops.reshape(y_true, (-1,))
    y_pred_f = ops.reshape(y_pred, (-1,))
    
    # Calculate intersection and score using keras.ops
    intersection = ops.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (ops.sum(y_true_f) + ops.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def custom_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))
    return loss
