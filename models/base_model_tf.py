import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0


class BaseModel(Model):
    def __init__(self):
        super().__init__()
        self.feat_extractor = EfficientNetB0(weights='imagenet',input_shape=(640, 480, 3),  include_top=False,drop_connect_rate=0.2)

    
if __name__== "__main__":
    model = BaseModel()
    print(model.feat_extractor.summary())