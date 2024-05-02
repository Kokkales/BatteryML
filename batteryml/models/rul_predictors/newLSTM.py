# import tensorflow as tf
# from batteryml.builders import MODELS
# from batteryml.models.nn_model import NNModel

# @MODELS.register()
# class MyLSTMRULPredictor(tf.keras.Model):
#     def __init__(self,
#                  in_channels,
#                  channels,
#                  input_height,
#                  input_width,
#                  **kwargs):
#         # super(MyLSTMRULPredictor, self).__init__(**kwargs)
#         NNModel.__init__(self, **kwargs)
#         self.lstm = tf.keras.layers.LSTM(
#             units=channels,
#             return_sequences=True,
#             return_state=True)
#         self.fc = tf.keras.layers.Dense(units=1)

#     def forward(self,
#              feature,
#              label=None,
#              return_loss=False):
#         if len(feature.shape) == 3:
#             feature = tf.expand_dims(feature, axis=1)
#         B, _, H, _ = feature.shape
#         x = tf.transpose(feature, perm=[0, 2, 1, 3])
#         x = tf.reshape(x, [B, H, -1])
#         x, _, _ = self.lstm(x)
#         x = x[:, -1]
#         x = self.fc(x)
#         x = tf.squeeze(x, axis=-1)

#         if return_loss:
#             return tf.reduce_mean(tf.square(x - tf.reshape(label, [-1])))

#         return x
