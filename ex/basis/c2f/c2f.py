import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float) # features arr
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float) # labels arr

# # log out expected mappings for example
# for i,c in enumerate(celsius_q):
#   print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

# init single layer model
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

# compile model with standard loss and optimizer fns
model.compile(loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(0.1))

# train model
history = model.fit(celsius_q, fahrenheit_a, epochs=1000, verbose=False)
print("------------\n")
print("Finished training model. Final weights: {}".format(l0.get_weights()))

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

while True:
    celsInput = input("Input celsius value for model to predict: ")
    print("Prediction for " + celsInput + "°C in °F:")
    print(model.predict(x=np.array([int(celsInput)]))[0][0])