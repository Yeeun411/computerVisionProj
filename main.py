import scipy.io as io
import numpy as np

# load the data
data = io.loadmat("/Users/kang-yeeun/Desktop/face_landmark.mat")
X = data["images"]
Y = data["landmarks"]
X, Y = np.float32(X), np.float32(Y)

print("im_shape:", X.shape)
print("landmarks_shape:", Y.shape)

# reshape the images X into 3D (H,W,C)
X = np.reshape(X, [-1, 96, 96, 1])


from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# visualize some example
id = np.random.randint(len(X))
im = X[id] # 96x96
im = np.reshape(im, [96,96])
plt.imshow(im, cmap="gray")
keypoints = Y[id].reshape(-1, 2)
for point in keypoints:
    plt.plot(point[0], point[1], 'r+')
plt.title("class:" + str(Y[id]))
plt.show()

Y = np.reshape(Y, [-1, 30])
# randomly split into train (0.6), val(0.2), test(0.2)
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=101)

import tensorflow as tf
regularizer = tf.keras.regularizers.L2(0.001)

x_in = x = tf.keras.Input(shape=[96, 96, 1])
x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units=64, activation=tf.nn.softmax, kernel_regularizer=regularizer)(x)
x = tf.keras.layers.Dropout(rate=0.5)(x)
y = x = tf.keras.layers.Dense(units=30, activation=None, kernel_regularizer=regularizer)(x)
model = tf.keras.Model(inputs=x_in, outputs=y)
model.summary()

@tf.function
def predict(x):
    return model(x)

@tf.function
def loss_fn(y, y_true):
    y_true = tf.reshape(y_true, shape=(-1, 15, 2))
    y_pred = tf.reshape(y, shape=(-1, 15, 2))
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def optimize(x,y_true):
    with tf.GradientTape() as tape:
        y = predict(x)
        loss = loss_fn(y, y_true)
        loss += sum(model.losses)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss

max_epochs = 100
batch_size = 32
loss_history = []
val_loss_best = np.inf
for epoch in range(max_epochs):
    idx = np.random.permutation(len(X_train))
    batch_idx = idx[:batch_size]
    train_loss = optimize(X_train[batch_idx], Y_train[batch_idx])
    train_loss = train_loss.numpy()
    y = predict(X_val)
    val_loss = loss_fn(y, Y_val).numpy()
    if val_loss < val_loss_best:
        val_loss_best = val_loss
    print(epoch, '.', train_loss, val_loss, val_loss_best)
    loss_history.append([train_loss, val_loss])

p = model.predict(X_test)
mse = np.mean(np.square(Y_test - p))
print("Mean Square Error:", mse)