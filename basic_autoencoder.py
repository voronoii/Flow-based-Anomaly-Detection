from load_data_pretrain import load_data_3, prepare_testset_with_label2
import numpy as np
from keras.callbacks import History
from keras.models import Model
from keras.layers import Input,  Dense, concatenate
from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
import matplotlib.pyplot as plt



# configure problem
n_features = 4
n_timesteps_in = 5
n_timesteps_out = 2
input_seq_len = 5

embedding_dim = 64
epoch = 10


# load data

data = load_data_3()
data = np.array(data[:1048500])




train_data = data[:100000]
test_data = data[100000:len(data)]

# define model


data_inputs = Input(shape=(6,))


encoder = Dense(int(embedding_dim/2))(data_inputs)
encoder = Dense(embedding_dim, activation="tanh",
                activity_regularizer=regularizers.l1(10e-5))(encoder)

hidden_dense = Dense(int(embedding_dim/2), activation='relu')(encoder)

output_dense = Dense(6, activation='tanh')(hidden_dense)

model = Model(inputs=data_inputs, outputs=output_dense)


# evaluation

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.summary()

tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)


batch = 100
step_per_epoch = int(len(train_data)/batch)  # 1048500/500

def packet_generator(step_per_epoch, batch):  # 500

    block = batch

    while True:

        # total / batchsize = 10485/500
        for i in range(step_per_epoch):
            x= train_data[(i) * block: (i + 1) * block].reshape(batch, 6)

            yield (x, x)



generator = packet_generator(step_per_epoch, batch)

print("training...........")

history = model.fit_generator(generator, steps_per_epoch=step_per_epoch, epochs=30, validation_data=(test_data, test_data),  callbacks=[tensorboard])
# history = model.fit(x=[train_cate, train_attr], y=train_target,
#                     epochs=10,
#                     batch_size=64,
#                     shuffle=True,
#                     validation_data=([test_cate, test_attr], test_target),
#                     verbose=1,
#                     callbacks=[tensorboard]).history



score, acc = model.evaluate(test_data, test_data)
print('Test score:', score)
print('Test accuracy:', acc)

# serialize model to JSON
model_json = model.to_json()
with open("basic_autoencoder2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("basic_autoencoder2.h5")
print("Saved model to disk")


print("********************************************************************************************************")
print("**************************************** Testing ************************************************")
print("********************************************************************************************************")

# threshold = 0.155


data, label = prepare_testset_with_label2()

data_ = data[:988120]
data_ = data_.reshape((988120, 6))
label = label[:988120]
print("data shape : ", data_.shape)  # 1048500
print("label shape : ", len(label))


# ip = ip.reshape((206600, 5, 4))
# attr = attr.reshape((206600, 5, 4))
# label = (np.array(label)).reshape((206600, 5, 1))
# decoder_input = np.zeros((ip.shape[0], 100, 8))


"""Balanced class: In this situation, the F1 score can effectively be ignored, the mis-classification rate is key."""
import pandas as pd

predictions = model.predict(data_)

mse = np.mean(np.power(data_ - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': label})
error_df.describe()




fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')


plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
plt.title('Precision for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.show()


plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
plt.title('Recall for different threshold values')
plt.xlabel('Reconstruction error')
plt.ylabel('Recall')
plt.show()
