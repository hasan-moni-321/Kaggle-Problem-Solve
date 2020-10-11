##########################################################
# Loading necessary library
##########################################################

import pickle
import joblib
import matplotlib.pyplot as plt


from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.layers import Input, GlobalAveragePooling2D, concatenate, AveragePooling2D

from keras.optimizers import SGD, RMSprop
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3




########################################################
# Path of the train and test dataset
########################################################

train_path='/home/hasan/DATA SET/Indoor CVPR/train_image'
test_path='/home/hasan/DATA SET/Indoor CVPR/test_image'


##########################################################
# ImageDataGenerator
##########################################################

train_valid_datagen = ImageDataGenerator(
                        rotation_range=30,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        validation_split=0.3,
                        rescale=1./255
                        )


#test data
test_datagen = ImageDataGenerator(
                        rotation_range=30,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        featurewise_center=True
                        )



# train data
train_generator = train_valid_datagen.flow_from_directory(
                                          train_path,
                                          target_size=(299, 299),
                                          batch_size=32,
                                          class_mode='categorical', 
                                          subset='training'
                                          )


# vaild data
vaild_generator = train_valid_datagen.flow_from_directory(
                                          train_path,
                                          target_size=(299, 299),
                                          batch_size=32,
                                          class_mode='categorical', 
                                          subset='validation'
                                          )


# test data
test_generator = test_datagen.flow_from_directory(
                                          test_path,
                                          target_size=(299, 299),
                                          batch_size=32
)


########################################################
# Using InceptionV3 Model
########################################################

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
predictions = Dense(67, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False


# Summary of the model
model.summary()

# Compile of the Model
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


###################################################################
#Training the model    
####################################################################

model_history = model.fit_generator(train_generator,
                    validation_data=vaild_generator,
                    epochs=200,
                    steps_per_epoch=train_generator.n/32,
                    validation_steps=vaild_generator.n/32,
                    shuffle=True,
                    verbose=1)


###############################################################
# Accuracy and Loss curve
###############################################################
accuracy = model_history.history['accuracy']
val_accuracy = model_history.history['val_accuracy']

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'go', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'g', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'go', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.show()


###############################################################
# Predicting with test data
###############################################################
loss,acc=model.evaluate_generator(test_generator,
                                  steps=vaild_generator.n/32)
#print('Test result:loss:%f,acc:%f'%(loss,acc))
print("Accuracy of the test data is :", acc)
print("Loss of the test data is :", loss)


###############################################################
# Save Model
##############################################################
pickle_out = open("Indoor_CVPR_Model1.p", 'wb')
pickle.dump(model, pickle_out)
pickle_out.close()


##############################################################
# Save train history
##############################################################
history=model.history
joblib.dump((history), "Indoor_CVPR_Train_History1.pkl", compress=3)


###############################################################
# Plot Model
###############################################################

plot_model(model)
plot_model(model, to_file='Indoor_CVPR_Plot_Model1.png')


############################################################
# Choose necessary layers
############################################################

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:17+1]:
   layer.trainable = False
for layer in model.layers[17+1:]:
   layer.trainable = True


# Compile the model
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model_history = model.fit_generator(train_generator,
                    validation_data=vaild_generator,
                    epochs=200,
                    steps_per_epoch=train_generator.n/32,
                    validation_steps=vaild_generator.n/32,
                    shuffle=True,
                    verbose=1)


#######################################################
# Accuracy and Loss graph
######################################################
accuracy = model_history.history['accuracy']
val_accuracy = model_history.history['val_accuracy']

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'go', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'g', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'go', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.show()   


###############################################
# Model Evaluate with test data
############################################
loss,acc=model.evaluate_generator(test_generator,
                                  steps=vaild_generator.n/32)
#print('Test result:loss:%f,acc:%f'%(loss,acc))
print("Accuracy of the test data is :", acc)
print("Loss of the test data is :", loss)


################################################
# Save Model
#################################################
pickle_out = open("Indoor_CVPR_Model2.p", 'wb')
pickle.dump(model, pickle_out)
pickle_out.close()


##########################################################
# Save Train History
###########################################################
history=model.history
joblib.dump((history), "Indoor_CVPR_Train_History2.pkl", compress=3)


###############################################################
# Plot Model
##############################################################
# let's visualize layer names and layer indices to see how many layers we should freeze:
plot_model(model)
plot_model(model, to_file='Indoor_CVPR_Plot_Model2.png')