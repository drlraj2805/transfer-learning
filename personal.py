#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D


# In[2]:


from keras.layers import MaxPooling2D


# In[3]:


from keras.layers import Flatten


# In[4]:


from keras.layers import Dense


# In[5]:


from keras.models import Sequential


# In[6]:


model = Sequential()


# In[7]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))


# In[8]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[9]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))


# In[10]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[11]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))


# In[12]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[13]:


model.add(Flatten())


# In[20]:


model.summary()


# In[16]:


model.add(Dense(units=138, activation='relu'))


# In[17]:


model.add(Dense(units=110, activation='relu'))


# In[18]:


model.add(Dense(units=1, activation='sigmoid'))


# In[19]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[21]:


from keras_preprocessing.image import ImageDataGenerator


# In[22]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'anmol/train/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'anmol/validation/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=70,
        epochs=15,
        validation_data=test_set,
        validation_steps=10)


# In[ ]:


model.save("myface2.h5")


# In[ ]:


## 64_Face Structure


# In[ ]:


from keras.applications import VGG16

# Setting the input size now to 64 x 64 pixel 
img_rows = 64
img_cols = 64 

# Re-loads the VGG16 model without the top or FC layers
vgg16 = VGG16(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))

# Here we freeze the last 4 layers 
# Layers are set to trainable as True by default
for layer in vgg16.layers:
    layer.trainable = False
    
# Let's print our layers 
for (i,layer) in enumerate(vgg16.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# In[ ]:


from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'divyaraj/train/'
validation_data_dir = 'divyaraj/validation/'

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# Change the batchsize according to your system RAM
train_batchsize = 16
val_batchsize = 10
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=train_batchsize,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Re-loads the VGG16 model without the top or FC layers
vgg16 = VGG16(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))

# Freeze layers
for layer in vgg16.layers:
    layer.trainable = False
    
# Number of classes in the Flowers-17 dataset
num_classes = 17

FC_Head = addTopModel(vgg16, num_classes)

model = Model(inputs=vgg16.input, outputs=FC_Head)

print(model.summary())


# In[ ]:



