# Import libraries

import numpy as np
import tensorflow as tf
import cv2 as cv
import random
import matplotlib.pyplot as plt

# Generate training images and labels
N=4000
H,W=256,256

train_img=np.zeros([N,H,W,3],dtype=np.uint8)
train_img.fill(255)

train_label_coord=np.zeros([N,2],dtype=np.int32)
train_label_size=np.zeros([N,2],dtype=np.int32)
train_label_class=np.zeros([N,1],dtype=np.int32)

for n in range(N):
    x,y=random.randint(0,W-1),random.randint(0,H-1)
    bw,bh=random.randint(int(W/16),int(W/4)),random.randint(int(H/16),int(H/4))
    
    if(x-bw/2<0): x=x-(x-bw/2)
    elif(x+bw/2>W-1): x=x-(x+bw/2-(W-1))
    if(y-bh/2<0): y=y-(y-bh/2)
    elif(y+bh/2>H-1): y=y-(y+bh/2-(H-1))
    
    x=int(x); y=int(y)
    
    train_label_coord[n,0]=x; train_label_coord[n,1]=y
    train_label_size[n,0]=bw; train_label_size[n,1]=bh
    train_label_class[n]=random.randint(0,3) # 0: red, 1: green, 2: blue, 3: yellow
    
    if train_label_class[n]==0: # red
        cv.rectangle(train_img[n],(x-int(bw/2),y-int(bh/2)),(x+int(bw/2),y+int(bh/2)), color=(0,0,255), thickness=-1)
    elif train_label_class[n]==1: # green
        cv.rectangle(train_img[n],(x-int(bw/2),y-int(bh/2)),(x+int(bw/2),y+int(bh/2)), color=(0,255,0), thickness=-1)
    elif train_label_class[n]==2: # blue
        cv.rectangle(train_img[n],(x-int(bw/2),y-int(bh/2)),(x+int(bw/2),y+int(bh/2)), color=(255,0,0), thickness=-1)
    else: # yellow
        cv.rectangle(train_img[n],(x-int(bw/2),y-int(bh/2)),(x+int(bw/2),y+int(bh/2)), color=(255,255,0), thickness=-1)

# Display some images

fig = plt.figure(figsize=(12,12))
rows=4
columns=5

for idx in range(1, columns * rows +1):
    fig.add_subplot(rows, columns, idx)
    plt.imshow(train_img[idx])

# Preprocess data
train_img = tf.keras.applications.vgg16.preprocess_input(train_img)
train_label_coord=train_label_coord.astype(np.float32)
train_label_coord[:,0]=train_label_coord[:,0]/W
train_label_coord[:,1]=train_label_coord[:,1]/H
train_label_size=train_label_size.astype(np.float32)
train_label_size[:,0]=train_label_size[:,0]/(W/4)
train_label_size[:,1]=train_label_size[:,1]/(H/4)

# Build model
base_model = tf.keras.applications.VGG16(input_shape=[H,W,3], include_top=False, weights='imagenet')
x = base_model.output
x = tf.keras.layers.Flatten()(x)

out_coord=tf.keras.layers.Dense(2,activation='sigmoid',name='out_coord')(x)
out_size=tf.keras.layers.Dense(2, activation='sigmoid',name='out_size')(x)
out_class=tf.keras.layers.Dense(4,activation='softmax',name='out_class')(x) # 클래스가 4개

model=tf.keras.Model(inputs=base_model.input,outputs=[out_coord,out_size,out_class])
model.summary()

# Custom loss functions
def coord_loss_func(y_true,y_pred):
    loss_coord=tf.keras.losses.MSE(y_true,y_pred)
    return loss_coord

def size_loss_func(y_true,y_pred):
    loss_size=tf.keras.losses.MSE(y_true,y_pred)
    return loss_size
    
def class_loss_func(y_true,y_pred):  
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, 4)
    y_true = tf.reshape(y_true, [-1,4])

    # One Hot Encoding을 한 경우 e.g. [0,0,1,0] : categorical_crossentropy, 그렇지 않은 경우 (e.g. [3,0,1] : spare_categorical_crossentropy 
    loss_class=tf.keras.losses.categorical_crossentropy(y_true,y_pred)
    return loss_class

# Train and save model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss={'out_coord':coord_loss_func,'out_size':size_loss_func,'out_class':class_loss_func},
    loss_weights={'out_coord':3,'out_size':2,'out_class':2})

history = model.fit(x=train_img,
    y={'out_coord':train_label_coord,'out_size':train_label_size,'out_class':train_label_class},
    epochs=20,batch_size=10,validation_split=0.25)
model.save('model.keras')

# Show training history
plt.figure()
plt.plot(history.history['loss'], 'b-', label='training')
plt.plot(history.history['val_loss'], 'r--', label='validation')
plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend()
plt.show()

# Generate test images and labels
N=20
H,W=256,256

test_img=np.zeros([N,H,W,3],dtype=np.uint8)
test_img.fill(255)
test_label_coord=np.zeros([N,2],dtype=np.int32)
test_label_size=np.zeros([N,2],dtype=np.int32)
test_label_class=np.zeros([N,1],dtype=np.int32) # 출력 라벨 클래스가 4개

for n in range(N):
    x,y=random.randint(0,W-1),random.randint(0,H-1)
    bw,bh=random.randint(int(W/16),int(W/4)),random.randint(int(H/16),int(H/4))
    
    if(x-bw/2<0): x=x-(x-bw/2)
    elif(x+bw/2>W-1): x=x-(x+bw/2-(W-1))
    if(y-bh/2<0): y=y-(y-bh/2)
    elif(y+bh/2>H-1): y=y-(y+bh/2-(H-1))
    
    x=int(x); y=int(y)
    
    test_label_coord[n,0]=x; test_label_coord[n,1]=y
    test_label_size[n,0]=bw; test_label_size[n,1]=bh
    test_label_class[n]=random.randint(0,3) # 0: red, 1: green, 2: blue, 3: yellow
    
    if test_label_class[n]==0: # red
        cv.rectangle(test_img[n],(x-int(bw/2),y-int(bh/2)),(x+int(bw/2),y+int(bh/2)), color=(0,0,255), thickness=-1)
    elif test_label_class[n]==1: # green
        cv.rectangle(test_img[n],(x-int(bw/2),y-int(bh/2)),(x+int(bw/2),y+int(bh/2)), color=(0,255,0), thickness=-1)
    elif test_label_class[n]==2: # blue
        cv.rectangle(test_img[n],(x-int(bw/2),y-int(bh/2)),(x+int(bw/2),y+int(bh/2)), color=(255,0,0), thickness=-1)
    else: # yellow
        cv.rectangle(test_img[n],(x-int(bw/2),y-int(bh/2)),(x+int(bw/2),y+int(bh/2)), color=(255,255,0), thickness=-1)

# Preprocess test images
test_img_ = tf.keras.applications.vgg16.preprocess_input(test_img)
# Predict object locations in test images
model=tf.keras.models.load_model('model.keras', custom_objects={'coord_loss_func':coord_loss_func,
                                                          'size_loss_func':size_loss_func, 'class_loss_func':class_loss_func})

pred_coord,pred_size,pred_class=model.predict(test_img_)
pred_coord[:,0]=pred_coord[:,0]*H
pred_coord[:,1]=pred_coord[:,1]*W
pred_size[:,0]=pred_size[:,0]*(H/4)
pred_size[:,1]=pred_size[:,1]*(W/4)
pred_class=np.argmax(pred_class,axis=1)

# Display prediction results

fig = plt.figure(figsize=(12, 12))
rows = 4
columns=5

for n in range(N):
    x=pred_coord[n,0].astype('int')
    y=pred_coord[n,1].astype('int')
    bw=pred_size[n,0].astype('int')
    bh=pred_size[n,1].astype('int')
    
    obj_class=pred_class[n]
    
    if obj_class==0: # red
        cv.rectangle(test_img[n],(x-int(bw/2),y-int(bh/2)),(x+int(bw/2),y+int(bh/2)),color=(0,0,0),thickness=1)
        cv.rectangle(test_img[n],(0,0),(W-1,H-1),color=(0,0,255),thickness=4)
    elif obj_class==1: # green
        cv.rectangle(test_img[n],(x-int(bw/2),y-int(bh/2)),(x+int(bw/2),y+int(bh/2)), color=(0,0,0),thickness=1)
        cv.rectangle(test_img[n],(0,0),(W-1,H-1),color=(0,255,0),thickness=4)
    elif obj_class==2: # blue
        cv.rectangle(test_img[n],(x-int(bw/2),y-int(bh/2)),(x+int(bw/2),y+int(bh/2)),color=(0,0,0),thickness=1)
        cv.rectangle(test_img[n],(0,0),(W-1,H-1),color=(255,0,0),thickness=4)
    else : # yellow
        cv.rectangle(test_img[n],(x-int(bw/2),y-int(bh/2)),(x+int(bw/2),y+int(bh/2)),color=(0,0,0),thickness=1)
        cv.rectangle(test_img[n],(0,0),(W-1,H-1),color=(255,255,0),thickness=4)
                                 
    cv.circle(test_img[n],center=(x,y),radius=2,color=(0,0,0),thickness=2)
    fig.add_subplot(rows, columns, n+1)
    plt.imshow(test_img[n])