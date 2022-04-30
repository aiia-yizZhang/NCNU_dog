import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint

train_x=np.load('npys/train_x.npy')
train_y=np.load('npys/train_y.npy')
valid_x=np.load('npys/valid_x.npy')
valid_y=np.load('npys/valid_y.npy')
train_x_neg=np.load('npys/train_x_neg.npy')
valid_x_neg=np.load('npys/valid_x_neg.npy')

train_y=to_categorical(train_y, num_classes=10)
valid_y=to_categorical(valid_y, num_classes=10)
train_y_neg=np.zeros((len(train_x_neg), 10), dtype='float32')
valid_y_neg=np.zeros((len(valid_x_neg), 10), dtype='float32')

print(train_x.shape, train_x.dtype, train_y.shape, train_y.dtype)
print(valid_x.shape, valid_x.dtype, valid_y.shape, valid_y.dtype)
print(train_x_neg.shape, train_x_neg.dtype, train_y_neg.shape, train_y_neg.dtype)
print(valid_x_neg.shape, valid_x_neg.dtype, valid_y_neg.shape, valid_y_neg.dtype)

train_x=np.concatenate([train_x, train_x_neg], axis=0)
train_y=np.concatenate([train_y, train_y_neg], axis=0)
valid_x=np.concatenate([valid_x, valid_x_neg], axis=0)
valid_y=np.concatenate([valid_y, valid_y_neg], axis=0)
np.random.seed(20220425)
np.random.shuffle(train_x)
np.random.seed(20220425)
np.random.shuffle(train_y)
print(train_x.shape, train_x.dtype, train_y.shape, train_y.dtype)
print(valid_x.shape, valid_x.dtype, valid_y.shape, valid_y.dtype)


idg=ImageDataGenerator(
    brightness_range=[0.6,1.4],
    fill_mode='constant',
    cval=0.0,
    horizontal_flip=True
)

def generator(set_x, set_y, batch_size=32):
    g=idg.flow(set_x, set_y, batch_size=batch_size)
    for a,b in g:
        a=a.astype('float32')/255.0
        yield a,b


model=ResNet50(
    weights=None, input_shape=(224,224,3),
    classes=10, classifier_activation='sigmoid'
)
model.compile(
    optimizer='adam', loss='binary_crossentropy',
    metrics=['acc']
)

mcp=ModelCheckpoint(
    'model.h5', monitor='val_loss',
    verbose=1, save_best_only=True, save_weights_only=True, mode='min'
)

model.fit(
    generator(train_x, train_y),
    steps_per_epoch=35,
    validation_data=generator(valid_x, valid_y),
    validation_steps=35,
    epochs=100,
    callbacks=[mcp]
)


model.load_weights('model.h5')


'''
g=generator(train_x, train_y)
for ind,(a,b) in enumerate(g):
    a=(a*255.0).astype('uint8')
    b=np.sum(b, axis=1)
    cv2.imwrite('test.jpg', a[0])
    print(b[0])
    break
'''


probability_constraint=0.5

pred=model.predict(valid_x.astype('float32')/255.0)
pred=np.max(pred, axis=1)
pred=np.where(pred>probability_constraint,1,0)
true=np.where(np.sum(valid_y, axis=1)!=0,1,0)

acc=accuracy_score(true, pred)
precision=precision_score(true, pred, average='macro')
recall=recall_score(true, pred, average='macro')
print(probability_constraint, acc, precision, recall)
cf_matrix = confusion_matrix(true, pred)
cf_matrix=(cf_matrix/np.array([np.sum(cf_matrix, axis=1)]).T)
sns.set(rc={'figure.figsize':(7, 5)})
sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='.1%')
plt.show()






