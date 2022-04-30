from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np, tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
from matplotlib import pyplot as plt
from triplet_loss import batch_hard_triplet_loss

train_x=np.load('npys/train_x.npy')
train_y=np.load('npys/train_y.npy')
train_x_neg=np.load('npys/train_x_neg.npy')
s=np.max(train_y)+1
train_y_neg=np.arange(s,s+len(train_x_neg))

valid_x=np.load('npys/valid_x.npy')
valid_y=np.ones((len(valid_x),), dtype='int32')
valid_x_neg=np.load('npys/valid_x_neg.npy')
valid_y_neg=np.zeros((len(valid_x_neg),), dtype='int32')

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

def resmodel():
    base_model=ResNet50(
        input_shape=(None,None,3),
        include_top=False, weights='imagenet',
    )
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(128, activation='linear')(x)
    output=Lambda(lambda  x: K.l2_normalize(x,axis=1))(output)
    model = Model(
        inputs=[base_model.input],
        outputs=[output]
    )
    return model

model=resmodel()

def augment(x, y):
    x=x.astype('float32')
    y=y.astype('float32')
    for y_ in np.unique(y):
        pick=np.where(y==y_)
        idg=ImageDataGenerator(
            brightness_range=[0.6,1.4],
            fill_mode='constant',
            cval=0.0,
            horizontal_flip=True
        )
        new_x,_=next( idg.flow(x[pick], y[pick], batch_size=len(pick)) )
        x[pick]=new_x
    return x,y

def generator(type_per_batch=2, images_per_type=5):
    x=train_x.copy()
    y=train_y.copy()
    y_unique=np.unique(y)
    while 1:
        types = np.random.choice(y_unique, type_per_batch, replace=False)
        n_choices = {
            i:np.random.choice(
                len(x[np.where(y==i)]),
                images_per_type,
                replace=True
            )
            for i in types
        }
        x_=np.concatenate([x[np.where(y==i)][n_choices[i]].copy() for i in types], axis=0)
        y_=np.concatenate([y[np.where(y==i)][n_choices[i]].copy() for i in types], axis=0)
        yield x_, y_

def embed_to_labels(embed, data_embed, data_y):
    embed=np.array([embed])
    dists=np.linalg.norm(embed-data_embed, axis=-1)
    dists_inds=np.argsort(dists, -1)
    return data_y[dists_inds].copy(), dists[dists_inds].copy()
def predict(db_data, unknow_data):
    data_embed=model.predict(db_data.copy().astype('float32')/255.0)
    data_embed_test=model.predict(unknow_data.copy().astype('float32')/255.0)
    pred_y=[]
    pred_y_dist=[]
    for embed in data_embed_test:
        preds, dists=embed_to_labels(embed, data_embed, train_y)
        pred_y.append(preds[0])
        pred_y_dist.append(dists[0])
    pred_y=np.array(pred_y)
    pred_y_dist=np.array(pred_y_dist)
    return pred_y, pred_y_dist

def evalu():
    pred_y, pred_y_dist=predict(train_x, valid_x)
    pred_y=np.where(pred_y<10,1,0)
    return accuracy_score(valid_y, pred_y)

steps_per_epoch=50
optimizer = Adam(learning_rate=1e-4)

g_train=generator(type_per_batch=10, images_per_type=2) # 64
best=-np.inf
for epoch in range(1, 100):
    losses=[]
    bar=tqdm.tqdm(range(1, steps_per_epoch+1), desc=f'Epoch {epoch}')
    for step in bar:
        x,y=next(g_train)
        if np.random.rand()>0.5:
            x,y=augment(x,y)
        with tf.GradientTape() as tape:
            embeds=model(x.astype('float32')/255.0, training=True)
            loss=batch_hard_triplet_loss(y, embeds)
        grads=tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(loss)
        bar.set_postfix({'loss': tf.reduce_mean(losses).numpy()})
    acc=evalu()
    print('acc:', acc)
    if acc>best:
        best=acc
        model.save_weights('model_embed.h5')
        print('save weights')



model.load_weights('model_embed.h5')


distance_constraint=0.5

p,_=predict(train_x, valid_x)
p=np.where((p<10)&(_<=distance_constraint),1,0)
y_true=valid_y.copy()

acc=accuracy_score(y_true, p)
precision=precision_score(y_true, p, average='macro')
recall=recall_score(y_true, p, average='macro')
print(distance_constraint, acc, precision, recall)
cf_matrix = confusion_matrix(y_true, p)
cf_matrix=(cf_matrix/np.array([np.sum(cf_matrix, axis=1)]).T)
sns.set(rc={'figure.figsize':(7, 5)})
sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='.1%')
plt.show()







