import cv2, numpy as np, tqdm, os

cv2.imread=lambda path: cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


''''''''' Positive '''''''''
data_x=[]
data_y=[]
for ind,pack in enumerate(os.listdir('dataset/images')):
    bar=tqdm.tqdm(os.listdir(f'dataset/images/{pack}'))
    bar.set_description(pack)
    for file in bar:
        img=cv2.imread(f'dataset/images/{pack}/{file}')
        img=cv2.resize(img, (224,224))
        data_x.append(img)
        data_y.append(ind)
for ind,pack in enumerate(os.listdir('dataset/instagram_test/images')):
    bar=tqdm.tqdm(os.listdir(f'dataset/instagram_test/images/{pack}'))
    bar.set_description(pack)
    for file in bar:
        img=cv2.imread(f'dataset/instagram_test/images/{pack}/{file}')
        img=cv2.resize(img, (224,224))
        data_x.append(img)
        data_y.append(ind)

data_x=np.array(data_x)
data_y=np.array(data_y)

np.random.seed(20220425)
train_x=[]
train_y=[]
valid_x=[]
valid_y=[]
for cla in np.unique(data_y):
    indices=np.where(data_y==cla)[0]
    np.random.shuffle(indices)
    valid_pick=np.random.choice(indices, size=int(len(indices)*0.1), replace=False)
    train_pick=indices[~np.isin(indices, valid_pick)]
    train_x.extend(data_x[train_pick])
    train_y.extend(data_y[train_pick])
    valid_x.extend(data_x[valid_pick])
    valid_y.extend(data_y[valid_pick])

xy=list(zip(train_x,train_y))
np.random.shuffle(xy)
train_x,train_y=zip(*xy)
xy=list(zip(valid_x,valid_y))
np.random.shuffle(xy)
valid_x,valid_y=zip(*xy)
del xy

train_x=np.array(train_x)
train_y=np.array(train_y)
valid_x=np.array(valid_x)
valid_y=np.array(valid_y)
print()
print(train_x.shape, train_x.dtype, train_y.shape, train_y.dtype)
print(valid_x.shape, valid_x.dtype, valid_y.shape, valid_y.dtype)
'''''''''          '''''''''



''''''''' Negative '''''''''
files=np.array(os.listdir('dataset/coco_test/crop'))
np.random.seed(20220425)
train_files=np.random.choice(files, size=500, replace=False)
valid_files=files[~np.isin(files,train_files)]
train_x_neg=[]
valid_x_neg=[]
for file in tqdm.tqdm(train_files):
    img=cv2.imread(f'dataset/coco_test/crop/{file}')
    img=cv2.resize(img, (224,224))
    train_x_neg.append(img)
for file in tqdm.tqdm(valid_files):
    img=cv2.imread(f'dataset/coco_test/crop/{file}')
    img=cv2.resize(img, (224,224))
    valid_x_neg.append(img)
train_x_neg=np.array(train_x_neg)
valid_x_neg=np.array(valid_x_neg)
print()
print(train_x_neg.shape, train_x_neg.dtype)
print(valid_x_neg.shape, valid_x_neg.dtype)
'''''''''          '''''''''


np.save('npys/train_x', train_x)
np.save('npys/train_y', train_y)
np.save('npys/valid_x', valid_x)
np.save('npys/valid_y', valid_y)
np.save('npys/train_x_neg', train_x_neg)
np.save('npys/valid_x_neg', valid_x_neg)







