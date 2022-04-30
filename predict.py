import sys
yolor_path='yolor'
sys.path.append(yolor_path)
from models.models import Darknet
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
import torch, cv2, numpy as np, base64
from utils.torch_utils import select_device
model = Darknet(f'{yolor_path}/cfg/yolor_p6.cfg', 1280).cuda()
device = select_device('0')
model.load_state_dict(torch.load('yolor_p6.pt', map_location=device)['model'])
model.to(device).eval()
model.half()
valid_label=['dog', 'cat', 'bear']
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
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
def embed_to_labels(embed, data_embed, data_y):
    embed=np.array([embed])
    dists=np.linalg.norm(embed-data_embed, axis=-1)
    dists_inds=np.argsort(dists, -1)
    return data_y[dists_inds].copy(), dists[dists_inds].copy()
def predict(unknow_data):
    data_embed=model_embed.predict(data_x.copy().astype('float32')/255.0)
    data_embed_test=model_embed.predict(unknow_data.copy().astype('float32')/255.0)
    pred_y=[]
    pred_y_dist=[]
    for embed in data_embed_test:
        preds, dists=embed_to_labels(embed, data_embed, data_y)
        pred_y.append(preds[0])
        pred_y_dist.append(dists[0])
    pred_y=np.array(pred_y)
    pred_y_dist=np.array(pred_y_dist)
    distance_constraint=0.5
    p=np.where((pred_y<10)&(pred_y_dist<=distance_constraint),1,0)
    return p
data_x=np.concatenate([np.load('npys/train_x.npy'),np.load('npys/valid_x.npy')], axis=0)
data_y=np.concatenate([np.load('npys/train_y.npy'),np.load('npys/valid_y.npy')], axis=0)
data_x_neg=np.concatenate([np.load('npys/train_x_neg.npy'),np.load('npys/valid_x_neg.npy')], axis=0)
s=np.max(data_y)+1
data_y_neg=np.arange(s,s+len(data_x_neg))
data_x=np.concatenate([data_x, data_x_neg], axis=0)
data_y=np.concatenate([data_y, data_y_neg], axis=0)
model_embed=resmodel()
model_embed.load_weights('model_embed.h5')


''''''''' TEST '''''''''
img=cv2.imread('__test__/images3.jpg')
crops=[]
boxes=[]
with torch.no_grad():
    inp=letterbox(img.copy(), new_shape=1280, auto_size=64)[0]
    inp=inp[:, :, ::-1].transpose(2, 0, 1)
    inp=np.ascontiguousarray(inp)
    inp=torch.from_numpy(inp).to(device)
    inp=inp.half()
    inp /= 255.0
    if inp.ndimension() == 3:
        inp=inp.unsqueeze(0)
    pred = model(inp, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.5, classes=None, agnostic=False)
    det=pred[0]
    if det is not None and len(det):
        det[:, :4]=scale_coords(inp.shape[2:], det[:, :4], img.shape).round()
        for *xyxy, conf, cls_ in det:
            label = names[int(cls_)]
            if label in valid_label:
                min_x,min_y,max_x,max_y=[int(i.cpu().numpy()) for i in xyxy]
                crop=img[min_y:max_y+1, min_x:max_x+1].copy()
                crop=cv2.resize(crop, (224,224))
                crops.append(crop)
                boxes.append([(min_x,min_y),(max_x,max_y)])
crops=np.array(crops)
ps=predict(crops)
for p,(p1,p2) in zip(ps,boxes):
    if p==1:
        cv2.rectangle(img, p1, p2, (0,255,0), thickness=3)
    else:
        cv2.rectangle(img, p1, p2, (0,0,255), thickness=3)

cv2.imwrite('predict.jpg', img)


