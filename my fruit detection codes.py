#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opendatasets')


# In[9]:


import opendatasets as od


# In[10]:


dataset='https://www.kaggle.com/datasets/whenamancodes/date-fruit-datasets'


# In[ ]:





# In[11]:


od.download(dataset)


# In[13]:


import pandas as pd 


# In[16]:


spreadsheet=pd.read_csv('C:/Users/loukm/OneDrive/Desktop/archive/Date_Fruit_Datasets.csv')


# In[17]:


spreadsheet.head()


# In[121]:


spreadsheet.head()
spreadsheet.EQDIASQ.hist()
import matplotlib.pyplot as plt
plt.show()


# In[30]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:





# In[ ]:





# In[64]:


df = pd.read_csv('C:/Users/loukm/OneDrive/Desktop/archive/Date_Fruit_Datasets.csv')
print(df.head(10))


# In[68]:


import numpy as np
from pandas import DataFrame
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

Index=['0','1','2','3','4','5']
Cols =['PERIMETER','MAJOR_AXIS','MINOR_AXIS','EQDIASQ','KurtosisRG',' ASPECT_RATIO']
df = DataFrame(abs(np.random.randn(6,6)), index=Index, columns=Cols)


sns.heatmap(df, annot=True)


# In[69]:


sns.heatmap(df.corr());


# In[122]:


# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


# Creating dataset
np.random.seed(10)
data = np.random.normal(200, 40, 300)

fig = plt.figure(figsize =(20, 10))

# Creating plot
plt.boxplot(data)

# show plot
plt.show()


# In[192]:


fig=px.histogram(df,x=['KurtosisRR ','ASPECT_RATIO'])
fig.show()


# In[ ]:





# In[ ]:





# In[123]:


sns.pairplot(df)


# In[84]:


fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].imshow(_img)
i = axes[1].imshow(grad_eval,cmap="jet",alpha=0.8)
fig.colorbar(i)


# In[86]:


df.info()


# In[92]:


df['KurtosisRG'].value_counts()


# In[91]:


df.head()


# In[93]:


for i in df.index:
   print(df.at[i, 'KurtosisRG'])


# In[95]:


for i in df.index:
    df.at[i, 'KurtosisRG'] = df.at[i, 'KurtosisRG'].upper()
df['KurtosisRG'].value_counts()


# In[97]:


df2 = pd.read_csv('C:/Users/loukm/OneDrive/Desktop/archive/Date_Fruit_Datasets.csv')
df2.loc[df2["KurtosisRG"] == "llc", "KurtosisRG"] = "LLC"
df2.loc[df2["KurtosisRG"] == "c corp", "KurtosisRG"] = "C corp"
df2.loc[df2["KurtosisRG"] == "s corp", "KurtosisRG"] = "S corp"
df2['KurtosisRG'].value_counts()


# In[99]:


df2.to_csv('C:/Users/loukm/OneDrive/Desktop/archive/Date_Fruit_Datasets.csv')
df3 = pd.read_csv('C:/Users/loukm/OneDrive/Desktop/archive/Date_Fruit_Datasets.csv')
df3['KurtosisRG'].value_counts()


# In[124]:


import os
import shutil
import fruits


# In[108]:


My_data_folder=r"C:\Users\loukm\OneDrive\Desktop\fruits"
train_folder=r"C:\Users\loukm\OneDrive\Desktop\train"
validation_folder=r"C:\Users\loukm\OneDrive\Desktop\validation"
test_folder=r"C:\Users\loukm\OneDrive\Desktop\test"


# In[125]:


os.mkdir(train_folder)
os.mkdir(validation_folder)
os.mkdir(test_folder)


# In[183]:


import numpy as np
import matplotlib.pyplot
import os
import cv2

DATADIR = "C:/Users/loukm/OneDrive/Desktop/fruits/MY_data/test"
CATEGORIES = ["banana","apple","orange","kiwi","mango","orange","pineapple","strawberries","watermelon"]

for categories in CATEGORIES:
    path = os.path.join(DATADIR, categories)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array , cmap="gray")
        plt.show()
        break
        
    



# In[ ]:


print(img_array.shape)
    


# In[184]:


IMG_SIZE = 30
new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap = 'gray')
plt.show()


# In[186]:


training_data =[]
def create_training_data():
    for categories in CATEGORIES:
        path = os.path.join(DATADIR, categories)
        class_num = CATEGORIES.index(categories)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                training_data.apprend([new_array, class_num])
         
            except Exception as e:
                pass
            
create_training_data()
            


# In[185]:


print(len(training_data))


# In[198]:



import random 
random.shuffle(training_data)


# In[190]:


for sample in training_data[:10]:
    print(sample[1])


# In[ ]:





# In[ ]:





# In[208]:


import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection.model import GridSearchCV
mlp =MLPClassifier(random_state =10)
grid_values =('max_iter',[200, 1000, 2000],'hidden_layer_sizes',[(2,),(3,),(10,),(20,)])
grid_mlp = GridSearchCV(mlp, param_grid =grid_values, scoring = 'accuracy',cv= 5)


# In[207]:


grid_mlp.fit(x_train, y_train)


# In[ ]:





# In[ ]:





# In[1]:


pip install tensorflow


# In[2]:


pip install keras-vis tensorflow


# In[216]:


"""A2_final.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1l9XlOmp351XoXHOOskmIZWrfWn6hY0dQ
"""


import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.applications.vgg16 import preprocess_input

import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from vis.utils import utils
from tensorflow.keras.applications.vgg16 import decode_predictions
import json

#utility function to show images
def display_imgs(images):
  subprot_args = {
   'nrows': 1,
   'ncols': 4,
   'figsize': (6, 3),
   'subplot_kw': {'xticks': [], 'yticks': []}
  }
  f, ax = plt.subplots(**subprot_args)
  for i in range(len(images)):
    ax[i].imshow(images[i])
  plt.tight_layout()
  plt.show()

# Load model
model = Model(weights='imagenet', include_top=True)
model.summary()

#Load images
#img1 = load_img('/content/drive/MyDrive/test_imgs/cat.jpg', target_size=(224, 224))
#img2 = load_img('/content/drive/MyDrive/test_imgs/dog.jpg', target_size=(224, 224))
#img3 = load_img('/content/drive/MyDrive/test_imgs/hen.jpg', target_size=(224, 224))
#img4 = load_img('/content/drive/MyDrive/test_imgs/tiger.jpeg', target_size=(224, 224))

img1 = load_img('ovacado.jpeg', target_size=(224, 224))
img2 = load_img('orange.jpeg', target_size=(224, 224))
img3 = load_img('kiwi.jpeg', target_size=(224, 224))
img4 = load_img('strawberry.jpeg', target_size=(224, 224))

#plt.imshow(img1)
#plt.show()
#plt.imshow(img2)
#plt.show()
#plt.imshow(img3)
#plt.show()
#plt.imshow(img4)
#plt.show()

#create array of images
images = np.asarray([np.array(img1), np.array(img2), np.array(img3), np.array(img4)])

#show images
display_imgs(images)

#convert to numpy array for reshaping
img1 = img_to_array(img1)
img2 = img_to_array(img2)
img3 = img_to_array(img3)
img4 = img_to_array(img4)

#reshape to prepare for processing
img1 = img1.reshape(1,224,224,3)
img2 = img2.reshape(1,224,224,3)
img3 = img3.reshape(1,224,224,3)
img4 = img4.reshape(1,224,224,3)

#preprocess to prepare for input
img1 = preprocess_input(img1)
img2 = preprocess_input(img2)
img3 = preprocess_input(img3)
img4 = preprocess_input(img4)

# predictions with input images
yhat1 = model.predict(img1)
yhat2 = model.predict(img2)
yhat3 = model.predict(img3)
yhat4 = model.predict(img4)

#decode predictions
label1 = decode_predictions(yhat1)
label2 = decode_predictions(yhat2)
label3 = decode_predictions(yhat3)
label4 = decode_predictions(yhat4)

# extract top most prediction for each input
label1 = label1[0][0]
label2 = label2[0][0]
label3 = label3[0][0]
label4 = label4[0][0]

#plt.imshow(image1)
print('%s (%.2f%%)' % (label1[1], label1[2]*100))
#plt.imshow(image2)
print('%s (%.2f%%)' % (label2[1], label2[2]*100))
#plt.imshow(image3)
print('%s (%.2f%%)' % (label3[1], label3[2]*100))
#plt.imshow(image4)
print('%s (%.2f%%)' % (label4[1], label4[2]*100))

#download class file
get_ipython().system('wget "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"')

#prepare 1000 classes
CLASS_INDEX = json.load(open("imagenet_class_index.json"))
classlabel = []
for i_dict in range(len(CLASS_INDEX)):
    classlabel.append(CLASS_INDEX[str(i_dict)][1])
print("N of class={}".format(len(classlabel)))

#Top 5 classes predicted
class_idxs_sorted1 = np.argsort(yhat1.flatten())[::-1]
class_idxs_sorted2 = np.argsort(yhat2.flatten())[::-1]
class_idxs_sorted3 = np.argsort(yhat3.flatten())[::-1]
class_idxs_sorted4 = np.argsort(yhat4.flatten())[::-1]

topNclass         = 5

print('\nfirst image\n')
for i, idx in enumerate(class_idxs_sorted1[:topNclass]):
    print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
          i + 1,classlabel[idx],idx,yhat1[0,idx]))

print('\nsecond image\n')
for i, idx in enumerate(class_idxs_sorted2[:topNclass]):
    print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
          i + 1,classlabel[idx],idx,yhat2[0,idx]))

print('\nthird image\n')
for i, idx in enumerate(class_idxs_sorted3[:topNclass]):
    print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
          i + 1,classlabel[idx],idx,yhat3[0,idx]))

print('\nFourth image\n')
for i, idx in enumerate(class_idxs_sorted4[:topNclass]):
    print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
          i + 1,classlabel[idx],idx,yhat4[0,idx]))

# swap softmax layer with linear layer 
layer_idx = utils.find_layer_idx(model, 'predictions')
model.layers[-1].activation = tf.keras.activations.linear
model = utils.apply_modifications(model)

#get the input image index
from tf_keras_vis.utils.scores import CategoricalScore
#cat - 281, dog -235 , hen -8, tiger - 292
score = CategoricalScore([281, 235, 8 , 292])

from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

input_classes = ['Cat', 'Dog', 'Hen', 'Tiger']

input_images = preprocess_input(images)

# Create Gradcam object
gradcam = Gradcam(model,
                  clone=True)

# Generate heatmap with GradCAM
cam = gradcam(score,
              input_images,
              penultimate_layer=-1)

#show generated images
f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
for i, img_class in enumerate(input_classes):
    heatmap = np.uint8(cm.jet(cam[i])[..., :4] * 255)
    ax[i].set_title(img_class, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
    ax[i].axis('off')
plt.tight_layout()
plt.show()

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

#Create Saliency object
saliency = Saliency(model, clone=False)

# Generate saliency map
saliency_map = saliency(score, input_images)
saliency_map = normalize(saliency_map)

subprot_args = {
   'nrows': 1,
   'ncols': 4,
   'figsize': (6, 3),
   'subplot_kw': {'xticks': [], 'yticks': []}
}
f, ax = plt.subplots(**subprot_args)
for i in range(len(saliency_map)):
   ax[i].imshow(saliency_map[i], cmap='jet')
plt.tight_layout()
plt.show()

saliency_map = saliency(score, input_images, smooth_samples=20)
saliency_map = normalize(saliency_map)

f, ax = plt.subplots(**subprot_args)
for i in range(len(saliency_map)):
   ax[i].imshow(saliency_map[i], cmap='jet')
plt.tight_layout()
plt.show()


# In[ ]:


conda create --name tensorflow python


# In[ ]:


conda install tensorflow


# In[ ]:




