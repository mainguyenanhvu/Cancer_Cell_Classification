from keras.preprocessing import image
from skimage.segmentation import slic
import matplotlib.pylab as pl
import numpy as np
import shap
from matplotlib.pyplot import savefig
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input, ELU, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.vgg16 import preprocess_input
#parameter
channel_axis = 1 if K.image_dim_ordering() == "th" else -1
img_width, img_height = 250, 250
img_channels = 3
classes_num = 9

#model
input = Input(shape=(img_height, img_width, img_channels))
#x = Lambda(filter_layer)(input)
x = Conv2D(32, (3, 3), padding='same')(input)
x = ELU(alpha=1.0)(x)
x = BatchNormalization(axis=channel_axis)(x)
x = Conv2D(64, (3, 3), padding="same")(x)
x = ELU(alpha=1.0)(x)
x = BatchNormalization(axis=channel_axis)(x)
x = MaxPooling2D(pool_size=(3, 3))(x)

x = Conv2D(64, (5, 5), padding="same")(x)
x = ELU(alpha=1.0)(x)
x = BatchNormalization(axis=channel_axis)(x)
x = MaxPooling2D(pool_size=(3, 3))(x)

x = Conv2D(128, (3, 3), padding="same")(x)
x = ELU(alpha=1.0)(x)
x = BatchNormalization(axis=channel_axis)(x)
x = MaxPooling2D(pool_size=(3, 3))(x)

x = Conv2D(128, (3, 3), padding="same")(x)
x = ELU(alpha=1.0)(x)
x = BatchNormalization(axis=channel_axis)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(1000)(x)
x = ELU(alpha=1.0)(x)
x = BatchNormalization(axis=channel_axis)(x)
#x = Dense(1000)(x)
#x = ELU(alpha=1.0)(x)
#x = BatchNormalization(axis=channel_axis)(x)

x = Dropout(0.5)(x)
output = Dense(classes_num, activation='softmax')(x)
model = Model(inputs=input, outputs=output)

model.load_weights('/home/cngc3/HPA/weight/XVII-4-cont-200-reseg6_weights.85-0.37.hdf5')

#list out feature
feature_names = {"0": ["o", "A549"],
                 "1": ["n", "CACO-2"],
                 "2": ["n", "HEK"],
                 "3": ["i", "HeLa"],
                 "4": ["_", "MCF7"],
                 "5": ["c", "PC-3"],
                 "6": ["h", "RT4"],
                 "7": ["a", "U-2"],
                 "8": ["n", "U-251"]}

# define a function that depends on a binary mask representing if an image region is hidden
def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0,1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i,:,:,:] = image
        for j in range(zs.shape[1]):
            if zs[i,j] == 0:
                out[i][segmentation == j,:] = background
    return out
def f(z):
    a = (mask_image(z, segments_slic, img_orig, background = 0)) #function of an image prediction with background = 0
    a /= 255.
    return model.predict(a)

"""def f(z):
    return model.predict(mask_image(z, segments_slic, img_orig, 255))"""

def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out
def get_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3]) #imput the image, resize to model input shape
    x = image.img_to_array(img) #process the image for model input
    x = np.expand_dims(x, axis=0)
    x /= 255.
    return img, x

# make a color map
from matplotlib.colors import LinearSegmentedColormap
colors = []
for l in np.linspace(1,0,100):
    colors.append((245/255,39/255,87/255,l))
for l in np.linspace(0,1,100):
    colors.append((24/255,196/255,93/255,l))
cm = LinearSegmentedColormap.from_list("shap", colors)

#load data list
import pandas as pd
pd_list = pd.read_csv("./assets/top10images_each_class-umap.csv",header=None)
filelist = pd_list[0].tolist()

for file in filelist:
    img, x = get_image(file)
    img_orig = image.img_to_array(img).copy()
    # segment the image so we don't have to explain every pixel
    segments_slic = slic(img, n_segments=100, compactness=20, sigma=3)

    # use Kernel SHAP to explain the network's predictions
    explainer = shap.KernelExplainer(f, np.zeros((1, 50))) #input the function
    shap_values = explainer.shap_values(np.ones((1, 50)), nsamples=1000, l1_reg=0)  # return the values, 9000 times run

    preds = model.predict(x)
    top_preds = np.argsort(-preds)

    # plot our explanations
    fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(10, 4))
    inds = top_preds[0]
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title('Original')
    max_val = np.max([np.max(np.abs(shap_values[i][:, :-1])) for i in range(len(shap_values))])
    for i in range(1):
        m = fill_segmentation(shap_values[inds[i]][0], segments_slic)
        axes[i + 1].set_title(feature_names[str(inds[i])][1])
        axes[i + 1].imshow(img.convert('LA'), alpha=0.15)
        im = axes[i + 1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
        axes[i + 1].axis('off')
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
    cb.outline.set_visible(False)
    #pl.show()
    #pl.close(fig)
    name = "%s_SHAP" % ((file.split("training_seg_5/")[1]).split(".png")[0])
    name = name.replace("/", "-")
    name = "assets/SHAP-UMAP-2/%s" % (name)
    savefig(name, bbox_inches='tight')


# load an image
img, x = get_image("training_seg_5/A549/-1OBWqdh7Z2nyQw5qpXWPR3gcvSDk8kGB_20180329_cell5.png")
img_orig = image.img_to_array(img).copy()

# segment the image so we don't have to explain every pixel
segments_slic = slic(img, n_segments=100, compactness=20, sigma=3)

# use Kernel SHAP to explain the network's predictions
explainer = shap.KernelExplainer(f, np.zeros((1,50)))
shap_values = explainer.shap_values(np.ones((1,50)), nsamples=1000) # runs VGG16 1000 times

preds = model.predict(x)
top_preds = np.argsort(-preds)

# plot our explanations
fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(12,4))
inds = top_preds[0]
axes[0].imshow(img)
axes[0].axis('off')
axes[0].set_title('Original')
max_val = np.max([np.max(np.abs(shap_values[i][:,:-1])) for i in range(len(shap_values))])
for i in range(1):
    m = fill_segmentation(shap_values[inds[i]][0], segments_slic)
    axes[i+1].set_title(feature_names[str(inds[i])][1])
    axes[i+1].imshow(img.convert('LA'), alpha=0.15)
    im = axes[i+1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
    axes[i+1].axis('off')
cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
cb.outline.set_visible(False)
pl.show()
name = "%s_SHAP" % ((file.split("training_seg_5/")[1]).split(".png")[0])
name = name.replace("/", "-")
name = "assets/%s" % (name)
savefig(name, bbox_inches='tight')

for i in range(0,9):
    print (shap_values[i].sum())