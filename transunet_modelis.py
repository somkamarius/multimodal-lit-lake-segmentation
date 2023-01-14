import os
import cv2
import numpy as np
from PIL import Image
from patchify import patchify
from matplotlib import pyplot as plt
import random
import numpy as np
import segmentation_models as sm
from keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import imageio.v3 as iio

scaler = MinMaxScaler()

root_directory = '/Users/bf3237/PycharmProjects/bakalauras/data'
# /images ir /masks

patch_size = 256  # investigate ar apsimoka

image_dataset = []
for path, subdirs, files in os.walk(root_directory):
    # print(subdirs)
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'images_multiband':  # Find all 'images' directories
        images = os.listdir(path)  # List of all image names in this subdirectory
        for i, image_name in enumerate(images):
            if image_name.endswith(".tif"):  # Only read tif images...

                image = iio.imread(path + "/" + image_name, extension='.tif')
                # image_rgb = cv2.imread('/Users/bf3237/PycharmProjects/bakalauras/data/images' + "/" + image_name, 1)  # Read each image as BGR
                # print(image)
                SIZE_X = (image.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
                # print(varis)
                # print(image.shape(3))
                print(type(image))
                print(image.shape[2])
                SIZE_Y = (image.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size
                all_bands = []
                for i in range(0, image.shape[2]):
                    single_band_image = image[:,:,i]
                    single_band_image = Image.fromarray(single_band_image)
                    single_band_image = single_band_image.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
                    single_band_image = np.array(single_band_image)
                    all_bands.append(single_band_image)
                image = np.dstack(tuple(all_bands))

                # Extract patches from each image
                print("Now patchifying image:", path + "/" + image_name)
                patches_img = patchify(image, (patch_size, patch_size, 3),
                                       step=patch_size)  # Step=256 for 256 patches means no overlap

                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        single_patch_img = patches_img[i, j, :, :]

                        # Use minmaxscaler instead of just dividing by 255.
                        single_patch_img = scaler.fit_transform(
                            single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)

                        # single_patch_img = (single_patch_img.astype('float32')) / 255.
                        single_patch_img = single_patch_img[
                            0]  # Drop the extra unecessary dimension that patchify adds.
                        image_dataset.append(single_patch_img)

mask_dataset = []
for path, subdirs, files in os.walk(root_directory):
    # print(path)
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'masks':  # Find all 'images' directories
        masks = os.listdir(path)  # List of all image names in this subdirectory
        for i, mask_name in enumerate(masks):
            if mask_name.endswith(".tif"):  # Only read png images... (masks in this dataset)

                mask = cv2.imread(path + "/" + mask_name,
                                  1)  # Read each image as Grey (or color but remember to map each color to an integer)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
                SIZE_Y = (mask.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size
                mask = Image.fromarray(mask)
                mask = mask.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
                mask = np.array(mask)

                # Extract patches from each image
                print("Now patchifying mask:", path + "/" + mask_name)
                patches_mask = patchify(mask, (patch_size, patch_size, 3),
                                        step=patch_size)  # Step=256 for 256 patches means no overlap

                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[i, j, :, :]
                        # single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                        single_patch_mask = single_patch_mask[
                            0]  # Drop the extra unecessary dimension that patchify adds.
                        mask_dataset.append(single_patch_mask)

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

print('image_dataset', len(image_dataset))
print('mask_dataset', len(mask_dataset))


# eck, view few mages

image_number = random.randint(0, len(image_dataset))
image_number = 6
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3)))
plt.subplot(122)
plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
plt.show()



Lake = '#000000'.lstrip('#')
Lake = np.array(tuple(int(Lake[i:i + 2], 16) for i in (0, 2, 4)))  # 60, 16, 152

Unlabeled = '#FFFFFF'.lstrip('#')
Unlabeled = np.array(tuple(int(Unlabeled[i:i + 2], 16) for i in (0, 2, 4)))  # 155, 155, 155

label = single_patch_mask


def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format.
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape, dtype=np.uint8)
    label_seg[np.all(label == Lake, axis=-1)] = 0
    label_seg[np.all(label == Unlabeled, axis=-1)] = 1

    label_seg = label_seg[:, :, 0]  # Just take the first channel, no need for all 3 channels

    return label_seg


labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)

labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)

print("Unique labels in label dataset are: ", np.unique(labels))

n_classes = len(np.unique(labels))

from keras.utils import to_categorical

labels_cat = to_categorical(labels, num_classes=n_classes)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.20, random_state=42)

weights = [0.5, 0.5]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Lambda
from keras import backend as K


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


metrics = ['accuracy', jacard_coef]

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# preprocess input
X_train_prepr = preprocess_input(X_train)
X_test_prepr = preprocess_input(X_test)

# define model
# model_resnet_backbone = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')


# define number of channels
N = X_test_prepr.shape[-1]

base_model = sm.Linknet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')

inp = Input(shape=(None, None, N))
l1 = Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
out = base_model(l1)

model_resnet_backbone = Model(inp, out, name=base_model.name)

# compile keras model with defined optimozer, loss and metrics
# model_resnet_backbone.compile(optimizer='adam', loss=focal_loss, metrics=metrics)
model_resnet_backbone.compile(optimizer='adam', loss=total_loss, metrics=metrics)

print(model_resnet_backbone.summary())

history2 = model_resnet_backbone.fit(X_train_prepr,
                                     y_train,
                                     batch_size=16,
                                     epochs=30,
                                     verbose=1,
                                     validation_data=(X_test_prepr, y_test))
model_resnet_backbone.save("linknet_multiband")
history = history2
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['jacard_coef']
val_acc = history.history['val_jacard_coef']

plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()

y_pred = model_resnet_backbone.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)

from keras.metrics import MeanIoU

# n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test_argmax, y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

import random

test_img_number = random.randint(0, len(X_test))
print(test_img_number
      )
test_img = X_test[test_img_number]
ground_truth = y_test_argmax[test_img_number]
# test_img_norm=test_img[:,:,0][:,:,None]
test_img_input = np.expand_dims(test_img, 0)
prediction = (model_resnet_backbone.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()
