# Data_augmentation.py
# Reference: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from os import path, makedirs


class DataAugmentation:
    @staticmethod
    def dataAugmentationPreview(imagePath, writeDir):

        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

        img = load_img(imagePath)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        if (path.isdir(writeDir) == False):
            makedirs(writeDir, exist_ok=True)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=writeDir, save_prefix='cervixCancer', save_format='jpeg'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely

    @staticmethod
    def prepareDataAugmentation(train_data):
        '''
        Reference: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        :param train_data:
        :return:
        '''
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

        datagen.fit(train_data)

        return datagen


if __name__ == '__main__':
    DataAugmentation.dataAugmentationPreview(imagePath="../data/train_256/Type_2/178.jpg",
                                             writeDir="../data/augmentationPreview/")
