# Test num 001. Marek
# <https://www.kaggle.com/marek3000/test-num-001/code/>

from image_utils import ImageUtils
imgSize = 128
useAditional = False

if __name__ == '__main__':
    imgUtils = ImageUtils(imgSize, useAditional)

    imgUtils.dataPreparation()
