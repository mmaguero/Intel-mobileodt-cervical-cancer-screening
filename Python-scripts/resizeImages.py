# Test num 001. Marek
# <https://www.kaggle.com/marek3000/test-num-001/code/>

from image_utils import ImageUtils
imgSize = 10
useAditional = True
keepAspectRatio = False

if __name__ == '__main__':
    imgUtils = ImageUtils(imgSize, useAditional, keepAspectRatio)

    imgUtils.dataPreparation()
