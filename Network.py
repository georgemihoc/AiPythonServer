import glob
import numpy as np
import cv2 as cv
import nibabel as nib
from keras.models import Model
from keras.layers import Concatenate, UpSampling2D, Dense, concatenate, Input, MaxPool2D, Conv2D, Conv2DTranspose, \
    BatchNormalization, Dropout
from PIL import Image
from keras import backend as K
import SimpleITK as sitk
from skimage import measure


import os

checkpoint_path = "model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

FILENAME = "files/file.nii"
PHOTO_SIZE = 320
EPOCHS = 10
NF = 16
INPUT_SIZE = 120


def cropCenter(matrixInput):
    cropedImage = []
    for i in range(PHOTO_SIZE // 2):
        line = []
        for j in range(PHOTO_SIZE // 2):
            x = i + PHOTO_SIZE // 4
            y = j + PHOTO_SIZE // 4
            line.append(matrixInput[x][y])
            line.append(matrixInput[x][y])
        cropedImage.append(line)
        cropedImage.append(line)

    return cropedImage


def keepCenter(outputPixelList):
    result = []
    for i in range(PHOTO_SIZE * PHOTO_SIZE):
        result.append(250)
    visited = []
    queue = []
    x = PHOTO_SIZE // 2
    y = PHOTO_SIZE // 2

    start = [x, y]
    queue.append(start)

    while len(queue) > 0:
        current = queue.pop(0)
        current_x = current[0]
        current_y = current[1]

        if outputPixelList[current_x * PHOTO_SIZE + current_y + 1] == 250 and [current_x, current_y + 1] not in visited:
            new_neighboor = [current_x, current_y + 1]
            queue.append(new_neighboor)
            visited.append([current_x, current_y + 1])
            result[current_x * PHOTO_SIZE + current_y + 1] = 0

        if outputPixelList[current_x * PHOTO_SIZE + current_y - 1] == 250 and [current_x, current_y - 1] not in visited:
            new_neighboor = [current_x, current_y - 1]
            queue.append(new_neighboor)
            visited.append([current_x, current_y - 1])
            result[current_x * PHOTO_SIZE + current_y - 1] = 0

        if outputPixelList[(current_x + 1) * PHOTO_SIZE + current_y] == 250 and [current_x + 1,
                                                                                 current_y] not in visited:
            new_neighboor = [current_x + 1, current_y]
            queue.append(new_neighboor)
            visited.append([current_x + 1, current_y])
            result[(current_x + 1) * PHOTO_SIZE + current_y] = 0

        if outputPixelList[(current_x - 1) * PHOTO_SIZE + current_y] == 250 and [current_x - 1,
                                                                                 current_y] not in visited:
            new_neighboor = [current_x - 1, current_y]
            queue.append(new_neighboor)
            visited.append([current_x - 1, current_y])
            result[(current_x - 1) * PHOTO_SIZE + current_y] = 0
    '''        
    for i in range(PHOTO_SIZE):
        for j in range(PHOTO_SIZE):
            elem = [i,j]
            if elem not in visited:
                pixel = 0
                outputPixelList[i * PHOTO_SIZE + j] = pixel
    '''
    return result


def probabilityToPixel(outputPixelList):
    # print(outputPixelList)
    for i in range(len(outputPixelList)):
        pixel = outputPixelList[i]
        # 255 II ALB
        if pixel <= 0.5:
            pixel = 0
        else:
            pixel = 250

        # pixel = pixel * 255
        outputPixelList[i] = pixel

    return outputPixelList


def sigmoidToPhoto(outputExample, sliceFile):
    outputPhoto = outputExample[0]

    outputPixelList = np.squeeze(np.asarray(outputPhoto))
    outputPixelList = np.matrix.flatten(outputPixelList)

    # print(outputPixelList)
    outputPixelList = probabilityToPixel(outputPixelList)
    outputPixelList = keepCenter(outputPixelList)

    outputPixelList = np.reshape(outputPixelList, (PHOTO_SIZE, PHOTO_SIZE))
    img = Image.fromarray((outputPixelList).astype(np.uint8))
    # img = Image.fromarray(outputPixelList)
    img = img.convert('RGB')
    outputSliceFile = sliceFile.split(".")[0]
    for i in range(1):
        img.save(outputSliceFile + str(i) + "O" + ".jpeg")
    # img.show()


def loadOnePhoto(filename):
    trainInput = []

    imInput = Image.open(filename, 'r')
    width, height = imInput.size
    # print(width, height)
    if width != PHOTO_SIZE or height != PHOTO_SIZE:
        imInput = imInput.resize((PHOTO_SIZE, PHOTO_SIZE))
    # width, height = imInput.size
    pixel_valuesInput = list(imInput.getdata())

    for i in range(len(pixel_valuesInput)):
        pixel_valuesInput[i] = pixel_valuesInput[i] / 255

    inputMatrix = vectorToMatrix(pixel_valuesInput, PHOTO_SIZE)
    inputMatrix = cropCenter(inputMatrix)

    inputCube = []
    inputCube.append(inputMatrix)

    trainInput.append(inputCube)
    trainInput = np.reshape(trainInput, (1, PHOTO_SIZE, PHOTO_SIZE, 1))
    # inputCube = np.reshape(trainInput, (PHOTO_SIZE, PHOTO_SIZE,1))
    return trainInput


def predictSlice(model, sliceFile):
    model.load_weights(checkpoint_path)

    inputExample = loadOnePhoto(sliceFile)
    print("Predicting image")
    outputExample = model.predict(inputExample)

    # print(outputExample)
    sigmoidToPhoto(outputExample, sliceFile)


'''
    Return the loss function so it can be used by the model
'''

# def custom_loss():
#     '''
#         Returns value of dice lose
#         input: y_true - real result of segmentation
#                y_pred - predicted result of segmentation
#
#         dice_loss =   ( 2 *  A intersect. B )  /  ( A + B )
#     '''
#
#     def dice_loss(y_true, y_pred):
#
#         # print(y_true)
#         numitor = 0
#         numarator = 0
#         # sess = tf.compat.v1.Session()
#         # print(sess.run(y_true)[1,1,1])
#
#         for i in range(PHOTO_SIZE):
#             for j in range(PHOTO_SIZE):
#                 sum = y_true.eval()[i][j] + y_pred.eval()[i][j]
#                 intersect = y_true.eval()[i][j] * y_pred.eval()[i][j]
#                 numitor = numitor + sum
#                 numarator = numarator + 2 * intersect
#
#         # return 1 - (numarator / numitor)
#         return 1
#
#     return dice_loss


'''
    Returns value of dice_coef for y_true, y_pred
    input: y_true, y_pred - tensors 
    output: value [0 - 1] 1 - for perfect prediction
'''


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=-1)
    suma = (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
    return (2 * intersection + smooth) / suma


'''
    Return the loss function so it can be used by the model
    Dice_coef is 1 for perfect result -> For dice_loss we use 1 - dice_coef
'''


def custom_loss2():
    def dice_loss(y_true, y_pred):
        dice_coef_result = dice_coef(y_true, y_pred)
        return 1 - dice_coef_result

    return dice_loss


'''
    Functions plots the model's performance using the history of the model.fit()
    input: history 
'''

# def plotModelPerformance(history):
#     plt.plot(history.history['dice_coef'])
#     plt.title('model accuracy')
#     plt.ylabel('dice_coef')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()


'''
    Transform liniar list into a square matrix with n = size
    input: vector - list/vector to be transformed
           size - number of columns/rows for the new matrix 
    output: matrix [size, size]  - contains the elements from vector arranged as a matrix.

    Function used in loadData.
'''


def vectorToMatrix(vector, size):
    matrix = []
    line = []
    for elem in vector:
        line.append(float(elem))
        if len(line) % size == 0:
            matrix.append(line)
            line = []

    return matrix


'''
    Load train input and output for the network
    input: folderPath - string:  path to the folder where images are stored
    output: trainInput - List [INPUT_SIZE, PHOTO_SIZE, PHOTO_SIZE, 1] - List with all the pixels for every input image.
            trainOutput - List [INPUT_SIZE, PHOTO_SIZE, PHOTO_SIZ, 1] - List with all the pixels for every output image.
'''


def loadData(folderPath):
    trainInput = []
    trainOutput = []

    # prefix = "data/MSD_data_sagital/"
    prefix = folderPath
    # views = ["axial/","coronal/","sagital/"]
    # views = ["imagesTr"]

    for i in range(INPUT_SIZE):
        # filenameInput = prefix + "training/" + view + "img" + str(i) +".jpeg"
        # filenameOutput = prefix + "ground_truth/" + view + "img" + str(i) +".jpg"
        filenameInput = prefix + "imagesTr" + "/img" + str(i) + ".jpeg"
        filenameOutput = prefix + "labelsTr_jpg" + "/img" + str(i) + ".jpeg"

        imInput = Image.open(filenameInput, 'r')
        # imInput = imInput.resize((PHOTO_SIZE, PHOTO_SIZE))

        pixel_valuesInput = list(imInput.getdata())

        for i in range(len(pixel_valuesInput)):
            pixel_valuesInput[i] = pixel_valuesInput[i] / 255

        # print(pixel_valuesInput)

        inputMatrix = vectorToMatrix(pixel_valuesInput, PHOTO_SIZE)
        inputCube = []
        inputCube.append(inputMatrix)

        trainInput.append(inputCube)

        imOutput = Image.open(filenameOutput, 'r')
        imOutput = imOutput.resize((PHOTO_SIZE, PHOTO_SIZE))
        # width, height = imOutput.size
        pixel_valuesOutput = list(imOutput.getdata())

        for i in range(len(pixel_valuesOutput)):
            if pixel_valuesOutput[i] > 245:
                pixel_valuesOutput[i] = 1
            else:
                pixel_valuesOutput[i] = 0

        outputMatrix = vectorToMatrix(pixel_valuesOutput, PHOTO_SIZE)
        outputCube = []
        outputCube.append(outputMatrix)

        trainOutput.append(outputCube)

    trainInput = np.reshape(trainInput, (INPUT_SIZE, PHOTO_SIZE, PHOTO_SIZE, 1))
    trainOutput = np.reshape(trainOutput, (INPUT_SIZE, PHOTO_SIZE, PHOTO_SIZE, 1))
    # trainInput = np.expand_dims(trainInput, axis = 0)
    # trainOutput = np.expand_dims(trainOutput, axis = 0)
    return trainInput, trainOutput


'''
    Create first module of our network
    output: input - Input layer
              x - current state of the model
'''


def getFirstModule():
    inputs = Input(shape=(PHOTO_SIZE, PHOTO_SIZE, 1))
    conv2d = Conv2D(NF, (3, 3), activation="relu", padding="same")
    x = conv2d(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(NF, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2))(x)
    return inputs, x


'''
    Adds a encoder module (2 Covolutions, 2 BatchNormalizations, 1 MaxPooling) to the network
    input: x - current state of the module
           NF - number of filters used by the convolution
    output: x - state of the model after the new layers were added
'''


def addEncoderModule(x, NF):
    x = Conv2D(NF, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(NF, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    return x


'''
    Adds a decoder module (Concatenate, 2 Convolutions, 2 BatchNormalizations)
    input: c - result of one of the encoder modules
           u - result of oone of the decoder modules
           NF - number of filters used by the convolution 
    output: u - state of the model after the new layers were added
    c, u will be concatanated
'''


def addDecoderModule(c, u, NF):
    # x = concatenate([c,u],axis=0)
    # x = Concatenate([c,  u])

    x = Concatenate()([c, u])
    x = Conv2D(NF, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(NF, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    return x


'''
    Returs the upscaled result of a layer
    input: x - result of a layer
           NF - number of filters used by the Cov2DTranspose layer
    output: upscaled - upscaled result of that layer
'''


def upscaleResult(x, NF):
    x = UpSampling2D()(x)
    x = Conv2DTranspose(NF, (3, 3), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    return x


'''
    Builds the network
    output: inputs - the input layer
            x - the model
'''


def buildEncoderDecoder():
    '''
    Encoder
    '''
    inputs, c1 = getFirstModule()
    c2 = MaxPool2D((2, 2))(c1)
    c2 = addEncoderModule(c2, 2 * NF)
    c3 = MaxPool2D((2, 2))(c2)
    c3 = Dropout(0.3)(c3)
    c3 = addEncoderModule(c3, 4 * NF)
    c4 = MaxPool2D((2, 2))(c3)
    c4 = addEncoderModule(c4, 8 * NF)
    u1 = MaxPool2D((2, 2))(c4)
    u1 = addEncoderModule(u1, 16 * NF)
    '''
    Decoder
    '''
    u1 = Conv2DTranspose(8 * NF, (3, 3), activation='relu', padding="same")(u1)
    u1 = upscaleResult(u1, 8 * NF)
    u2 = addDecoderModule(c4, u1, 8 * NF)
    u2 = Conv2DTranspose(4 * NF, (3, 3), activation='relu', padding="same")(u2)
    u2 = upscaleResult(u2, 4 * NF)
    u3 = addDecoderModule(c3, u2, 4 * NF)
    u3 = Conv2DTranspose(2 * NF, (3, 3), activation='relu', padding="same")(u3)
    u3 = upscaleResult(u3, 2 * NF)
    u4 = addDecoderModule(c2, u3, 2 * NF)
    u4 = Dropout(0.3)(u4)
    u4 = Conv2DTranspose(NF, (3, 3), activation='relu', padding="same")(u4)
    u4 = upscaleResult(u4, NF)
    x = addDecoderModule(c1, u4, NF)
    x = UpSampling2D()(x)
    x = Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)
    # x = BatchNormalization()(x)
    # x = Conv2D(1, (1,1), activation='sigmoid',padding="same")(x)
    return inputs, x


def split_images():
    j = 0
    # finds the nii elements from folder
    epi_img = nib.load(FILENAME)
    epi_img_data = epi_img.get_fdata()
    shp = epi_img_data.shape
    # print(shp)
    for i in range(shp[2]):
        # epi_img_data=normalize(epi_img_data)
        slice_2 = epi_img_data[:, :, i]  # sagital
        # img_c =cv.normalize(src=slice_0, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        # img_s =cv.normalize(src=slice_2, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        # img_c=cv.resize(img_c,(320,320))
        # img_a = cv.resize(img_a, (320, 320))
        # img_s = cv.resize(img_s,(320, 320))
        # cv.imwrite('/img' + str(j) + '.jpeg', img_c)
        cv.imwrite('slice' + str(j) + '.jpeg', slice_2)
        j += 1
    return shp[2]


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def glue_slices():
    file_names = glob.glob('*O.jpeg')
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    vol = reader.Execute()
    sitk.WriteImage(vol, 'volume.nii')


def niiToObject(filename):
    nifty_img = nib.load(filename)
    nifty_data = nifty_img.get_fdata()
    # print(nifty_data.shape)

    nifty_data = np.moveaxis(nifty_data, 4, 0)

    nifty_data = nifty_data[0]

    nifty_data = np.moveaxis(nifty_data, 3, 0)
    nifty_data = nifty_data[0]

    verts, faces, normals, values = measure.marching_cubes_lewiner(nifty_data, 0)
    faces = faces + 1

    thefile = open('files/file.obj', 'w')
    for item in verts:
        thefile.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))

    for item in normals:
        thefile.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))

    for item in faces:
        thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0], item[1], item[2]))

    thefile.close()


def predict():
    noSlices = split_images()

    inputs, x = buildEncoderDecoder()
    model = Model(inputs=inputs, outputs=x, name="segmentation_model")

    model.compile(optimizer='adam', metrics=["accuracy", dice_coef], loss="binary_crossentropy")
    for i in range(noSlices):
        sliceFile = "slice" + str(i) + ".jpeg"
        predictSlice(model, sliceFile)
    glue_slices()

    niiToObject('volume.nii')
