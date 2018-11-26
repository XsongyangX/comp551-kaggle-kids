
# coding: utf-8

# In[1]:


import numpy as np
import math
import os
from scipy import ndimage
from skimage import transform


# # Preprocessing

# In[2]:


def process(images, filename):
    """
    Pre-process the images in the data set to a list of 900-dimensional array. Each image has no more noise and centers
    the drawing. In addition, all images are scaled to fit the 30x30 frame.
    
    The new images are stored in a file.
    
    Takes
        images: a numpy array from numpy.load, the array comes from the original data set
        
        filename: a path string, where the new data set is stored (must be a .npy file)
    
    """
    processed_images = []
    for im in images:
        # reshape vector to correct size
        pim = im[1].reshape(100,100)

        # locates components (connected portions) of image
        # creates binary image of components, label_im
        mask = pim > pim.mean()
        label_im, nb_labels = ndimage.label(mask)
        sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
        
        # locates largest component of image
        mask_size = sizes < max(sizes)
        remove_pixel = mask_size[label_im]

        # in both the original image and the binary image, remove pixels that are
        # not in the largest component (reduces noise)
        label_im[remove_pixel] = 0
        pim[remove_pixel] = 0

        # crop image around largest component
        labels = np.unique(label_im)
        label_im = np.searchsorted(labels, label_im)
        slice_x, slice_y = ndimage.find_objects(label_im==1)[0]
        
        # transforms/scales largest component to size 30x30. This will stretch and add antialiasing 
        # to the image to accomodate.
        # Can choose to use original image (pim) or binary image (label_im), but I find transforming
        # the binary image loses a lot of data, and the end result isn't binary anyways
        roi = transform.resize(pim[slice_x, slice_y],(30,30))
        roi = roi.reshape(1,900) #represent image as vector
        processed_images.append(roi[0])
    # Saves all preprocessed images in file
    # file is loaded the same way as the others (using numpy.load)
    np.save(filename,np.array(processed_images))

def process_pad(images, filename):
    """
    Pre-process the images in the data set to a list of 1024-dimensional array. Each image has no more noise and centers
    the drawing. In contrast to the previous pre-processing method, this one pads white space around smaller
    pictures instead of scaling.
    
    The new images are stored in a file.
    
    Takes
        images: a numpy array from numpy.load, the array comes from the original data set
        
        filename: a path string, where the new data set is stored (must be a .npy file)
    
    """
    processed_images = []
    for im in images:
        # reshape vector to correct size
        pim = im[1].reshape(100,100)

        # locates components (connected portions) of image
        # creates binary image of components, label_im
        mask = pim > pim.mean()
        label_im, nb_labels = ndimage.label(mask)
        sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
        
        # locates largest component of image
        mask_size = sizes < max(sizes)
        remove_pixel = mask_size[label_im]

        # in both the original image and the binary image, remove pixels that are
        # not in the largest component (reduces noise)
        label_im[remove_pixel] = 0
        pim[remove_pixel] = 0

        # crop image around largest component
        labels = np.unique(label_im)
        label_im = np.searchsorted(labels, label_im)
        slice_x, slice_y = ndimage.find_objects(label_im==1)[0]
        
        if max(sizes) < 50:
            pim[:] = 0
        
        final = pim[slice_x, slice_y]
        if len(final) > 32:
            final = transform.resize(final,(32,len(final[0])))
        if len(final[0]) > 32:
            final = transform.resize(final,(len(final),32))
        
        # pads image to size 32x32
        pad_x = 32 - len(final[0])
        pad_x_l = max([0,pad_x // 2])
        pad_x_r = max([0,math.ceil(pad_x / 2)])
        
        pad_y = 32 - len(final)
        pad_y_t = max([0,pad_y // 2])
        pad_y_b = max([0,math.ceil(pad_y / 2)])
            
        final = np.pad(final,((pad_y_t,pad_y_b),(pad_x_l,pad_x_r)),'constant')
        # transforms/scales largest component to size 30x30. This will stretch and add antialiasing 
        # to the image to accomodate.
        # Can choose to use original image (pim) or binary image (label_im), but I find transforming
        # the binary image loses a lot of data, and the end result isn't binary anyways
        roi = final.reshape(1,1024) #represent image as vector
        processed_images.append(roi[0])
    # Saves all preprocessed images in file
    # file is loaded the same way as the others (using numpy.load)
    np.save(filename,np.array(processed_images))
     

def generateIntegerData(path, destination):
    """
    Generates the integer version of the scaled data.
    
    Takes
        path: a path string, where the .npy is stored
        destination: a path string, where the new data set is stored (must have .npy extension)
    """
    
    data = np.load(path, encoding='latin1')
    
    newData = []
    
    for entry in data:
        
        newEntry = [int(k) for k in entry]
        
        newData.append(newEntry)
        
    newData = np.array(newData)
    np.save(destination, newData)

def generateFatBinary(path, destination):
    """
    Turn an integer version of the data set into a binary data set.
    Here is the rule of conversion.
        For each element in an image vector:
            element = 0 if 0,
                    = 1 otherwise.
                    
    It is called 'fat' because the lines of the drawing are drastically thickened. 
    
    Takes
        path: a path string, where the .npy integer data set is stored
        destination: a path string, where the new .npy file is stored
    """
    
    data = np.load(path, encoding='latin1')
    
    newData = []
    
    def binary(integer):
        if integer == 0:
            return 0
        else:
            return 1
    
    for entry in data:
        
        newEntry = [binary(k) for k in entry]
        
        newData.append(newEntry)

    newData = np.array(newData)
    np.save(destination, newData)


# # Loading and writing files

# In[3]:


def ensure(path):
    """
    Checks if the specified path exists
    
    Takes
        path: a path string
        
    Return 
        True: path exists
        False: otherwise
    """
    
    return os.path.exists(path)

def checkOriginal() :
    """
    Checks if the original data set is correctly placed
    
    Return
        True: file hierarchy satisfied
        False: otherwise
    """
    
    if not ensure("../input/f2018-hand-drawn-pictures/train_images.npy")     or not ensure("../input/f2018-hand-drawn-pictures/test_images.npy")     or not ensure("../input/f2018-hand-drawn-pictures/train_labels.csv"):
        return False
    
    return True

def generate():
    """
    Generates the pre-processed data sets
    """
    
    if checkOriginal() == False:
        print("Please put 'train_labels.csv' and the unzipped 'train_images.npy' and 'test_images.npy' in the "               + "'input/f2018-hand-drawn-pictures' folder"               + " such that this script can access them via '../input/f2018-hand-drawn-pictures/'")
        raise FileNotFoundError("Missing original data set in correct places")
    
    # get the original data set
    train_images = np.load("../input/f2018-hand-drawn-pictures/train_images.npy", encoding='latin1')
    test_images = np.load("../input/f2018-hand-drawn-pictures/test_images.npy", encoding='latin1')
    
    # generate real valued 900-dimensional vectors
    realVectors = "../input/preprocessed-hand-drawn-pics/"
    os.makedirs(realVectors, exist_ok=True)
    process(train_images, realVectors + "proc_train_images.npy")
    process(test_images, realVectors + "proc_test_images.npy")
    
    # generate integer valued 900-dimensional vectors
    intVectors = "../input/preprocessed-hand-drawn-pics-int/"
    os.makedirs(intVectors, exist_ok=True)
    generateIntegerData(realVectors + "proc_train_images.npy", intVectors + "proc_train_images_int.npy")
    generateIntegerData(realVectors + "proc_test_images.npy", intVectors + "proc_test_images_int.npy")
    
    # generate binary valued 900-dimensional vectors
    binVectors = "../input/preprocessed-hand-drawn-pics-fat-bin/"
    os.makedirs(binVectors, exist_ok=True)
    generateFatBinary(intVectors + "proc_train_images_int.npy", binVectors + "proc_train_images_fat_binary.npy")
    generateFatBinary(intVectors + "proc_test_images_int.npy", binVectors + "proc_test_images_fat_binary.npy")
    
    # generate padded 1024-dimensional vectors
    padVectors = "../input/preprocessed-hand-drawn-pics-pad/"
    os.makedirs(padVectors, exist_ok=True)
    process_pad(train_images, padVectors + "proc_train_images_pad.npy")
    process_pad(test_images, padVectors + "proc_test_images_pad.npy")
    
    return

generate()

