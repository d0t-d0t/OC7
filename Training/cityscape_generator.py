import numpy as np
import keras
from pathlib import Path
from data_tools import get_image_n_mask, get_split_dic

class CityscapeDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_list, mask_dic, image_dir, masks_dir, batch_size=32, dim=(256,256), n_channels=3,
                 n_classes=8, shuffle=True,augmentation_func = None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.mask_dic = mask_dic
        self.image_dir = image_dir
        self.mask_dir = masks_dir
        self.image_list = image_list
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation_func
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        image_list_temp = [self.image_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(image_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.dim[0]*self.dim[1], self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(image_list_temp):
            temp_x,temp_y_unflat = get_image_n_mask(i, self.image_dir, self.mask_dir, image_list_temp, self.mask_dic,
                                             img_height= self.dim[0],img_width = self.dim[1],flatten_mask=False)
            

             
            # Apply data augmentation if provided
            if self.augmentation:
                # Apply same augmentation to both image and mask
                augmented = self.augmentation(image=temp_x, mask=temp_y_unflat)
                temp_x = augmented['image']
                temp_y_unflat = augmented['mask']

            # Flatten the mask after augmentation
            temp_y = np.resize(temp_y_unflat, (temp_x.shape[0]*temp_x.shape[1], self.n_classes))
            
            # Store sample
            X[i,] = temp_x

            # Store class
            y[i] = temp_y
        return X, y
    


def get_split_generator(params={},splits=['train','val'],data_root='Datas/reorganized_cityscape_data'):
    split_dic = get_split_dic(splits=splits,data_root=data_root)
    generators_dic = {}
    for split in splits:   
        split_images_dir = Path(data_root) / split / 'images'
        split_masks_dir = Path(data_root) / split / 'masks'
        split_data_dic = split_dic[split]
        image_list = split_data_dic['image_list'] 
        mask_dic = split_data_dic['label_dic']
        split_batch_generator = CityscapeDataGenerator(image_list,mask_dic,split_images_dir,split_masks_dir,**params)
        generators_dic[split]=split_batch_generator
    return generators_dic