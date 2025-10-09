from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path
import tensorflow as tf



cats = {'void': [0, 1, 2, 3, 4, 5, 6],
 'flat': [7, 8, 9, 10],
 'construction': [11, 12, 13, 14, 15, 16],
 'object': [17, 18, 19, 20],
 'nature': [21, 22],
 'sky': [23],
 'human': [24, 25],
 'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}

def get_split_dic(splits = ['train', 'val', 'test'],
                  data_root = 'Datas/reorganized_cityscape_data'):
    
    MASK_MASK = 'labelIds'
    split_dic = {}
    
    for split in splits:
        print(f'In {split}:\n' )
        images_dir = Path(data_root) / split / 'images'
        masks_dir = Path(data_root) / split / 'masks'
        # image_dir = 'Datas/images'
        # mask_dir = 'Datas/masks'
        image_list = os.listdir(images_dir)
        mask_list = [file for file in os.listdir(masks_dir) if MASK_MASK in file]
        image_list.sort()
        mask_list.sort()
        # create label_dic
        label_dic = {}
        for i,k in enumerate(image_list):
            label_dic[k] = mask_list[i]

        split_dic[split]={
            'image_list' : image_list,
            'label_dic' : label_dic
        }

        print(f'Number of images: {len(image_list)}\nNumber of masks: {len(mask_list)}')
        get_image_n_mask(0, images_dir, masks_dir, image_list, label_dic, display_img_n_info=True)

    return split_dic
# def get_mask(mask_list):
#     for m in tqdm.tqdm(tmask_list):
#         img = image.load_img(f'{train_dir}/{m}', grayscale=True, target_size=(512, 512))
#         img = np.squeeze(image.img_to_array(img))

def simp_8cat_mask(img, resize = True):
    mask = np.zeros((img.shape[0], img.shape[1], 8))
    img = np.squeeze(img)
    for i in range(-1, 34):
        if i in cats['void']:
            mask[:,:,0] = np.logical_or(mask[:,:,0],(img==i))
        elif i in cats['flat']:
            mask[:,:,1] = np.logical_or(mask[:,:,1],(img==i))
        elif i in cats['construction']:
            mask[:,:,2] = np.logical_or(mask[:,:,2],(img==i))
        elif i in cats['object']:
            mask[:,:,3] = np.logical_or(mask[:,:,3],(img==i))
        elif i in cats['nature']:
            mask[:,:,4] = np.logical_or(mask[:,:,4],(img==i))
        elif i in cats['sky']:
            mask[:,:,5] = np.logical_or(mask[:,:,5],(img==i))
        elif i in cats['human']:
            mask[:,:,6] = np.logical_or(mask[:,:,6],(img==i))
        elif i in cats['vehicle']:
            mask[:,:,7] = np.logical_or(mask[:,:,7],(img==i))
    if resize:
        mask = np.resize(mask,(img.shape[0]*img.shape[1], 8))
    return mask

# Category names for legend
category_names = ['void', 'flat', 'construction', 'object', 
                    'nature', 'sky', 'human', 'vehicle']
colors = [
    [128, 64, 128],    # void - purple
    [70, 70, 70],      # flat - dark gray  
    [190, 153, 153],   # construction - light brown
    [153, 153, 153],   # object - gray
    [107, 142, 35],    # nature - green
    [70, 130, 180],    # sky - blue
    [220, 20, 60],     # human - red
    [0, 0, 142]        # vehicle - dark blue
]

def get_mask_as_chart(mask,colors=colors):
    """
    Convert an 8-channel mask into a color-coded image.
    
    Args:
        mask: numpy array of shape (H, W, 8) where each channel represents a category
        
    Returns:
        colored_mask: RGB image of shape (H, W, 3) with each category colored differently
    """
    # Define colors for each of the 8 categories (RGB format)

    
    colors = np.array(colors, dtype=np.uint8)
    
    # Get the dominant class for each pixel (channel with highest value)
    # This assumes your mask contains probabilities or binary values
    class_mask = np.argmax(mask, axis=-1)
    
    # Create colored image
    H, W = class_mask.shape
    colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    for class_idx in range(8):
        mask_indices = class_mask == class_idx
        colored_mask[mask_indices] = colors[class_idx]
    
    return colored_mask

    
def get_image_n_mask(index, images_dir, masks_dir, image_list, label_dic,
                     display_img_n_info=False,
                     simplify_mask = True,
                     img_height = None,
                     img_width = None,
                     flatten_mask = False
                     ):
        id = image_list[index]
        mask_path = label_dic[id]

        
        target_size = None
        if img_height != None:
            target_size=(img_height, img_width)

        
        test_image = image.img_to_array(image.load_img(f'{images_dir}/{id}',target_size=target_size))/255.
        test_mask = image.img_to_array(image.load_img(f'{masks_dir}/{mask_path}',
                                                       color_mode='grayscale',
                                                       target_size=target_size,
                                                    ))
        test_simplified_mask = simp_8cat_mask(test_mask,resize=flatten_mask)

        if display_img_n_info :
            unique_values = np.unique(test_mask)
            print(f"Number of unique mask values: {len(unique_values)}")
            print(f"Unique values: {unique_values}")
            test_mask = np.squeeze(test_mask)


            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(1, 3, 1)
            ax.set_title('Image')
            ax.imshow(test_image)

            ax1 = fig.add_subplot(1, 3, 2)
            ax1.set_title('GT_Mask')
            ax1.imshow(test_mask)

            ax2 = fig.add_subplot(1, 3, 3)
            ax2.set_title('8cat_Mask')
            ax2.imshow(get_mask_as_chart(test_simplified_mask))

        if simplify_mask:            
            # test_simplified_mask = np.squeeze(test_simplified_mask)
            test_mask = test_simplified_mask
        return test_image,test_mask


def reorganize_dataset(images_root, masks_root, target_root):
    """
    Réorganise les dossiers d'images et de masks dans la structure train/val/test avec images/masks
    """
    splits = ['train', 'val', 'test']
    
    for split in splits:
        # Chemins sources
        images_split_path = Path(images_root) / split
        masks_split_path = Path(masks_root) / split
        
        # Chemins cibles
        target_split_path = Path(target_root) / split
        target_images_path = target_split_path / 'images'
        target_masks_path = target_split_path / 'masks'
        
        # Créer les dossiers cibles
        target_images_path.mkdir(parents=True, exist_ok=True)
        target_masks_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Traitement du split: {split}")
        
        # Traiter les images
        if images_split_path.exists():
            for city_folder in images_split_path.iterdir():
                if city_folder.is_dir():
                    for image_file in city_folder.glob('*.*'):
                        # if image_file.name.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                            # Nouveau nom avec préfixe de la ville pour éviter les conflits                        
                        # new_name = f"{city_folder.name}_{image_file.name}"
                        shutil.copy2(image_file, target_images_path / image_file.name)
                        print(f"  Image copiée: {image_file.name}")
        
        # Traiter les masks (utiliser labelID comme format de mask)
        if masks_split_path.exists():
            for city_folder in masks_split_path.iterdir():
                if city_folder.is_dir():
                    # labelID_folder = city_folder / 'labelID'
                    # if labelID_folder.exists():
                        for mask_file in city_folder.glob('*.*'):
                            # if mask_file.suffix.lower() in ['.png', '.tiff', '.bmp', '.tif']:
                                # Même nom que l'image correspondante
                                # new_name = f"{city_folder.name}_{mask_file.name}"
                                shutil.copy2(mask_file, target_masks_path / mask_file.name)
                                print(f"  Mask copié: {mask_file.name}")

def verify_structure(target_root):
    """
    Vérifie la structure finale des dossiers
    """
    target_path = Path(target_root)
    
    print("\nVérification de la structure:")
    for split in ['train', 'val', 'test']:
        split_path = target_path / split
        if split_path.exists():
            images_count = len(list((split_path / 'images').glob('*.*')))
            masks_count = len(list((split_path / 'masks').glob('*.*')))
            print(f"  {split}: {images_count} images, {masks_count} masks")
        else:
            print(f"  {split}: dossier manquant")

def get_prediction(model,image_raw):
    """RETURN a dic with:
        image_resized : the image at size model was fed
        array_image_resized : same but as array
        mask :  the predicted mask
        mask_array :  same but as array
        mask_resized :  this array resized at image raw size

    """
    result_dic = {}
    # if training_target_size is None:
    input_shape = model.input_shape
    training_target_size = (input_shape[1],input_shape[2])
    
    # image_resized = image.load_img(image_test_path,target_size=training_target_size)
    result_dic['image_resized'] = tf.image.resize(
        image_raw, training_target_size, method='bilinear'
    )
    #image.smart_resize(image_raw, size=training_target_size)

    
    result_dic['array_image_resized'] = image.img_to_array(result_dic['image_resized'])/255.


    # Add batch dimension to make it (1, 256, 256, 3)
    input_batch = np.expand_dims(result_dic['array_image_resized'], axis=0)

    mask = model.predict(input_batch)
    mask = mask[0]  
    # Reshape the mask from (65536, 8) to (256, 256, 8)
    result_dic['mask'] = mask.reshape(training_target_size[0], training_target_size[1], 8)

    result_dic['mask_array'] = get_mask_as_chart(result_dic['mask'])
    result_dic['mask_image'] = image.array_to_img(result_dic['mask_array'])

    # Resize mask to raw image dimensions for overlay
    array_image_raw = image.img_to_array(image_raw)/255.
    raw_height, raw_width = array_image_raw.shape[0], array_image_raw.shape[1]
    result_dic['mask_resized'] =  tf.image.resize(
        result_dic['mask_array'], (raw_height, raw_width), method='nearest'
    )
    
    # image.smart_resize(result_dic['mask_array'], (raw_height, raw_width), 
    #                                 interpolation='nearest')
    
    return result_dic


IMAGE_TEST_PATH = 'Datas/reorganized_cityscape_data/test/images'

def visualize_model_prediction(model,
                               image_test_path = IMAGE_TEST_PATH,
                               training_target_size = None,
                               ):
    
    image_raw = image.load_img(image_test_path)
    array_image_raw = image.img_to_array(image_raw)/255.

    prediction_dic = get_prediction(model,image_raw)


    fig = plt.figure(figsize=(15, 15))
    # ax = fig.add_subplot(1, 3, 1)
    # ax.set_title('Image')
    # ax.imshow(array_image_raw)

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.set_title('Image')
    ax1.imshow(prediction_dic['array_image_resized'])

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.set_title('Mask')
    ax2.imshow(prediction_dic['mask_array'])

    
    # Create overlayed image 
    alpha = 0.6
    overlay = (array_image_raw * (1 - alpha) + prediction_dic['mask_resized']/255.0 * alpha)

    ax3 = fig.add_subplot(1, 3, 1)
    ax3.set_title('Overlay (Mask on Original)')
    ax3.imshow(overlay)
    ax3.axis('off')

    return fig

import random


def visualize_multi_model_predictions(model,
                               directory_path=IMAGE_TEST_PATH,
                               n=5,
                               random_seed=42,
                               ):
    """
    Visualize model predictions for n random images from a directory.
    
    Args:
        model: The model to use for predictions
        directory_path: Path to directory containing images
        n: Number of random images to select (default=3)
        random_seed: Random seed for reproducibility (default=42)
    
    Returns:
        fig: matplotlib figure with predictions
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get all image files from directory
    all_images = []
    for file in os.listdir(directory_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            all_images.append(os.path.join(directory_path, file))
    
    if not all_images:
        raise ValueError(f"No images found in directory: {directory_path}")
    
    # Randomly select n images
    if len(all_images) < n:
        print(f"Warning: Only {len(all_images)} images found, using all available")
        selected_images = all_images
    else:
        selected_images = random.sample(all_images, n)
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(15, 5 * n))
    
    for i, image_path in enumerate(selected_images):
        try:
            # Load and process image
            image_raw = image.load_img(image_path)
            array_image_raw = image.img_to_array(image_raw) / 255.0
            
            # Get prediction
            prediction_dic = get_prediction(model, image_raw)
            
            # Create overlay
            alpha = 0.6
            overlay = (array_image_raw * (1 - alpha) + 
                      prediction_dic['mask_resized'] / 255.0 * alpha)
            
            # Plot 1: Overlay
            ax1 = fig.add_subplot(n, 3, i * 3 + 1)
            ax1.set_title(f'Overlay - Image {i+1}\n{os.path.basename(image_path)}')
            ax1.imshow(overlay)
            ax1.axis('off')
            
            # Plot 2: Resized image
            ax2 = fig.add_subplot(n, 3, i * 3 + 2)
            ax2.set_title(f'Resized Input - Image {i+1}')
            ax2.imshow(prediction_dic['array_image_resized'])
            ax2.axis('off')
            
            # Plot 3: Mask
            ax3 = fig.add_subplot(n, 3, i * 3 + 3)
            ax3.set_title(f'Prediction Mask - Image {i+1}')
            ax3.imshow(prediction_dic['mask_array'])
            ax3.axis('off')
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            # Create empty subplots with error message
            ax_err = fig.add_subplot(n, 3, i * 3 + 1)
            ax_err.set_title(f'Error - Image {i+1}')
            ax_err.text(0.5, 0.5, f'Error loading image:\n{str(e)}', 
                       ha='center', va='center', transform=ax_err.transAxes)
            ax_err.axis('off')
    
    plt.tight_layout()
    return fig