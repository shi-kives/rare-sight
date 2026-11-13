import albumentations as A
from albumentations.pytorch import ToTensorV2

# standard ImageNet RGB statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_test_transform(): # for val and test set (sensitive to variations)

    return A.Compose([
        A.Resize(224,224),
        A.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD),
        ToTensorV2() # numpy (H, W, 3) -> tensor (3, H, W)
    ])

def get_path_train_transform():

    return A.Compose([
        A.RandomResizedCrop(size = (224, 224), scale = (0.7, 1.0)), # select a random crop size from 70 - 100% and resize to (224, 224)
        A.HorizontalFlip(p = 0.5), # 50% probability of flipping left and right
        A.VerticalFlip(p = 0.5), # 50% probability of flipping top and bottom
        A.RandomRotate90(p = 0.5), # 50% probability of rotating the image by a multiple of 90deg
        A.HueSaturationValue(hue_shift_limit = 10, sat_shift_limit = 20, val_shift_limit = 15), # change color, intensity & brightness
        # A.ElasticTransform(alpha = 90, sigma = 6, p = 0.2), # 20% probability of spatial distortion
        # A.GaussNoise(std_range=(0.2, 0.4), p = 0.2), # 20% probability of adding gaussian noise.
        A.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD),
        ToTensorV2()
    ])

def get_transform(modality, mode):
    if mode in ['val', 'test']:
        return get_test_transform()

    if mode != "train":
        raise ValueError(f"Unknown mode: {mode}")
    
    available_modalities = {
        'path': get_path_train_transform,
    }

    if modality not in available_modalities:
        raise ValueError(f"unknown modality {modality}. currently available: {available_modalities}")

    return available_modalities[modality]()
