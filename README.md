# Wavelet_loss
#### Wavelet_loss  Function

Implementation of Wavelet Loss Function for Auto-Encoder (https://ieeexplore.ieee.org/document/9351990) in PyTorch

![image](https://user-images.githubusercontent.com/44399667/171102010-ad19b55c-834e-40d5-b6e6-7efd7c0b1b5c.png)





## Requirements

This script requires:
- pytorch
- pytorch_wavelets
- torchvision
- PIL



If you don't already have pytorch or torchvision please have a look at https://pytorch.org/ as the installation command may vary depending on your OS and your version of CUDA.

You can install all other dependencies with pip by running terminal and get the result images by wavelet transform.

### Just run 

```
python test.py 
```

## wavelet transform at every level (eg. 128*128).

![img_grid1_2](https://user-images.githubusercontent.com/44399667/171095470-7557c737-dd47-4a4c-bc74-300032db7ce6.jpg)

![img_grid1_1](https://user-images.githubusercontent.com/44399667/171095445-6b4b3c50-7891-4fbc-8fce-56a019f5d095.jpg)

![img_grid1_0](https://user-images.githubusercontent.com/44399667/171095429-1631a23b-24a2-4a51-8238-fe135ec8203d.jpg)


## And by changing weight in every level, you get some different results.

![image](https://user-images.githubusercontent.com/44399667/171097447-9e59b551-aca2-486b-a8cd-65fbb32438a4.png)

## Take a try for Autoencoder's training.


![image](https://user-images.githubusercontent.com/44399667/171097864-1c19dffc-ffed-4f00-a62e-c926bc46c23a.png)


### Reference
pytorch_wavelets : https://github.com/fbcotter/pytorch_wavelets



### LICENSE
This project is under MIT license.
    
