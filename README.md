# Wavelet_loss
#### Wavelet_loss  Function

## Requirements

This script requires:
- pytorch
- pytorch_wavelets
- torchvision
- PIL



If you don't already have pytorch or torchvision please have a look at https://pytorch.org/ as the installation command may vary depending on your OS and your version of CUDA.

You can install all other dependencies with pip by running terminal and get the result images by wavelet transform.Just like it.

### Just run 

```
python test.py 
```
## wavelet transform at every level
![img_grid1_0](https://user-images.githubusercontent.com/44399667/171095429-1631a23b-24a2-4a51-8238-fe135ec8203d.jpg)
![img_grid1_1](https://user-images.githubusercontent.com/44399667/171095445-6b4b3c50-7891-4fbc-8fce-56a019f5d095.jpg)
![img_grid1_2](https://user-images.githubusercontent.com/44399667/171095470-7557c737-dd47-4a4c-bc74-300032db7ce6.jpg)

And by change weight in every level, get some different results.
![effect](https://user-images.githubusercontent.com/44399667/171094805-cd4f92ac-5e87-40b1-b080-b16e97477b41.jpeg)

## Take a try for Autoencoder's training

![image](https://user-images.githubusercontent.com/44399667/171096283-82989122-298c-44ca-999c-446688f8d446.png)

### Reference
pytorch_wavelets: https://github.com/fbcotter/pytorch_wavelets

### LICENSE
This project is under MIT license.
    
