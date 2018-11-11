# Image Steganography
The word Steganography is derived from two Greek words- ‘stegos’ meaning ‘to cover’ and ‘grayfia’, meaning ‘writing’, thus translating to ‘covered writing’, or ‘hidden writing’.
Steganography is the practice of concealing information within an image.
It is one of the methods employed to protect secret or sensitive data from malicious attacks.

The advantage of steganography over cryptography is that the intended secret message does not attract attention to itself as an object of scrutiny. Plainly visible encrypted messages, no matter how unbreakable they are, arouse interest and may in be incriminating in countries in which encryption is illegal.

A novel reversible data hiding scheme for encrypted images based on the adaptive embedding mechanism is implemented. 
First, the image is divided into 2x2 non-overlapping blocks and is encrypted. Through the pre-processing of RLC incorporated with Huffman coding, all encrypted blocks can be categorized into two sets based on the encoded lengths of RLC. For the blocks with shorter encoded lengths of RLC, the pixel differences in the blocks are compressed by RLC; for the blocks with longer encoded lengths of RLC, matrix compression is implemented on the LSB layers. In this way, a large amount of space is obtained for embedding of user data bits. During decryption, with both the encryption key and the data-hiding key, in addition to direct decryption and data extraction, the encrypted blocks with difference compression can be recovered by RLC decoding, and the encrypted blocks with matrix compression can be recovered with the aid of coding length judgement and hash authentication.

The comparison of the output image is done against the original image using PSNR (Peak signal-to-noise ratio) and SSIM (Structural similarity) measures.

## Prerequisites and Installation
1. Python (v 2.7)
2. Numpy
```
pip install numpy
```
3. OpenCV
```
pip install opencv-python
```
4. Skimage (for testing)
```
pip install scikit-image
```

## Deployment
Store all the files in a single folder and run the decryption.py file. This file runs encryption.py and the encrypted image is stored as encrypted_img.bmp in the same folder. The final decrypted image is stored as decrypted_img.bmp in the folder and the user data is given as output along with the total time taken for execution. 

To change the image used for encryption, one can change the image in line 6 of encryption.py to any image in the dataset folder. One can also add new images to the dataset folder and use them for encryption (the image should be grayscale and of size 512x512 pixels). The user data can also be changed in line 254 of encryption.py.

To test the output image, the test.py file should be run after the execution of decryption.py. In test.py, the image in line 10 should be changed to the image used for encryption.

## Acknowledgement
This project is based on the research paper, 'Reversible data hiding in encrypted image with separable capability and high embedding capacity', by Chuan Qin, Zhihong He, Xiangyang Luo and Jing Dong.
