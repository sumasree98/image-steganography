import cv2
import numpy as np 
from random import randint

#Extracting the image and printing the pixels
img = cv2.imread("dataset/barbara_gray.bmp", 0)
#img = cv2.blur(img,(17,17))
'''for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        print img[i][j],
    print "\n"
print img.size'''

#storing pixels of image in an array
pixel = np.zeros((img.shape[0],img.shape[1],8),dtype=np.int)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(8):
            pixel[i,j,k] = int((img[i,j]/(2**k))%2)

#The encryption key
e_key = 16
divide = e_key*2

#Encrypting the image
encr_img = np.zeros((img.shape[0],img.shape[1]),dtype=np.int)
for i in range(img.shape[0]/divide):
    for j in range(img.shape[1]/divide):
        for k in range(e_key):
            for l in range(e_key):
                encr_img[(divide*i)+k+e_key,(divide*j)+l+e_key] = img[(divide*i)+k,(divide*j)+l]
                encr_img[(divide*i)+k+e_key,(divide*j)+l] = img[(divide*i)+k,(divide*j)+l+e_key]
                encr_img[(divide*i)+k,(divide*j)+l+e_key] = img[(divide*i)+k+e_key,(divide*j)+l]
                encr_img[(divide*i)+k,(divide*j)+l] = img[(divide*i)+k+e_key,(divide*j)+l+e_key]

#cv2.imwrite('encrypted_img.bmp',encr_img)

encr_pixel = np.zeros((img.shape[0],img.shape[1],8),dtype=np.int)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(8):
            encr_pixel[i,j,k] = int((encr_img[i,j]/(2**k))%2)

#difference between bits
diff = np.zeros((img.shape[0]/2,img.shape[1]/2,3),dtype=np.int)
for i in range(0,img.shape[0],2):
    for j in range(0,img.shape[1],2):
        diff[i/2,j/2,0] = encr_img[i,j+1] - encr_img[i,j]
        diff[i/2,j/2,1] = encr_img[i+1,j] - encr_img[i,j]
        diff[i/2,j/2,2] = encr_img[i+1,j+1] - encr_img[i,j]
'''for i in range(0,img.shape[0],2):
    for j in range(0,img.shape[1],2):
        for k in range(3):
            print diff[i/2,j/2,k],
        print ""'''

#RLC compression
rlc = np.zeros((img.shape[0]/2,img.shape[1]/2,3,2), dtype=np.int)
for i in range(img.shape[0]/2):
    for j in range(img.shape[1]/2):
        loopcount = 0
        for k in range(3):
            if (diff[i,j,k]!=0):
                count = 0
                index = k-1
                while (diff[i,j,index]==0 and index>=0):
                    count += 1
                    index -= 1
                rlc[i,j,loopcount,0]=count
                rlc[i,j,loopcount,1]=diff[i,j,k]
                loopcount += 1
'''for i in range(img.shape[0]/2):
    for j in range(img.shape[1]/2):
        for k in range(3):
            print rlc[i,j,k,0],"-",rlc[i,j,k,1],
        print ""'''

#huffman table
huffman_code_length = np.zeros((3,11),dtype=np.int)
huffman_table = np.chararray((3,11),itemsize=16)

huffman_code_length[0,0] = 4
huffman_code_length[0,1] = 2
huffman_code_length[0,2] = 2
huffman_code_length[0,3] = 3
huffman_code_length[0,4] = 4
huffman_code_length[0,5] = 5
huffman_code_length[0,6] = 7
huffman_code_length[0,7] = 8
huffman_code_length[0,8] = 10
huffman_code_length[0,9] = 16
huffman_code_length[0,10] = 16
huffman_code_length[1,1] = 4
huffman_code_length[1,2] = 5
huffman_code_length[1,3] = 7
huffman_code_length[1,4] = 9
huffman_code_length[1,5] = 11
huffman_code_length[1,6] = 16
huffman_code_length[1,7] = 16
huffman_code_length[1,8] = 16
huffman_code_length[1,9] = 16
huffman_code_length[1,10] = 16
huffman_code_length[2,1] = 5
huffman_code_length[2,2] = 8
huffman_code_length[2,3] = 10
huffman_code_length[2,4] = 12
huffman_code_length[2,5] = 16
huffman_code_length[2,6] = 16
huffman_code_length[2,7] = 16
huffman_code_length[2,8] = 16
huffman_code_length[2,9] = 16
huffman_code_length[2,10] = 16

huffman_table[0,0] = '1010'
huffman_table[0,1] = '00'
huffman_table[0,2] = '01'
huffman_table[0,3] = '100'
huffman_table[0,4] = '1011'
huffman_table[0,5] = '11010'
huffman_table[0,6] = '1111000'
huffman_table[0,7] = '11111000'
huffman_table[0,8] = '1111110110'
huffman_table[0,9] = '1111111110000010'
huffman_table[0,10] = '1111111110000011'
huffman_table[1,1] = '1100'
huffman_table[1,2] = '11011'
huffman_table[1,3] = '1111001'
huffman_table[1,4] = '111110110'
huffman_table[1,5] = '11111110110'
huffman_table[1,6] = '1111111110000100'
huffman_table[1,7] = '1111111110000101'
huffman_table[1,8] = '1111111110000110'
huffman_table[1,9] = '1111111110000111'
huffman_table[1,10] = '1111111110001000'
huffman_table[2,1] = '11100'
huffman_table[2,2] = '11111001'
huffman_table[2,3] = '1111110111'
huffman_table[2,4] = '111111110100'
huffman_table[2,5] = '1111111110001001'
huffman_table[2,6] = '1111111110001010'
huffman_table[2,7] = '1111111110001011'
huffman_table[2,8] = '1111111110001100'
huffman_table[2,9] = '1111111110001101'
huffman_table[2,10] = '1111111110001110'

#huffman coding
huffman = np.chararray((img.shape[0]/2,img.shape[1]/2),itemsize=50)
huffman_length = np.zeros((img.shape[0]/2,img.shape[1]/2),dtype=np.int)
threshold_set = np.zeros((img.shape[0]/2,img.shape[1]/2),dtype=np.int)
l_threshold = 0
m_threshold = 0
rlc_sign_len = 0
for i in range(img.shape[0]/2):
    for j in range(img.shape[1]/2):
        if (rlc[i,j,0,1]>10 or rlc[i,j,1,1]>10 or rlc[i,j,2,1]>10 or rlc[i,j,0,1]<-10 or rlc[i,j,1,1]<-10 or rlc[i,j,2,1]<-10):
            m_threshold += 1
            threshold_set[i,j] = 0
        else:
            huffman[i,j] = ""
            huffman_length[i,j] = 0
            index = 0
            count1 = 0
            while (index<=2 and rlc[i,j,index,1]!=0):
                huffman[i,j] += huffman_table[rlc[i,j,index,0],rlc[i,j,index,1]]
                huffman_length[i,j] += huffman_code_length[rlc[i,j,index,0],rlc[i,j,index,1]]
                index += 1
                count1 += 1
            if (huffman_length[i,j]>=24):
                m_threshold += 1
                threshold_set[i,j] = 0
            else:
                l_threshold += 1
                threshold_set[i,j] = 1
                rlc_sign_len += count1

rlc_sign = np.zeros((rlc_sign_len),dtype=np.int)
count = 0
for i in range(img.shape[0]/2):
    for j in range(img.shape[1]/2):
        if (threshold_set[i,j] == 1):
            index = 0
            while (index<=2 and rlc[i,j,index,1]!=0):
                if (rlc[i,j,index,1]<0):
                    rlc_sign[count] = 1
                else:
                    rlc_sign[count] = 0
                count += 1
                index += 1

#collecting bit stream F and storing location map in LSB
bitstream_f = np.zeros((img.shape[0]/2,img.shape[1]/2),dtype=np.int)
for i in range(0,img.shape[0],2):
    for j in range(0,img.shape[1],2):
        bitstream_f[i/2,j/2] = encr_pixel[i,j,0]
        encr_pixel[i,j,0] = threshold_set[i/2,j/2]

#compressing LSB of blocks belonging to set 2
u = 2
no_blocks = 8
no_groups = m_threshold/no_blocks + 1
column_vectors_g = np.zeros((no_groups,no_blocks,4,u),dtype=np.int)
block_count = 0
group_count = 0
for i in range(0,img.shape[0],2):
    for j in range(0,img.shape[1],2):
        if (encr_pixel[i,j,0]==0):
            for k in range(u):
                if (k!=0):
                    column_vectors_g[group_count,block_count,0,k] = encr_pixel[i,j,k]
                column_vectors_g[group_count,block_count,1,k] = encr_pixel[i+1,j,k]
                column_vectors_g[group_count,block_count,2,k] = encr_pixel[i,j+1,k]
                column_vectors_g[group_count,block_count,3,k] = encr_pixel[i+1,j+1,k]
            block_count += 1
            if (block_count == no_blocks):
                block_count = 0
                group_count += 1

alpha = 16
q = no_blocks*(4*u - 1)
p = q - alpha
data_hiding_key = np.zeros((p*alpha),dtype=np.int)
for i in range(p*alpha):
    data_hiding_key[i] = randint(0,1)

matrix_psi = np.zeros((p,q),dtype=np.int)
for i in range(p):
    for j in range (p):
        if (i==j):
            matrix_psi[i,j] = 1
        else:
            matrix_psi[i,j] = 0
index = 0
for i in range(p):
    for j in range(p,p+alpha):
        matrix_psi[i,j] = data_hiding_key[index]
        index += 1

binary_column_vector = np.zeros((p,no_groups),dtype=np.int)
for i in range(no_groups):
    for j in range(p):
        b_c = 0
        pixel_c = 0
        u1 = 0
        k=0
        while(True):
            if (pixel_c!=0 or u1!=0):
                binary_column_vector[j,i] += matrix_psi[j,k]*column_vectors_g[i,b_c,pixel_c,u1]
                k += 1
                if(k==q):
                    break
            u1 += 1
            if(u1 == u):
                pixel_c += 1
                u1 = 0
            if(pixel_c==4):
                b_c += 1
                pixel_c = 0
        binary_column_vector[j,i] = binary_column_vector[j,i]%2

#user data
user_data = "Hello World!"
ascii_bin = np.zeros((len(user_data),8),dtype=np.int)
index = 0
for ch in user_data:
    ascii_v = ord(ch)
    for i in range(8):
        ascii_bin[index,i] = int((ascii_v/(2**i))%2)
    index += 1

#concatenating all the bits into one stream
huffman_length_bits = np.zeros((l_threshold,5),dtype=np.int)
index = 0
huffman_len = 0
for i in range(0,img.shape[0],2):
    for j in range(0,img.shape[1],2):
        if(encr_pixel[i,j,0]==1):
            for k in range(5):
                huffman_length_bits[index,k] = int((huffman_length[i/2,j/2]/(2**k))%2)
            index += 1
            huffman_len += huffman_length[i/2,j/2]
huffman_length_len = l_threshold*5
bcv_len = p*no_groups
bitstream_f_len = (img.shape[0]*img.shape[1])/4
user_data_len = len(user_data)*8

total_len = huffman_length_len + huffman_len + rlc_sign_len + bcv_len + bitstream_f_len + user_data_len

embedding_bits = np.zeros((total_len),dtype=np.int)
index = 0
for i in range(l_threshold):
    for j in range(5):
        embedding_bits[index] = huffman_length_bits[i,j]
        index += 1
for i in range(0,img.shape[0],2):
    for j in range(0,img.shape[1],2):
        if(encr_pixel[i,j,0]==1):
            for k in range (huffman_length[i/2,j/2]):
                embedding_bits[index] = int(huffman[i/2,j/2][k])
                index += 1
for i in range(rlc_sign_len):
    embedding_bits[index] = rlc_sign[i]
    index += 1
for i in range(no_groups):
    for j in range(p):
        embedding_bits[index] = binary_column_vector[j,i]
        index += 1
for i in range(img.shape[0]/2):
    for j in range(img.shape[1]/2):
        embedding_bits[index] = bitstream_f[i,j]
        index += 1
for i in range(len(user_data)):
    for j in range(8):
        embedding_bits[index] = ascii_bin[i,j]
        index += 1

#embedding the bits into the image
output_img_bits = np.zeros((img.shape[0],img.shape[1],8),dtype=np.int)
for i in range(0,img.shape[0],2):
    for j in range(0,img.shape[1],2):
        output_img_bits[i,j,0] = encr_pixel[i,j,0]
        if (encr_pixel[i,j,0]==1):
            for k in range(1,8):
                output_img_bits[i,j,k] = encr_pixel[i,j,k]
        else:
            for k in range(u,8):
                output_img_bits[i,j,k] = encr_pixel[i,j,k]
                output_img_bits[i,j+1,k] = encr_pixel[i,j+1,k]
                output_img_bits[i+1,j,k] = encr_pixel[i+1,j,k]
                output_img_bits[i+1,j+1,k] = encr_pixel[i+1,j+1,k]

def func_embed():
    index = 0
    for i in range(0,img.shape[0],2):
        for j in range(0,img.shape[1],2):
            if(encr_pixel[i,j,0]==1):
                for k in range(8):
                    output_img_bits[i,j+1,k] = embedding_bits[index]
                    index += 1
                    if(index==total_len):
                        return True
                for k in range(8):
                    output_img_bits[i+1,j,k] = embedding_bits[index]
                    index += 1
                    if(index==total_len):
                        return True
                for k in range(8):
                    output_img_bits[i+1,j+1,k] = embedding_bits[index]
                    index += 1
                    if(index==total_len):
                        return True
            else:
                for k in range(1,u):
                    output_img_bits[i,j,k] = embedding_bits[index]
                    index += 1
                    if(index==total_len):
                        return True
                for k in range(u):
                    output_img_bits[i,j+1,k] = embedding_bits[index]
                    index += 1
                    if(index==total_len):
                        return True
                for k in range(u):
                    output_img_bits[i+1,j,k] = embedding_bits[index]
                    index += 1
                    if(index==total_len):
                        return True
                for k in range(u):
                    output_img_bits[i+1,j+1,k] = embedding_bits[index]
                    index += 1
                    if(index==total_len):
                        return True
    return False

print func_embed()

#converting binary pixels to decimal
output_img = np.zeros((img.shape[0],img.shape[1]),dtype=np.int)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        sum = 0
        for k in range(8):
            sum = sum + (output_img_bits[i,j,k]*(2**k))
        output_img[i,j] = sum

cv2.imwrite('final_img.bmp',output_img)