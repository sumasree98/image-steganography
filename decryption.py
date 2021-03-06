import cv2
import numpy as np 
from random import randint
import time

from encryption import output_img
from encryption import e_key 
from encryption import data_hiding_key

from encryption import total_len
from encryption import u
from encryption import no_blocks
from encryption import alpha
from encryption import huffman_table

start_time = time.time()

#extracting bits of output_img
output_img_bits = np.zeros((output_img.shape[0],output_img.shape[1],8),dtype=np.int)
for i in range(output_img.shape[0]):
    for j in range(output_img.shape[1]):
        for k in range(8):
            output_img_bits[i,j,k] = int((output_img[i,j]/(2**k))%2)

#extracting the embedded bits
embedded_bits = np.zeros((total_len),dtype=np.int)
def func_retrieve():
    index = 0
    for i in range(0,output_img.shape[0],2):
        for j in range(0,output_img.shape[1],2):
            if (output_img_bits[i,j,0]==1):
                for k in range(8):
                    embedded_bits[index] = output_img_bits[i,j+1,k]
                    index += 1
                    if(index==total_len):
                        return True
                for k in range(8):
                    embedded_bits[index] = output_img_bits[i+1,j,k]
                    index += 1
                    if(index==total_len):
                        return True
                for k in range(8):
                    embedded_bits[index] = output_img_bits[i+1,j+1,k]
                    index += 1
                    if(index==total_len):
                        return True
            else:
                for k in range(1,u):
                    embedded_bits[index] = output_img_bits[i,j,k]
                    index += 1
                    if(index==total_len):
                        return True
                for k in range(u):
                    embedded_bits[index] = output_img_bits[i,j+1,k]
                    index += 1
                    if(index==total_len):
                        return True
                for k in range(u):
                    embedded_bits[index] = output_img_bits[i+1,j,k]
                    index += 1
                    if(index==total_len):
                        return True
                for k in range(u):
                    embedded_bits[index] = output_img_bits[i+1,j+1,k]
                    index += 1
                    if(index==total_len):
                        return True
    return False

print func_retrieve()

#calculating the number of blocks above and below threshold
l_threshold = 0
m_threshold = 0
for i in range(0,output_img.shape[0],2):
    for j in range(0,output_img.shape[1],2):
        if (output_img_bits[i,j,0]==1):
            l_threshold += 1
        else:
            m_threshold += 1

#extracting the lengths of huffman codes
huffman_length_len = l_threshold*5
huffman_length_bits = np.zeros((l_threshold,5),dtype=np.int)
index = 0
for i in range(l_threshold):
    for j in range(5):
        huffman_length_bits[i,j] = embedded_bits[index]
        index += 1

huffman_len = 0
huffman_length = np.zeros((l_threshold),dtype=np.int)
for i in range(l_threshold):
    sum = 0
    for j in range(5):
        sum = sum + (huffman_length_bits[i,j]*(2**j))
    huffman_length[i] = sum
    huffman_len += sum

#extracting huffman codes
huffman = np.chararray((l_threshold),itemsize=50)
for i in range(l_threshold):
    huffman[i] = ""
    for j in range(huffman_length[i]):
        huffman[i] += str(embedded_bits[index])
        index += 1

#calculating the rlc pairs
rlc = np.zeros((l_threshold,3,2), dtype=np.int)
for i in range(l_threshold):
    temp = huffman[i]
    temp1 = ""
    index1 = 0
    while (temp!=""):
        temp1 += temp[:1]
        temp = temp[1:]
        for j in range(3):
            for k in range(11):
                if (j!=2 or k!=0):
                    if (j!=1 or k!=0):
                        if (temp1==huffman_table[j,k]):
                            temp1 = ""
                            rlc[i,index1,0] = j
                            rlc[i,index1,1] = k
                            index1 += 1
'''for i in range(5):
    print huffman[i]
    for j in range(3):
        print rlc[i,j]'''

#calculating the differences
diff = np.zeros((l_threshold,3),dtype=np.int)
for i in range(l_threshold):
    if (rlc[i,0,0]==0):
        diff[i,0] = rlc[i,0,1]
        if (rlc[i,1,0]==0):
            diff[i,1] = rlc[i,1,1]
            diff[i,2] = rlc[i,2,1]
        else:
            diff[i,1] = 0
            diff[i,2] = rlc[i,1,1]
    elif (rlc[i,0,0]==1):
        diff[i,0] = 0
        diff[i,1] = rlc[i,0,1]
        diff[i,2] = rlc[i,1,1]
    else:
        diff[i,0] = 0
        diff[i,1] = 0
        diff[i,2] = rlc[i,0,1]

rlc_sign_len = 0
for i in range(l_threshold):
    for j in range(3):
        if (diff[i,j]!=0):
            rlc_sign_len += 1
#print rlc_sign_len

#extracting the sign of every difference
rlc_sign = np.zeros((rlc_sign_len),dtype=np.int)
for i in range(rlc_sign_len):
    rlc_sign[i] = embedded_bits[index]
    index += 1

#extracting binary column vector
no_groups = m_threshold/no_blocks + 1
q = no_blocks*(4*u - 1)
p = q - alpha
binary_column_vector = np.zeros((p,no_groups),dtype=np.int)
for i in range(no_groups):
    for j in range(p):
        binary_column_vector[j,i] = embedded_bits[index]
        index += 1

#extracting the bitsream F
bitstream_f = np.zeros((output_img.shape[0]/2,output_img.shape[1]/2),dtype=np.int)
bitstream_f_len = (output_img.shape[0]*output_img.shape[1])/4
for i in range(output_img.shape[0]/2):
    for j in range(output_img.shape[1]/2):
        bitstream_f[i,j] = embedded_bits[index]
        index += 1

#replacing the first bits of the blocks with bitstream F
threshold_set = np.zeros((output_img.shape[0]/2,output_img.shape[1]/2),dtype=np.int)
for i in range(0,output_img.shape[0],2):
    for j in range(0,output_img.shape[1],2):
        threshold_set[i/2,j/2] = output_img_bits[i,j,0]
        output_img_bits[i,j,0] = bitstream_f[i/2,j/2]

for i in range(output_img.shape[0]):
    for j in range(output_img.shape[1]):
        sum = 0
        for k in range(8):
            sum = sum + (output_img_bits[i,j,k]*(2**k))
        output_img[i,j] = sum

final_img = np.zeros((output_img.shape[0],output_img.shape[1]),dtype=np.int)
for i in range(output_img.shape[0]):
    for j in range(output_img.shape[1]):
        final_img[i,j] = output_img[i,j]

#decoding the set 1 blocks of final image
index1 = 0
rlc_index = 0
for i in range(0,output_img.shape[0],2):
    for j in range(0,output_img.shape[1],2):
        if (threshold_set[i/2,j/2]==1):
            if (diff[index1,0]==0):
                final_img[i,j+1] = output_img[i,j]
            else:
                if (rlc_sign[rlc_index]==0):
                    final_img[i,j+1] = output_img[i,j] + diff[index1,0]
                else:
                    final_img[i,j+1] = output_img[i,j] - diff[index1,0]
                rlc_index += 1
            
            if (diff[index1,1]==0):
                final_img[i+1,j] = output_img[i,j]
            else:
                if (rlc_sign[rlc_index]==0):
                    final_img[i+1,j] = output_img[i,j] + diff[index1,0]
                else:
                    final_img[i+1,j] = output_img[i,j] - diff[index1,0]
                rlc_index += 1
            
            if (diff[index1,2]==0):
                final_img[i+1,j+1] = output_img[i,j]
            else:
                if (rlc_sign[rlc_index]==0):
                    final_img[i+1,j+1] = output_img[i,j] + diff[index1,0]
                else:
                    final_img[i+1,j+1] = output_img[i,j] - diff[index1,0]
                rlc_index += 1
            
            index1 += 1

final_img_bits = np.zeros((output_img.shape[0],output_img.shape[1],8),dtype=np.int)
for i in range(output_img.shape[0]):
    for j in range(output_img.shape[1]):
        for k in range(8):
            final_img_bits[i,j,k] = int((final_img[i,j]/(2**k))%2)

#decoding the set 2 blocks of final image
matrix_psi = np.zeros((p,q),dtype=np.int)
for i in range(p):
    for j in range (p):
        if (i==j):
            matrix_psi[i,j] = 1
        else:
            matrix_psi[i,j] = 0
index1 = 0
for i in range(p):
    for j in range(p,p+alpha):
        matrix_psi[i,j] = data_hiding_key[index1]
        index1 += 1

matrix_psi_i = matrix_psi.transpose()

column_vectors_g = np.zeros((no_groups,no_blocks,4,u),dtype=np.int)
for i in range(no_groups):
    for j in range(q):
        b_c = 0
        pixel_c = 0
        u1 = 0
        k=0
        while(True):
            if (pixel_c!=0 or u1!=0):
                column_vectors_g[i,b_c,pixel_c,u1] += matrix_psi_i[j,k]*binary_column_vector[k,i]
                k += 1
                if(k==p):
                    break
            u1 += 1
            if(u1 == u):
                pixel_c += 1
                u1 = 0
            if(pixel_c==4):
                b_c += 1
                pixel_c = 0

block_count = 0
group_count = 0
for i in range(0,output_img.shape[0],2):
    for j in range(0,output_img.shape[1],2):
        if (threshold_set[i/2,j/2]==0):
            for k in range(u):
                if (k!=0):
                     final_img_bits[i,j,k] = column_vectors_g[group_count,block_count,0,k]
                final_img_bits[i+1,j,k] = column_vectors_g[group_count,block_count,1,k]
                final_img_bits[i,j+1,k] = column_vectors_g[group_count,block_count,2,k]
                final_img_bits[i+1,j+1,k] = column_vectors_g[group_count,block_count,3,k]
            block_count += 1
            if (block_count == no_blocks):
                block_count = 0
                group_count += 1

#exracting the user's message
user_len = (total_len - index)/8
user_ascii_bin = np.zeros((user_len,8),dtype=np.int)
for i in range(user_len):
    for j in range(8):
        user_ascii_bin[i,j] = embedded_bits[index]
        index += 1
user_data = ""
for i in range(user_len):
    sum = 0
    for j in range(8):
        sum = sum + (user_ascii_bin[i,j]*(2**j))
    ch = chr(sum)
    user_data += ch

print "The hidden user data: ",user_data

'''for i in range(output_img.shape[0]):
    for j in range(output_img.shape[1]):
        sum = 0
        for k in range(8):
            sum = sum + (final_img_bits[i,j,k]*(2**k))
        final_img[i,j] = sum'''

#decrypting the image
divide = e_key*2
decr_img = np.zeros((output_img.shape[0],output_img.shape[1]),dtype=np.int)
for i in range(output_img.shape[0]/divide):
    for j in range(output_img.shape[1]/divide):
        for k in range(e_key):
            for l in range(e_key):
                decr_img[(divide*i)+k+e_key,(divide*j)+l+e_key] = final_img[(divide*i)+k,(divide*j)+l]
                decr_img[(divide*i)+k+e_key,(divide*j)+l] = final_img[(divide*i)+k,(divide*j)+l+e_key]
                decr_img[(divide*i)+k,(divide*j)+l+e_key] = final_img[(divide*i)+k+e_key,(divide*j)+l]
                decr_img[(divide*i)+k,(divide*j)+l] = final_img[(divide*i)+k+e_key,(divide*j)+l+e_key]

cv2.imwrite('decrypted_img.bmp',decr_img)

end_time = time.time()
print "The total time taken: %.2f" %round((end_time-start_time),2)