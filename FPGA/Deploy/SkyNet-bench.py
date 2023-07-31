
#################### import all libraries and initializations ############

import sys
import numpy as np 
import os
import time
import math
from PIL import Image
import cv2
from datetime import datetime
from pynq import allocate
from pynq import Overlay
import struct
from multiprocessing import Process, Pipe, Queue, Event, Manager
import pdb


print('\n**** Running SkyNet')

# xlnk = Xlnk()
# xlnk.xlnk_reset()

################### Download the overlay
overlay = Overlay("./SkyNet.bit")
print("Bitstream loaded")



########## Allocate memory for weights and off-chip buffers
os.system("cat /proc/meminfo | grep 'MemFree'")

mytype = 'B,'*63 + 'B'
dt = np.dtype(mytype)
img = allocate(shape=(3,162*2,322*2), dtype=np.uint8)


conv_weight_1x1_all = allocate(shape=(413, 32), dtype=dt)
conv_weight_3x3_all = allocate(shape=(64, 3, 3), dtype=dt)
bias_all = allocate(shape=(106), dtype=dt)
DDR_pool_3_out = allocate(shape=(2, 164, 324), dtype=dt)
DDR_pool_6_out = allocate(shape=(3, 84, 164), dtype=dt)
DDR_buf = allocate(shape=(128, 44, 84), dtype=dt)
predict_boxes = allocate(shape=(4, 5), dtype=np.float32)
constant = allocate(shape=(4, 3), dtype=np.int32)

os.system("cat /proc/meminfo | grep 'MemFree'")
print("Allocating memory done")


########### Load parameters from SD card to DDR
params = np.fromfile("./weights_fixed.bin", dtype=dt)
idx = 0
np.copyto(conv_weight_1x1_all, params[idx:idx+conv_weight_1x1_all.size].reshape(conv_weight_1x1_all.shape))
idx += conv_weight_1x1_all.size
np.copyto(conv_weight_3x3_all, params[idx:idx+conv_weight_3x3_all.size].reshape(conv_weight_3x3_all.shape))
idx += conv_weight_3x3_all.size
np.copyto(bias_all, params[idx:idx+bias_all.size].reshape(bias_all.shape))
print("Parameters loading done")





################## Utility functions 
IMG_DIR = './test_images/'
# Get image name list
def get_image_names():
    names_temp = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    names_temp.sort(key= lambda x:int(x[:-4]))
    return names_temp

# Process the images in batches, may help when write to XML
BATCH_SIZE = 4
def get_image_batch():
    image_list = get_image_names()
    batches = list()
    for i in range(0, len(image_list), BATCH_SIZE):
        batches.append(image_list[i:i+BATCH_SIZE])
    return batches

def stitch(image_queue, name_queue):
    blank = Image.new('RGB', (644, 324), (127, 127, 127))
    img = np.ndarray(shape=(3,162*2,322*2), dtype=np.uint8)
    
    for batch in get_image_batch():
        for i in range(0, len(batch), 4):
            while image_queue.full():
                continue
            
            pic_name = IMG_DIR + batch[0]
            image = Image.open(pic_name).convert('RGB')
            image = image.resize((320, 160))
            blank.paste(image, (1, 1))

            pic_name = IMG_DIR + batch[1]
            image = Image.open(pic_name).convert('RGB')
            image = image.resize((320, 160))
            blank.paste(image, (323, 1))

            pic_name = IMG_DIR + batch[2]
            image = Image.open(pic_name).convert('RGB')
            image = image.resize((320, 160))
            blank.paste(image, (1, 163))

            pic_name = IMG_DIR + batch[3]
            image = Image.open(pic_name).convert('RGB')
            image = image.resize((320, 160))
            blank.paste(image, (323, 163))

            image_stitched = np.transpose(blank, (2, 0, 1))
            image_queue.put(image_stitched)
            
def stitch_static(batch):
    #pdb.set_trace()
    blank = Image.new('RGB', (644, 324), (127, 127, 127))
    pic_name = IMG_DIR + batch[0]
    image = Image.open(pic_name).convert('RGB')
    image = image.resize((320, 160))
    blank.paste(image, (1, 1))

    pic_name = IMG_DIR + batch[1]
    image = Image.open(pic_name).convert('RGB')
    image = image.resize((320, 160))
    blank.paste(image, (323, 1))

    pic_name = IMG_DIR + batch[2]
    image = Image.open(pic_name).convert('RGB')
    image = image.resize((320, 160))
    blank.paste(image, (1, 163))

    pic_name = IMG_DIR + batch[3]
    image = Image.open(pic_name).convert('RGB')
    image = image.resize((320, 160))
    blank.paste(image, (323, 163))

    image_stitched = np.transpose(blank, (2, 0, 1))
    return image_stitched



def compute_bounding_box(boxes, output_queue):
    predict_boxes = np.empty([4, 5], dtype=np.float32)
    constant = np.empty([4, 3], dtype=np.int32)
    
    for batch in get_image_batch():
        print(batch)
        for i in range(0, len(batch), 4):
            
            while output_queue.full():
                continue
                
            outputs = output_queue.get()
            outputs_boxes = outputs[0]
            outputs_index = outputs[1]
            np.copyto(predict_boxes, np.array(outputs_boxes))
            np.copyto(constant, np.array(outputs_index))
                
            for idx in range(0, 4):
                predict_boxes[idx][0] = 1.0 / (1.0 + math.exp(-predict_boxes[idx][0])) + constant[idx][1]
                predict_boxes[idx][1] = 1.0 / (1.0 + math.exp(-predict_boxes[idx][1])) + constant[idx][2]

                if( constant[idx][0] == 0 ):
                    predict_boxes[idx][2] = math.exp(predict_boxes[idx][2]) * box[0]
                    predict_boxes[idx][3] = math.exp(predict_boxes[idx][3]) * box[1]
                else:
                    predict_boxes[idx][2] = math.exp(predict_boxes[idx][2]) * box[2]
                    predict_boxes[idx][3] = math.exp(predict_boxes[idx][3]) * box[3]
                predict_boxes[idx][4] = 1.0 / (1.0 + math.exp(-predict_boxes[idx][4]))

                predict_boxes[idx][0] = predict_boxes[idx][0] / 40
                predict_boxes[idx][1] = predict_boxes[idx][1] / 20
                predict_boxes[idx][2] = predict_boxes[idx][2] / 40
                predict_boxes[idx][3] = predict_boxes[idx][3] / 20
                #print(predict_boxes[idx])

                x1 = int(round((predict_boxes[idx][0] - predict_boxes[idx][2]/2.0) * 640))
                y1 = int(round((predict_boxes[idx][1] - predict_boxes[idx][3]/2.0) * 360))
                x2 = int(round((predict_boxes[idx][0] + predict_boxes[idx][2]/2.0) * 640))
                y2 = int(round((predict_boxes[idx][1] + predict_boxes[idx][3]/2.0) * 360))
                result_rectangle.append([x1, x2, y1, y2])

                print([x1, x2, y1, y2])

                
            
def compute_bounding_box_static(outputs_boxes, outputs_index, batch):
    #pdb.set_trace()
    np.copyto(predict_boxes, np.array(outputs_boxes))
    np.copyto(constant, np.array(outputs_index))
        
    for idx in range(0, 4):
        predict_boxes[idx][0] = 1.0 / (1.0 + math.exp(-predict_boxes[idx][0])) + constant[idx][1]
        predict_boxes[idx][1] = 1.0 / (1.0 + math.exp(-predict_boxes[idx][1])) + constant[idx][2]

        if( constant[idx][0] == 0 ):
            predict_boxes[idx][2] = math.exp(predict_boxes[idx][2]) * box[0]
            predict_boxes[idx][3] = math.exp(predict_boxes[idx][3]) * box[1]
        else:
            predict_boxes[idx][2] = math.exp(predict_boxes[idx][2]) * box[2]
            predict_boxes[idx][3] = math.exp(predict_boxes[idx][3]) * box[3]
        predict_boxes[idx][4] = 1.0 / (1.0 + math.exp(-predict_boxes[idx][4]))

        predict_boxes[idx][0] = predict_boxes[idx][0] / 40
        predict_boxes[idx][1] = predict_boxes[idx][1] / 20
        predict_boxes[idx][2] = predict_boxes[idx][2] / 40
        predict_boxes[idx][3] = predict_boxes[idx][3] / 20
        #print(predict_boxes[idx])

        x1 = int(round((predict_boxes[idx][0] - predict_boxes[idx][2]/2.0) * 640))
        y1 = int(round((predict_boxes[idx][1] - predict_boxes[idx][3]/2.0) * 360))
        x2 = int(round((predict_boxes[idx][0] + predict_boxes[idx][2]/2.0) * 640))
        y2 = int(round((predict_boxes[idx][1] + predict_boxes[idx][3]/2.0) * 360))
        result_rectangle.append([x1, x2, y1, y2])

        print([x1, x2, y1, y2])

        # raw_img = cv2.imread(IMG_DIR + batch[idx])
        # cv2.imshow('my webcam', cv2.rectangle(raw_img, (x1, y1), (x2, y2), (255, 0, 0), 2))
        # cv2.waitKey(0)

###########################################################
################ MAIN PART OF DETECTION ###################
###########################################################

SkyNet = overlay.SkyNet_0

SkyNet.write(0x10, img.physical_address)
SkyNet.write(0x1c, conv_weight_1x1_all.physical_address)
SkyNet.write(0x28, conv_weight_3x3_all.physical_address)
SkyNet.write(0x34, bias_all.physical_address)
SkyNet.write(0x40, DDR_pool_3_out.physical_address)
SkyNet.write(0x4c, DDR_pool_6_out.physical_address)
SkyNet.write(0x58, DDR_buf.physical_address)
SkyNet.write(0x70, predict_boxes.physical_address)
SkyNet.write(0x7c, constant.physical_address)

# print("full address of img: ", img.physical_address)
# print("full address of img 64 bit: ", img.physical_address >> 32)
# print("the one stored in register 0: ", SkyNet.read(0x10))
# print("the one stored in register 1: ", SkyNet.read(0x14))

# print("full address of conv_1x1: ", conv_weight_1x1_all.physical_address)
# print("the one stored in register: ", SkyNet.read(0x1c))
# print("the one stored in register: ", SkyNet.read(0x20))

# rails = pynq.get_rails()
# recorder = pynq.DataRecorder(rails['power1'].power)

box = [1.4940052559648322, 2.3598481287086823, 4.0113013115312155, 5.760873975661669]

################# Declare New Process ##############
image_queue = Queue(200) ## could be smaller
name_queue = Queue(200)
output_queue = Queue(10)
mgr = Manager()
result_rectangle = mgr.list()
# p1 = Process(target=stitch, args=(image_queue, name_queue))
# p2 = Process(target=compute_bounding_box, args=(result_rectangle, output_queue))

################### Start to detect ################
output_boxes = np.empty([4, 5], dtype=np.float32)
output_index = np.empty([4, 3], dtype=np.int32)

# p1.start()
# p2.start()

num_img = len(get_image_batch()) * BATCH_SIZE
print(num_img)


print("\n**** Start to detect")
start = time.time()

for batch in get_image_batch():
    preprocessed_img = stitch_static(batch)
    for i in range(0, len(batch), 4):
    
        # while image_queue.empty():
        #     continue
            
        # preprocessed_img = image_queue.get()
        #print("raw input to FPGA: \n", preprocessed_img)
        np.copyto(img, np.array(preprocessed_img))

        #print("control reg: ", SkyNet.read(0x00))

        SkyNet.write(0x00, 1)
        #print("control reg after start signal: ", SkyNet.read(0x00))
        isdone = SkyNet.read(0x00) & 0b10
        while( isdone != 0b10 ):
            #print("FPGA is still running")
            isdone = SkyNet.read(0x00) & 0b10
            

        outputs = []
        #print("raw output from FPGA: \n", predict_boxes)
        np.copyto(output_boxes, predict_boxes)
        np.copyto(output_index, constant)
        # outputs.append(output_boxes)
        # outputs.append(output_index)
        # output_queue.put(outputs)
        compute_bounding_box_static(output_boxes, output_index, batch)

# p1.join()   
# p2.join()
print("**** Detection finished\n")
        
end = time.time()
total_time = end - start
print('Total time: ' + str(total_time) + ' s')
print('FPS:', num_img/total_time)



############## clean up #############
#xlnk.xlnk_reset()  

