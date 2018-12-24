from PIL import Image
import cv2
import sys, os

#path
in_file = "./padded_santa/"
out_file = "./resize_santa"


#imagesリスト
files = os.listdir(in_file)

fileNo = 0
for i in files:
    #画像読み込み
    images = Image.open(in_file + i)
    #画像リサイズ
    images_resize = images.resize((64, 64))

    save_path = (out_file + '/' + str(fileNo) + '.jpg')
    images_resize.save(save_path)
    fileNo += 1