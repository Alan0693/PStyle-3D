import json
import numpy as np
from torch import float32
import os
from PIL import Image

f = open("/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/illustration_2_labels.json", encoding='utf-8')
setting = json.load(f)

images = setting['labels']
# name = images[0][0][:5]
# c = images[0][1]
# a = np.array(c)
# print(name)
# print(a)
# np.save(f'./projector_test_data/FFHQ/label/{name}.npy', a)

# c_path = "/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Pixar_label/pixar_generate_00.npy"
# ccc = np.load(c_path)
# # ccc = np.reshape(c,(1,25))
# print(ccc)
# i = 0
# for img in images:
#     if i == 2000:
#         break
#     if i % 2 == 0:
#         name = img[0][:5]
#         pose = img[1]
#         c = np.array(pose)
#         np.save(f'./projector_test_data/FFHQ/label/{name}.npy', c)
#         i = i + 1
#     else:
#         i = i + 1
#         continue
# for_name = 'img000'
# i = 999
# name = str(i).zfill(5)
# final_name = for_name + name
for img in images:
    # if i == 1000:
    #     break
    # name = img[0][:5]
    # name = str(i).zfill(5)
    name = str(img[0][:-4])
    # final_name = for_name + name
    pose = img[1][:25]
    c = np.array(pose)
    np.save(f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Illustration_2_label/{name}.npy', c)




# label_path = os.listdir('/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Illustration_new_label/')
# for labels in label_path:
#     print(labels)
#     fname = labels.split('.')[0]
#     # fname_new = fname + '_01'
#     # print(fname_new)
#     print(fname)
#     data_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Illustration_new/{fname}.png'
#     img = Image.open(data_path).convert('RGB')
#     img.save(f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Illustration_new_FFHQ/{fname}.png')


