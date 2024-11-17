import os

from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.npy'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def tensor2im(var):
    # var shape: (3, H, W)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def make_dataset(dir):
    images = []
    if os.path.isfile(dir):
        path = dir
        fname = os.path.basename(path)[:-4]
        images.append((fname, path))
    else:
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    fname = fname.split('.')[0]
                    c_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Illustration_2_label/{fname}.npy'
                    # c_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Pixar_label_new/{fname}.npy'
                    # c_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Illustration_label/{fname}.npy'
                    # c_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Caricature_label/{fname}.npy'
                    # c_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Comic_label/{fname}.npy'
                    # c_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Arance_label/{fname}.npy'
                    # c_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Impasto_label/{fname}.npy'
                    # c_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/Fantasy_label/{fname}.npy'
                    images.append((fname, path, c_path))
    return images


def make_dataset_label(dir):
    images = []
    if os.path.isfile(dir):
        path = dir
        fname = os.path.basename(path)[:-4]
        images.append((fname, path))
    else:
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    w_path = os.path.join(root, fname)
                    fname = fname.split('.')[0]
                    c_name = fname.split('_')[0]
                    c_path = f'/data1/sch/EG3D-projector-master/eg3d/projector_test_data/style_eg3d/label/{c_name}.npy'
                    images.append((fname, w_path, c_path))
    return images