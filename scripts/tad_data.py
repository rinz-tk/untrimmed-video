import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import py7zr
from itertools import islice


def read_skeleton(skeleton_info, n_joints):
    skeletons = skeleton_info.split("\n")

    skeletons = [s.split(" ")[:n_joints * 3] for s in skeletons]
    skeletons.pop()

    return np.array([
        [list(map(float, s[j*3 : (j+1)*3])) for j in range(len(s) // 3)] for s in skeletons
    ])


def interpolate_image(img, width):
    return tf.image.resize(img, size=(width, img.shape[1]), method=tf.image.ResizeMethod.BILINEAR).numpy()


def vit(skeleton):
    for t in skeleton:
        t -= t[0]


def tsi(skeleton, normalizing_const=255):
    maxes = np.max(skeleton, axis=(0, 1))
    mins = np.min(skeleton, axis=(0, 1))
    max_diff = np.max(maxes - mins)

    for c in range(skeleton.shape[-1]):
        skeleton[:,:,c] = np.floor(normalizing_const * (skeleton[:,:,c] - mins[c]) / max_diff) / normalizing_const


def create_motion(sk_image, width):
    sk_motion = sk_image[1:,:,:] - sk_image[:-1,:,:]
    sk_motion = interpolate_image(sk_motion, width)

    return sk_motion


def get_skeleton(skeleton_info, width, n_joints=25):
    sk_image = read_skeleton(skeleton_info, n_joints)

    frames = sk_image.shape[0]

    sk_image = interpolate_image(sk_image, width)
    vit(sk_image)
    tsi(sk_image)

    sk_motion = create_motion(sk_image, width)

    return sk_image, sk_motion, frames


def read_labels(label_info, label_count):
    labels = label_info.split("\n")
        
    labels = [list(map(int, l.split(",")[:-1])) for l in labels]
    labels.pop()

    return list(filter(lambda x: x[0] <= label_count, labels))


def interpolate_map(x, width, frames):
    return x * width / frames


def map_to_width(label, width, frames):
    center = (label[2] + label[1]) / 2

    start = np.floor(interpolate_map(label[1], width, frames))
    end = np.ceil(interpolate_map(label[2], width, frames))
    center = np.floor(interpolate_map(center, width, frames))

    return start, end, center


def map_to_out_width(start, end, center, stride):
    start = int(np.floor(start / stride) - 1)
    end = int(np.floor(end / stride) - 1)
    center = int(np.floor(center / stride) - 1)

    return start, end, center


def get_target(label_info, label_count, frames, width, stride):
    out_width = width // stride

    heatmap = np.zeros((out_width, 1, label_count))
    offset = np.zeros((out_width, 1, 1))
    size = np.zeros((out_width, 1, 1))

    labels = read_labels(label_info, label_count)

    for label in labels:
        category = label[0] - 1

        start, end, center = map_to_width(label, width, frames)
        off = (start / 2) - np.floor(start / 2)
        start, end, center = map_to_out_width(start, end, center, stride)
        sig = (end - start) / 6

        for x, h in enumerate(heatmap):
            h[0][category] += np.exp( - (x - center)**2 / ((2 * sig**2) + keras.backend.epsilon()))

        offset[center][0][0] = off
        size[center][0][0] = np.log(end - start + keras.backend.epsilon())

    return heatmap, offset, size


def get_dataset(skeleton_dir, label_dir, width, n_joints, stride, label_count, input_len=-1):
    skeleton_files = os.listdir(skeleton_dir)
    label_files = os.listdir(label_dir)

    if input_len < 0:
        input_len = len(skeleton_files)
    else:
        input_len = min(input_len, len(skeleton_files))

    input_image = np.empty((input_len, width, n_joints, 3))
    input_motion = np.empty((input_len, width, n_joints, 3))

    target_heatmap = np.empty((input_len, width // stride, 1, label_count))
    target_offset = np.empty((input_len, width // stride, 1, 1))
    target_size = np.empty((input_len, width // stride, 1, 1))

    for s, l, i in zip(skeleton_files, label_files, range(input_len)):
        skeleton_loc = os.path.join(skeleton_dir, s)
        label_loc = os.path.join(label_dir, l)

        with open(skeleton_loc) as f:
            skeleton_info = f.read()

        with open(label_loc) as f:
            label_info = f.read()
        
        input_image[i], input_motion[i], frames = get_skeleton(skeleton_info, width, n_joints)
        target_heatmap[i], target_offset[i], target_size[i] = get_target(label_info, label_count, frames, width, stride)

    return (input_image, input_motion), (target_heatmap, target_offset, target_size)


def get_dataset_from_compressed(skeleton_dir, label_dir, width, n_joints, stride, label_count, input_len=-1, load_batch_size=100):
    txt_filter = lambda x: x.endswith("txt")

    with py7zr.SevenZipFile(skeleton_dir, mode='r') as skeleton_z:
        skeleton_files = list(sorted(filter(txt_filter, skeleton_z.getnames())))

    with py7zr.SevenZipFile(label_dir, mode='r') as label_z:
        label_files = list(sorted(filter(txt_filter, label_z.getnames())))

    if input_len < 0:
        input_len = len(skeleton_files)
    else:
        input_len = min(input_len, len(skeleton_files))

    input_image = np.empty((input_len, width, n_joints, 3))
    input_motion = np.empty((input_len, width, n_joints, 3))

    target_heatmap = np.empty((input_len, width // stride, 1, label_count))
    target_offset = np.empty((input_len, width // stride, 1, 1))
    target_size = np.empty((input_len, width // stride, 1, 1))

    skeleton_iter = iter(skeleton_files)
    label_iter = iter(label_files)
    len_iter = iter(range(input_len))
    for batch in range(0, input_len, load_batch_size):
        print("Reading from {} ... {}".format(skeleton_files[batch], skeleton_files[min(input_len, batch+load_batch_size) - 1]))

        with py7zr.SevenZipFile(skeleton_dir, mode='r') as skeleton_z:
            skeleton_read = skeleton_z.read(skeleton_files[batch:batch+load_batch_size])

        with py7zr.SevenZipFile(label_dir, mode='r') as label_z:
            label_read = label_z.read(label_files[batch:batch+load_batch_size])

        for s, l, i in islice(zip(skeleton_iter, label_iter, len_iter), load_batch_size):
            skeleton_info = skeleton_read[s].getvalue().decode()
            label_info = label_read[l].getvalue().decode()

            input_image[i], input_motion[i], frames = get_skeleton(skeleton_info, width, n_joints)
            target_heatmap[i], target_offset[i], target_size[i] = get_target(label_info, label_count, frames, width, stride)

    return (input_image, input_motion), (target_heatmap, target_offset, target_size)
