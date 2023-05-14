import torch
import os
from tqdm import tqdm
import numpy as np
import argparse
import random
import csv

np.random.seed(0)

def write_npy(path, file):
    if not os.path.exists(path.rsplit('/', 1)[0]):
        os.makedirs(path.rsplit('/', 1)[0])
    np.save(path, file)

# def main(args):
#     filesList = list()
#     mix_lst=open('/home/panzexu/datasets/voxceleb2/audio_mixture/2_mix_min_800/mixture_data_list_2mix.csv').read().splitlines()
#     for line in mix_lst:
#         line=line.split(',')
#         for c in range(2):
#             from_path=args.lip_embedding_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.npy'
#             to_path = args.lip_embedding_direc_occl+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.npy'
#             filesList.append((from_path, to_path))

#     f=open("occluded_visual_list.csv",'w')
#     w=csv.writer(f)


#     train_max_length = 6
#     visual_sampling_rate = 25
#     min_occl_length = 2
#     min_non_occl_length = 2


#     for sample in tqdm(filesList):
#         (from_path, to_path) = sample
#         lips = np.load(from_path)
        

#         # if np.random.randint(0,10) == 0:
#         #     write_npy(to_path, lips)
#         #     continue

#         full_length = lips.shape[0]
#         occl_length = np.random.randint(0, full_length)
#         occl_lips = np.zeros((occl_length,256))

#         occl_start = np.random.randint(0, full_length - occl_length)

#         lips[occl_start:occl_start+occl_length,:] = occl_lips

#         write_npy(to_path, lips)
#         name = to_path[len(args.lip_embedding_direc_occl):]
#         w.writerow([name, occl_start, occl_length])

# def main_o(args):
#     filesList = list()
#     # mix_lst=open('/home/panzexu/datasets/voxceleb2/visual_embedding/sync/sync_av_occl/occluded_visual_list_sync_av.csv').read().splitlines()
#     mix_lst = open('./occluded_visual_list.csv').read().splitlines()
#     for line in mix_lst:
#         line=line.split(',')
#         from_path=args.lip_embedding_direc+line[0]
#         to_path = args.lip_embedding_direc_occl+line[0]
#         filesList.append((from_path, to_path, int(line[1]), int(line[2])))

#     f=open("occluded_visual_list.csv",'w')
#     w=csv.writer(f)

#     for sample in tqdm(filesList):
#         (from_path, to_path, occl_start, occl_length) = sample
#         lips = np.load(from_path)
        
#         occl_lips = np.zeros((occl_length,512))

#         lips[occl_start:occl_start+occl_length,:] = occl_lips

#         write_npy(to_path, lips)
#         name = to_path[len(args.lip_embedding_direc_occl):]
#         w.writerow([name, occl_start, occl_length])

def filter_repeat(args):
    # mix_lst=open('/home/panzexu/datasets/voxceleb2/visual_embedding/sync/sync_av_occl/occluded_visual_list_sync_av.csv').read().splitlines()
    mix_lst = open('./occluded_visual_list.csv').read().splitlines()
    visual_dic={}
    for line in mix_lst:
        line=line.split(',')
        if line[0].split('.')[0] in visual_dic:
            continue
        else:
            visual_dic[line[0].split('.')[0]] = (line[1],line[2])

    print(len(visual_dic))

    f=open("filtered_occluded_visual_list.csv",'w')
    w=csv.writer(f)
    for key in visual_dic:
        w.writerow([key, visual_dic[key][0], visual_dic[key][1]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LRS3 dataset')
    parser.add_argument('--lip_embedding_direc', default = '/home/panzexu/datasets/voxceleb2/visual_embedding/lip/', type=str)
    parser.add_argument('--lip_embedding_direc_occl',default = '/home/panzexu/datasets/voxceleb2/visual_embedding/lip_occl/', type=str)
    args = parser.parse_args()
    filter_repeat(args)


