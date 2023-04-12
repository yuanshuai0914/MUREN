import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def draw_feature_map(features,save_dir = '/home/ys/mmdetection/tools/feature_map/',name=None):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))  
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                #plt.imshow(superimposed_img,cmap='gray')
                #plt.show()
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                #superimposed_img = heatmap
                #plt.imshow(superimposed_img,cmap='gray')
                #plt.show()

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('img')
    parser.add_argument('saave_dir')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argumrnt('--device',default = 'cuda:1')
    model = init_detector(args.config, args.checkpoint,device=args.device)
    draw_feature_map(model,args.img,args.save_dir)


if __name__ == '__main__':
    main()
