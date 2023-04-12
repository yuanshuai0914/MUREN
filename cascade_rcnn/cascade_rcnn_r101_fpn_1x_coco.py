fp16=dict(loss_scale=512.)
_base_ = './cascade_rcnn_r50_fpn_1x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

