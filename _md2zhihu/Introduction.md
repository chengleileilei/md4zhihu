# 机器学习开源算法平台DemoHub

[TOC]

## 一、简介

随着人工智能技术的蓬勃发展，越来越多的人渴望体验AI的魅力。然而复杂的理论知识和编程技巧使初学者望而却步。为了降低AI学习的门槛，快速体验AI算法，培养学生对于AI的兴趣，我们开发了一个小项目 DEMOHUB。

DemoHub 是一个平台由北京交通大学[ADaM](https://adam-bjtu.org/)实验室开发，该平台汇总了经典和最新的开源算法，包括图像生成、图像分类、目标检测、图像分割、数字水印、图像增强、风格迁移、用户心理画像等多种任务，并提供可视化工具。平台支持中英文切换和模型检索。**快速访问：https://demohub.bjtu.edu.cn/**

## 二、支持任务

目前支持了图像分类、目标检测、图像分割、图像增广、数字水印、用户画像、图像生成等领域的一些经典和前沿算法，更多算法在逐步加入中，欢迎大家提出宝贵的意见。

<table>
<tr class="header">
<th style="text-align: center;">任务</th>
<th style="text-align: center;">模型</th>
</tr>
<tr class="odd">
<td style="text-align: center;"><strong>图像分类</strong></td>
<td style="text-align: center;"><a href="https://demohub.bjtu.edu.cn/#/model/classification/efficientnet">EfficientNet</a>、 <a href="https://demohub.bjtu.edu.cn/#/model/classification/resnext">ResNeXt</a>、 <a href="https://demohub.bjtu.edu.cn/#/model/classification/resnet">ResNet</a>、 <a href="https://demohub.bjtu.edu.cn/#/model/classification/regnet">RegNet</a>、 <a href="https://demohub.bjtu.edu.cn/#/model/classification/alexnet">Alexnet</a></td>
</tr>
<tr class="even">
<td style="text-align: center;"><strong>目标检测</strong></td>
<td style="text-align: center;"><a href="https://demohub.bjtu.edu.cn/#/model/object_detection/yolov5">YoloV5s</a>、 <a href="https://demohub.bjtu.edu.cn/#/model/object_detection/mm_maskrcnn">Mask R-CNN</a>、 <a href="https://demohub.bjtu.edu.cn/#/model/object_detection/mm_gcnet">GCNet</a>、 <a href="https://demohub.bjtu.edu.cn/#/model/object_detection/mm_gridrcnn">Grid R-CNN</a>、 <a href="https://demohub.bjtu.edu.cn/#/model/object_detection/mm_fasterrcnn">Faster R-CNN</a></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><strong>图像分割</strong></td>
<td style="text-align: center;"><a href="https://demohub.bjtu.edu.cn/#/model/segmentation/segment_anything">Segment Anything</a>、 <a href="https://demohub.bjtu.edu.cn/#/model/segmentation/mm_ccnet">CCNet</a>、 <a href="https://demohub.bjtu.edu.cn/#/model/segmentation/mm_ccnet">ANN、</a> <a href="https://demohub.bjtu.edu.cn/#/model/segmentation/mm_fast_fcn">Fast FCN</a></td>
</tr>
<tr class="even">
<td style="text-align: center;"><strong>图像增广</strong></td>
<td style="text-align: center;"><a href="https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel">Albumentations(像素级)</a>、 <a href="https://demohub.bjtu.edu.cn/#/model/augmentations">Albumentations(空间级)</a></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><strong>图像处理</strong></td>
<td style="text-align: center;"><a href="https://demohub.bjtu.edu.cn/#/model/image_processing/equalize_hist">EqualizeHist</a>、 <a href="https://demohub.bjtu.edu.cn/#/model/image_processing/canny">Canny</a></td>
</tr>
<tr class="even">
<td style="text-align: center;"><strong>数字水印</strong></td>
<td style="text-align: center;"><a href="https://demohub.bjtu.edu.cn/#/model/digital_watermark/lsb">LSB</a>、 <a href="https://demohub.bjtu.edu.cn/#/model/digital_watermark/svd">SVD</a></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><strong>用户画像</strong></td>
<td style="text-align: center;"><a href="https://demohub.bjtu.edu.cn/#/model/vbfi/vbfi">vBFI</a></td>
</tr>
<tr class="even">
<td style="text-align: center;"><strong>图像生成</strong></td>
<td style="text-align: center;"><a href="https://demohub.bjtu.edu.cn/#/model/diffusion/text2img">Text To Image</a>、<a href="Image%20To%20Image">Image To Image</a></td>
</tr>
</table>

-   **图像分类**

    1.  [EfficientNet](https://demohub.bjtu.edu.cn/#/model/classification/efficientnet)

    1.  [ResNeXt](https://demohub.bjtu.edu.cn/#/model/classification/resnext)

    1.  [ResNet](https://demohub.bjtu.edu.cn/#/model/classification/resnet)

    1.  [RegNet](https://demohub.bjtu.edu.cn/#/model/classification/regnet)

    1.  [Alexnet](https://demohub.bjtu.edu.cn/#/model/classification/alexnet)

-   **目标检测**

    1.  [YoloV5s](https://demohub.bjtu.edu.cn/#/model/object_detection/yolov5)
    1.  [Mask R-CNN](https://demohub.bjtu.edu.cn/#/model/object_detection/mm_maskrcnn)
    1.  [GCNet](https://demohub.bjtu.edu.cn/#/model/object_detection/mm_gcnet)
    1.  [Grid R-CNN](https://demohub.bjtu.edu.cn/#/model/object_detection/mm_gridrcnn)
    1.  [Faster R-CNN](https://demohub.bjtu.edu.cn/#/model/object_detection/mm_fasterrcnn)

-   **图像分割**

    1.  [Segment Anything](https://demohub.bjtu.edu.cn/#/model/segmentation/segment_anything)
    1.  [CCNet](https://demohub.bjtu.edu.cn/#/model/segmentation/mm_ccnet)
    1.  [ANN](https://demohub.bjtu.edu.cn/#/model/segmentation/mm_ccnet)
    1.  [Fast FCN](https://demohub.bjtu.edu.cn/#/model/segmentation/mm_fast_fcn)

-   **图像增强**

    1.  [Albumentations(像素级)](https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel)
    1.  [Albumentations(空间级)](https://demohub.bjtu.edu.cn/#/model/augmentations)

-   **图像处理**

    1.  [EqualizeHist](https://demohub.bjtu.edu.cn/#/model/image_processing/equalize_hist)
    1.  [Canny](https://demohub.bjtu.edu.cn/#/model/image_processing/canny)

-   **数字水印**

    1.  [LSB](https://demohub.bjtu.edu.cn/#/model/digital_watermark/lsb)
    1.  [SVD](https://demohub.bjtu.edu.cn/#/model/digital_watermark/svd)

-   **用户画像**

    1.  [vBFI](https://demohub.bjtu.edu.cn/#/model/vbfi/vbfi)

-   **Stable Diffusion**

    1.  [Text To Image](https://demohub.bjtu.edu.cn/#/model/diffusion/text2img)

    1.  [Image To Image](Image To Image)

## 三、平台使用

1.  以AlexNet分类模型为例，简介部分可直接访问模型的**论文地址**和**github仓库**；

![image-20230805154603988](https://cdn.jsdelivr.net/gh/chengleileilei/md4zhihu@main-md2zhihu-asset/Introduction/9ce81f03aed662ca-image-20230805154603988.png)

1.  输入部分可选择上传本地图片或直接使用系统提供的样例图片，本地上传支持jpg、gif、jpeg和png格式的图片，单张图片大小最大支持5M；

![image-20230805154206101](https://cdn.jsdelivr.net/gh/chengleileilei/md4zhihu@main-md2zhihu-asset/Introduction/ceeee8e4d73df011-image-20230805154206101.png)

1.  点击**Clear**可清除输入和输出结果，点击**Submit**后模型会对输入图片进行预测，预测结果在右侧显示；在预测过程中点击**clear**可中断计算；

![image-20230805154908739](https://cdn.jsdelivr.net/gh/chengleileilei/md4zhihu@main-md2zhihu-asset/Introduction/e6c411dbac2ff0c4-image-20230805154908739.png)

1.  参数模块：部分模型提供参数自定以功能

    -   图像增强部分参数结构如下，function用于选择增强函数，function args用于编辑函数参数，function为参数说明信息；

    ![image-20230805171549700](https://cdn.jsdelivr.net/gh/chengleileilei/md4zhihu@main-md2zhihu-asset/Introduction/018943a3d0dbebce-image-20230805171549700.png)

    -   Stable Diffusion prompt参数结构如下，目前仅支持英文prompt

    ![image-20230805172457142](https://cdn.jsdelivr.net/gh/chengleileilei/md4zhihu@main-md2zhihu-asset/Introduction/1a9494e049dbedf3-image-20230805172457142.png)

## 四、任务样例

1.  **图像分类**

    ![image-20230805174929414](https://cdn.jsdelivr.net/gh/chengleileilei/md4zhihu@main-md2zhihu-asset/Introduction/21cb8e608b6b09fa-image-20230805174929414.png)

1.  **目标检测**

    ![image-20230805175057442](https://cdn.jsdelivr.net/gh/chengleileilei/md4zhihu@main-md2zhihu-asset/Introduction/3c6b5778a8cc3d9a-image-20230805175057442.png)

1.  **图像分割**

    ![image-20230805175222401](https://cdn.jsdelivr.net/gh/chengleileilei/md4zhihu@main-md2zhihu-asset/Introduction/6ff0e2a6133ca5ec-image-20230805175222401.png)

1.  **图像增强**

    ![image-20230805175348186](https://cdn.jsdelivr.net/gh/chengleileilei/md4zhihu@main-md2zhihu-asset/Introduction/5f9f2594bc864bc0-image-20230805175348186.png)

1.  **图像处理**

    ![image-20230805175508138](https://cdn.jsdelivr.net/gh/chengleileilei/md4zhihu@main-md2zhihu-asset/Introduction/b242e1398206d3ea-image-20230805175508138.png)

1.  **数字水印**

    -   数字盲水印嵌入![image-20230805180031777](https://cdn.jsdelivr.net/gh/chengleileilei/md4zhihu@main-md2zhihu-asset/Introduction/63078fb3b775b669-image-20230805180031777.png)

    -   水印提取![image-20230805180126891](https://cdn.jsdelivr.net/gh/chengleileilei/md4zhihu@main-md2zhihu-asset/Introduction/b6b4262b3a43faca-image-20230805180126891.png)

1.  **用户画像**

    ![image-20230805180242063](https://cdn.jsdelivr.net/gh/chengleileilei/md4zhihu@main-md2zhihu-asset/Introduction/56fef3fe912cea92-image-20230805180242063.png)

1.  **图像生成**

    ![image-20230805180801082](https://cdn.jsdelivr.net/gh/chengleileilei/md4zhihu@main-md2zhihu-asset/Introduction/f98f01bd326781ef-image-20230805180801082.png)



Reference:

