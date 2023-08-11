# Albumentation文档

## 一、简介

**Albumentations** 是一个用于图像增强的 Python 库。图像增强可用于深度学习和计算机视觉任务以提高训练模型的质量。该库广泛应用于[工业界](https://albumentations.ai/whos_using#industry)、[深度学习研究](https://albumentations.ai/whos_using#research)、[机器学习竞赛](https://albumentations.ai/whos_using#competitions)和[开源项目](https://albumentations.ai/whos_using#open-source)。

**图像增强：**分为像素级增强和空间级增强

- 对于图像分割任务

  	- 像素级增强会更改原始图像的像素值，但不会更改输出蒙版；

  - **空间级增强**会改变图像和掩模，此时需要重新标注；

- 物体检测任务：

  - 像素级增强同样不会改变输出；
  - **像素级增强**在对图像应用变换同时，还需要对边界框坐标应用相同的变换，更新边界框的坐标以正确表示增强图像上对象的位置。

**优势：**

- 处理速度更快

- 适用多种任务：图像分类、语义分割、实例分割、对象检测、关键点检测

  - 对于图像分类任务：镜像、裁剪、改变亮度和对比度等基本图像变换满足任务要求；

  - 但对于分割、对象检测和关键点检测等任务，应用**空间级增强**后还需对蒙版或边界框等进行更新，Albumentation在对此类任务进行空间级增强后会对蒙版、边框等标签应用相同的变化，这大大减少了重新标注数据的工作量。

- 支持以管道的形式调用统一的接口进行图像转换：

      - ```python
        import albumentations as A
    
    transform = A.Compose([
        A.RandomCrop(512, 512),
        A.RandomBrightnessContrast(p=0.3),
        A.HorizontalFlip(p=0.5),
    ])
        ```



## 二、安装和使用

**从 PyPI 安装最新的稳定版本**：

```shell	
pip install -U albumentations
```

**Demo示例**

```python
import albumentations as A
import cv2

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

image = cv2.imread("/path/to/image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transformed = transform(image=image)
transformed_image = transformed["image"]

```

## 三、API

### 参数说明

- `p`：api生效的概率，为0到1之间的小数；
- `always_apply`：是否一直生效，`always_apply=True`会忽略`p`，直接生效。

### 像素级增强

1. AdvancedBlur

    - 说明：使用带有随机选择参数的广义法线滤波器对输入图像模糊处理。此变换还在卷积前向生成的内核添加乘性噪声。

    - 参数：

      - ```python
        blur_limit=(11, 21), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0), rotate_limit=90, beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1), always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.AdvancedBlur.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/AdvancedBlur

2. Blur

    - 说明：使用随机大小的内核模糊输入图像。

    - 参数：

      - ```python
        blur_limit=7, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Blur.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/Blur

3. CLAHE

    - 说明：在输入图像上应用对比度有限的适应直方图均衡化。

    - 参数：

      - ```python
        clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.CLAHE.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/CLAHE

4. ChannelDropout

    - 说明：输入图像中的随机丢弃通道。

    - 参数：

      - ```python
        channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.ChannelDropout.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/ChannelDropout

6. ChannelShuffle

    - 说明：随机重新排列输入 RGB 图像的通道。

    - 参数：

      - ```python
        always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.ChannelShuffle.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/ChannelShuffle

7. ColorJitter

    - 说明：随机更改图像的亮度、对比度和饱和度。 与 torchvision 的 ColorJitter 相比，此转换给出的结果略有不同，因为 Pillow（在 torchvision 中使用）和 OpenCV（在 Albumentations 中使用）通过不同的公式将图像转换为 HSV 格式。 另一个区别 是 Pillow 使用 uint8overflow，但Albumentations 使用值饱和。

    - 参数：

      - ```python
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.ColorJitter.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/ColorJitter

8. Defocus

    - 说明：使用离焦变换，请参考 https://arxiv.org/abs/1903.12261。

    - 参数：

      - ```python
        radius=(3, 10), alias_blur=(0.1, 0.5), always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Defocus.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/Defocus

9. Downscale

    - 说明：通过缩小和放大来降低图像质量。

    - 参数：

      - ```python
        scale_min=0.25, scale_max=0.25, interpolation=None, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Downscale.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/Downscale

10. Emboss

    - 说明：对输入图像进行浮雕并将结果与原始图像叠加。

    - 参数：

      - ```python
        alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Emboss.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/Emboss

11. Equalize

    - 说明：均衡图像直方图。

    - 参数：

      - ```python
        by_channels=True, mask=None, mask_params=(), always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Equalize.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/Equalize

12. FancyPCA

    - 说明：使用 Krizhevsky 论文中的 FancyPCA 增强 RGB 图像

    - 参数：

      - ```python
        alpha=0.9, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.FancyPCA.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/FancyPCA

13. FromFloat

    - 说明：获取一个输入数组，其中所有值都应在 [0, 1.0] 范围内，将它们乘以“max_value”，然后将结果值转换为“dtype”指定的类型。 如果“max_value”为“None”，则转换将尝试从“dtype”参数推断数据类型的最大值。

    - 参数：

      - ```python
        max_value=None, always_apply=False, p=1.0
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.FromFloat.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/FromFloat

14. GaussNoise

    - 说明：对输入图像应用高斯噪声。

    - 参数：

      - ```python
        var_limit=(10.0, 90.0), mean=0, per_channel=True, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.GaussNoise.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/GaussNoise

15. GaussianBlur

    - 说明：模糊的输入图像使用高斯滤波器与随机核大小。

    - 参数：

      - ```python
        blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.GaussianBlur.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/GaussianBlur

16. GlassBlur

    - 说明：对输入图像应用玻璃噪声。

    - 参数：

      - ```python
        sigma=0.7, max_delta=4, iterations=2, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.GlassBlur.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/GlassBlur

17. HueSaturationValue

    - 说明：HueSaturationValue

    - 参数：随机改变输入图像的色调、饱和度和明度。

      - ```python
        hue_shift_limit=90, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.HueSaturationValue.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/HueSaturationValue

18. ISONoise

    - 说明：应用照相机传感器噪声。

    - 参数：

      - ```python
        color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.ISONoise.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/ISONoise

19. ImageCompression

    - 说明：减少图像的 Jpeg、WebP 压缩。

    - 参数：

      - ```python
        quality_lower=1, quality_upper=100, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.ImageCompression.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/ImageCompression

20. InvertImg

    - 说明：通过从255减去像素值来反转输入图像。

    - 参数：

      - ```python
        always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.InvertImg.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/InvertImg

21. MedianBlur

    - 说明：使用具有随机孔径线性尺寸的中值滤波器模糊输入图像。

    - 参数：

      - ```python
        blur_limit=11, always_apply=False, p=1
          ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.MedianBlur.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/MedianBlur

22. MotionBlur

    - 说明：使用随机大小的内核将运动模糊应用于输入图像。

    - 参数：

      - ```python
        blur_limit=7, allow_shifted=True, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.MotionBlur.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/MotionBlur

23. MultiplicativeNoise

    - 说明：将图像乘以随机数或数组。

    - 参数：

      - ```python
        multiplier=(0.9, 1.9), per_channel=False, elementwise=False, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.MultiplicativeNoise.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/MultiplicativeNoise

24. Normalize

    - 说明：通过以下公式应用归一化：“img = (img -mean * max_pixel_value) / (std * max_pixel_value)”

    - 参数：

      - ```python
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Normalize.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/Normalize

25. Posterize

    - 说明：减少每个颜色通道的位数。

    - 参数：

      - ```python
        num_bits=4, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Posterize.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/Posterize

26. RGBShift

    - 说明：随机移动输入 RGB 图像的每个通道的值。

    - 参数：

      - ```python
        r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RGBShift.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/RGBShift

27. RandomBrightnessContrast

    - 说明：随机改变输入图像的亮度和对比度。

    - 参数：

      - ```python
        brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RandomBrightnessContrast.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/RandomBrightnessContrast

28. RandomFog

    - 说明：来自 https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    - 参数：

      - ```python
        fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RandomFog.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/RandomFog

29. RandomGamma

    - 说明：随机伽玛

    - 参数：

      - ```python
        gamma_limit=(80, 220), eps=None, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RandomGamma.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/RandomGamma

30. RandomRain

    - 说明：添加下雨效果。 来自 https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    - 参数：

      - ```python
        slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RandomRain.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/RandomRain

31. RandomShadow

    - 说明：模拟图像的阴影。来自 https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    - 参数：

      - ```python
        shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RandomShadow.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/RandomShadow

32. RandomSnow

    - 说明：模拟雪景漂白一些像素值。来自 https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    - 参数：

      - ```python
        snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RandomSnow.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/RandomSnow

33. RandomSunFlare

    - 说明：模拟图像的太阳耀斑。来自 https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    - 参数：

      - ```python
        flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RandomSunFlare.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/RandomSunFlare

34. RandomToneCurve

    - 说明：通过操纵色调曲线来随机改变图像的亮区和暗区之间的关系。

    - 参数：

      - ```python
        scale=0.6, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RandomToneCurve.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_pixel/RandomToneCurve

### 空间级增强
1. Affine

    - 说明：将仿射变换应用于图像的增强。主要是 OpenCV 中相应类和函数的包装。

    - 参数：

      - ```python
        scale=None, translate_percent=None, translate_px=None, rotate=None, shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=False, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Affine.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/Affine

2. CenterCrop

    - 说明：裁剪输入的中心部分。

    - 参数：

      - ```python
        height=200, width=300, always_apply=False, p=1.0
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.CenterCrop.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/CenterCrop

3. CoarseDropout

    - 说明：图像中矩形区域的 CoarseDropout。

    - 参数：

      - ```python
        max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.CoarseDropout.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/CoarseDropout

4. Crop

    - 说明：从图像中裁剪区域。

    - 参数：

      - ```python
        x_min=0, y_min=0, x_max=300, y_max=200, always_apply=False, p=1.0
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Crop.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/Crop

5. ElasticTransform

    - 说明：图像的弹性变形。

    - 参数：

      - ```python
        alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, approximate=False, same_dxdy=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.ElasticTransform.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/ElasticTransform

6. Flip

    - 说明：水平、垂直或水平和垂直翻转输入。

    - 参数：

      - ```python
        always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Flip.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/Flip

7. GridDistortion

    - 说明：网格畸变。

    - 参数：

      - ```python
        num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, normalized=False, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.GridDistortion.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/GridDistortion

8. GridDropout

    - 说明：以网格方式丢弃图像的矩形区域和相应的掩模。

    - 参数：

      - ```python
        ratio=0.5, unit_size_min=None, unit_size_max=None, holes_number_x=None, holes_number_y=None, shift_x=0, shift_y=0, random_offset=False, fill_value=0, mask_fill_value=None, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.GridDropout.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/GridDropout

9. HorizontalFlip

    - 说明：绕 y 轴水平翻转输入。

    - 参数：

      - ```python
        always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.HorizontalFlip.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/HorizontalFlip

10. Lambda

    - 说明：用户自定义变换函数。

    - 参数：

      - ```python
        image=None, mask=None, keypoint=None, bbox=None, name=None, always_apply=False, p=1.0
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Lambda.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/Lambda

11. LongestMaxSize

    - 说明：重新缩放图像，使最大边等于 max_size，同时保持初始图像的纵横比。

    - 参数：

      - ```python
        max_size=300, interpolation=1, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.LongestMaxSize.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/LongestMaxSize

12. NoOp

    - 说明：不做任何处理。

    - 参数：

      - ```python
        always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.NoOp.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/NoOp

13. OpticalDistortion

    - 说明：光学畸变。

    - 参数：

      - ```python
        distort_limit=0.95, shift_limit=0.95, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.OpticalDistortion.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/OpticalDistortion

14. PadIfNeeded

    - 说明：如果边数小于所需数量，则填充图像的边数/最大值。

    - 参数：

      - ```python
        min_height=450, min_width=700, pad_height_divisor=None, pad_width_divisor=None, border_mode=4, value=None, mask_value=None, always_apply=False, p=1.0
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)<img src="allbumentation/dog.jpg.albumentations.PadIfNeeded.jpg&t=0.jpeg" alt="img" style="zoom: 80%;" />

15. Perspective

    - 说明：对输入执行随机四点透视变换。

    - 参数：

      - ```python
        scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=False, interpolation=1, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Perspective.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/PadIfNeeded

16. PiecewiseAffine

    - 说明：应用局部邻域之间不同的仿射变换。 这种增强在图像上放置规则的点网格，并通过仿射变换随机移动这些点的邻域。 这会导致局部扭曲。

    - 参数：

      - ```python
        scale=(0.03, 0.05), nb_rows=4, nb_cols=4, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, absolute_scale=False, always_apply=False, keypoints_threshold=0.01, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.PiecewiseAffine.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/PiecewiseAffine

17. PixelDropout

    - 说明：以一定概率将像素设置为 0。

    - 参数：

      - ```python
        dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.PixelDropout.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/PixelDropout

18. RandomCrop

    - 说明：随机裁切。

    - 参数：

      - ```python
        height=250, width=400, always_apply=False, p=1.0
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RandomCrop.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/RandomCrop

19. RandomCropFromBorders

    - 说明：从图像中裁剪 bbox，从边框中随机剪切部分，最后不调整大小

    - 参数：

      - ```python
        crop_left=0.1, crop_right=0.1, crop_top=0.1, crop_bottom=0.1, always_apply=False, p=1.0
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RandomCropFromBorders.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/RandomCropFromBorders

20. RandomResizedCrop

    - 说明：Torchvision 的变体裁剪输入的随机部分并将其调整为一定大小。

    - 参数：

      - ```python
        height=200, width=300, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=False, p=1.0
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RandomResizedCrop.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/RandomResizedCrop

21. RandomRotate90

    - 说明：将输入随机旋转 90 度零次或多次。

    - 参数：

      - ```python
        always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RandomRotate90.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/RandomRotate90

22. RandomScale

    - 说明：

    - 参数：

      - ```python
        scale_limit=0.1, interpolation=1, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RandomScale.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/RandomScale

23. RandomSizedCrop

    - 说明：裁剪输入的随机部分并将其重新缩放到一定大小。

    - 参数：

      - ```python
        min_max_height=[100,300], height=250, width=400, w2h_ratio=1.0, interpolation=1, always_apply=False, p=1.0
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.RandomSizedCrop.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/RandomSizedCrop

24. Resize

    - 说明：将输入的大小调整为给定的高度和宽度。

    - 参数：

      - ```python
        height=250, width=400, interpolation=1, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Resize.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/Resize

25. Rotate

    - 说明：将输入旋转从均匀分布中随机选择的角度。

    - 参数：

      - ```python
        limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, crop_border=False, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Rotate.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/Rotate

26. SafeRotate

    - 说明：将输入在输入框架内旋转从均匀分布中随机选择的角度。

    - 参数：

      - ```python
        limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.SafeRotate.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/SafeRotate

27. ShiftScaleRotate

    - 说明：随机应用仿射变换：平移、缩放和旋转输入。

    - 参数：

      - ```python
        shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.ShiftScaleRotate.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/ShiftScaleRotate

28. SmallestMaxSize

    - 说明：重新缩放图像，使最小边等于 max_size，同时保持初始图像的纵横比。

    - 参数：

      - ```python
        max_size=300, interpolation=1, always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.SmallestMaxSize.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/SmallestMaxSize

29. Transpose

    - 说明：通过交换行和列来转置输入。

    - 参数：

      - ```python
        always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.Transpose.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/Transpose

30. VerticalFlip

    - 说明：绕 x 轴垂直翻转输入。

    - 参数：

      - ```python
        always_apply=False, p=1
        ```

    - 样例输出

      - ![img](allbumentation/dog.jpg)
      - ![img](allbumentation/dog.jpg.albumentations.VerticalFlip.jpg&t=0.jpeg)

    - 访问链接：https://demohub.bjtu.edu.cn/#/model/augmentations/albumentations_spatial/VerticalFlip

