# YOLO 資料集報告

產生時間：2025-11-10 21:47:29

## 整體摘要

| images | positive | negative | labels_per_image_mean | classes |
| --- | --- | --- | --- | --- |
| 20728 | 20728 | 0 | 1.000 | aortic_valve |


**BBox 統計（整體）**


| metric | count | mean | min | max | std |
| --- | --- | --- | --- | --- | --- |
| w_px | 20728 | 41.04 | 5.12 | 92.16 | 18.34 |
| h_px | 20728 | 41.39 | 6.14 | 92.16 | 18.38 |
| area_px | 20728 | 1935.90 | 41.94 | 7455.38 | 1361.17 |


## 目標中心熱力圖（10×10，ASCII）

```
          
          
          
   ░░░░   
   ░██░   
   ░██░   
   ░░░░   
          
          
          
```


---

### Split: train
**摘要**

| split | images | positive | negative | labels_per_image_mean |
| --- | --- | --- | --- | --- |
| train | 19572 | 19572 | 0 | 1.000 |

**部分影像清單（前 10 筆）**

| split | image_path | width | height | num_labels | is_positive |
| --- | --- | --- | --- | --- | --- |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201.png | 512 | 512 | 1 | 1 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_gauss.png | 512 | 512 | 1 | 1 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r180.png | 512 | 512 | 1 | 1 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r180_gauss.png | 512 | 512 | 1 | 1 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r180_sp.png | 512 | 512 | 1 | 1 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r270.png | 512 | 512 | 1 | 1 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r270_gauss.png | 512 | 512 | 1 | 1 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r270_sp.png | 512 | 512 | 1 | 1 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r90.png | 512 | 512 | 1 | 1 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r90_gauss.png | 512 | 512 | 1 | 1 |

**部分 BBox（前 10 筆）**

| split | image_path | cls | x_center | y_center | w | h | x_center_px | y_center_px | w_px | h_px | area_norm | area_px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201.png | 0 | 0.416 | 0.551 | 0.031 | 0.051 | 212.992 | 282.112 | 15.872 | 26.112 | 0.001581 | 414.449664 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_gauss.png | 0 | 0.416 | 0.551 | 0.031 | 0.051 | 212.992 | 282.112 | 15.872 | 26.112 | 0.001581 | 414.449664 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r180.png | 0 | 0.584 | 0.449 | 0.031 | 0.051 | 299.008 | 229.888 | 15.872 | 26.112 | 0.001581 | 414.449664 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r180_gauss.png | 0 | 0.584 | 0.449 | 0.031 | 0.051 | 299.008 | 229.888 | 15.872 | 26.112 | 0.001581 | 414.449664 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r180_sp.png | 0 | 0.584 | 0.449 | 0.031 | 0.051 | 299.008 | 229.888 | 15.872 | 26.112 | 0.001581 | 414.449664 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r270.png | 0 | 0.449 | 0.416 | 0.051 | 0.031 | 229.888 | 212.992 | 26.112 | 15.872 | 0.001581 | 414.449664 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r270_gauss.png | 0 | 0.449 | 0.416 | 0.051 | 0.031 | 229.888 | 212.992 | 26.112 | 15.872 | 0.001581 | 414.449664 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r270_sp.png | 0 | 0.449 | 0.416 | 0.051 | 0.031 | 229.888 | 212.992 | 26.112 | 15.872 | 0.001581 | 414.449664 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r90.png | 0 | 0.551 | 0.584 | 0.051 | 0.031 | 282.112 | 299.008 | 26.112 | 15.872 | 0.001581 | 414.449664 |
| train | C:\Users\Lab902\Desktop\week8\pc\datasets\train\images\patient0001_0201_r90_gauss.png | 0 | 0.551 | 0.584 | 0.051 | 0.031 | 282.112 | 299.008 | 26.112 | 15.872 | 0.001581 | 414.449664 |

### Split: val
**摘要**

| split | images | positive | negative | labels_per_image_mean |
| --- | --- | --- | --- | --- |
| val | 1156 | 1156 | 0 | 1.000 |

**部分影像清單（前 10 筆）**

| split | image_path | width | height | num_labels | is_positive |
| --- | --- | --- | --- | --- | --- |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0122.png | 512 | 512 | 1 | 1 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0123.png | 512 | 512 | 1 | 1 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0124.png | 512 | 512 | 1 | 1 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0125.png | 512 | 512 | 1 | 1 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0126.png | 512 | 512 | 1 | 1 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0127.png | 512 | 512 | 1 | 1 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0128.png | 512 | 512 | 1 | 1 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0129.png | 512 | 512 | 1 | 1 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0130.png | 512 | 512 | 1 | 1 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0131.png | 512 | 512 | 1 | 1 |

**部分 BBox（前 10 筆）**

| split | image_path | cls | x_center | y_center | w | h | x_center_px | y_center_px | w_px | h_px | area_norm | area_px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0122.png | 0 | 0.407 | 0.551 | 0.029 | 0.039 | 208.384 | 282.112 | 14.848 | 19.968 | 0.001131 | 296.484864 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0123.png | 0 | 0.407 | 0.55 | 0.033 | 0.049 | 208.384 | 281.6 | 16.896 | 25.088 | 0.0016170000000000002 | 423.88684800000004 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0124.png | 0 | 0.41 | 0.546 | 0.035 | 0.064 | 209.92 | 279.552 | 17.92 | 32.768 | 0.0022400000000000002 | 587.2025600000001 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0125.png | 0 | 0.407 | 0.545 | 0.041 | 0.07 | 208.384 | 279.04 | 20.992 | 35.84 | 0.0028700000000000006 | 752.3532800000002 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0126.png | 0 | 0.405 | 0.545 | 0.041 | 0.074 | 207.36 | 279.04 | 20.992 | 37.888 | 0.003034 | 795.344896 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0127.png | 0 | 0.406 | 0.545 | 0.039 | 0.074 | 207.872 | 279.04 | 19.968 | 37.888 | 0.0028859999999999997 | 756.5475839999999 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0128.png | 0 | 0.406 | 0.545 | 0.043 | 0.078 | 207.872 | 279.04 | 22.016 | 39.936 | 0.0033539999999999998 | 879.2309759999999 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0129.png | 0 | 0.405 | 0.544 | 0.045 | 0.084 | 207.36 | 278.528 | 23.04 | 43.008 | 0.00378 | 990.90432 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0130.png | 0 | 0.408 | 0.543 | 0.047 | 0.09 | 208.896 | 278.016 | 24.064 | 46.08 | 0.00423 | 1108.86912 |
| val | C:\Users\Lab902\Desktop\week8\pc\datasets\val\images\patient0031_0131.png | 0 | 0.405 | 0.541 | 0.053 | 0.098 | 207.36 | 276.992 | 27.136 | 50.176 | 0.005194 | 1361.575936 |
