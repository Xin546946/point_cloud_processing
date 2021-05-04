## Fundamental of Point cloud processing
### In this repository, we have implemented several point cloud processing algorithms in C++ and python. For each project, you will have an overview from the readme file, where we introduce the idea of the algorithms and some illustrations of the results.
### In this file, we will briefly introduce some results of implementation to draw your interests. 

### 1.1 PCA and Surface Normal (python)
<center class="half">
    <img src="./doc/fig1.png" width="300"/><img src="./doc/fig2.png" width="280"/>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.1 PCA and Surface Normal of the point cloud</div>

</center>

### 1.2 Upsampling(C++)
<center>
    <img src="./doc/fig3.png" width="500"/>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.2 Above: RGB image; Middle: Depth image; Down: Upsampled depth image</div>
</center>

### 2. NN Search Tree (python & C++)
~~~
My kdtree ------------
Kdtree: build 283.742ms, knn 11.855ms, radius 0.505ms, brute 13.218ms
~~~
~~~
My octree --------------
Octree: build 11727.653ms, knn 0.981ms, radius 0.732ms, brute 17.305ms
~~~
### 3. Clustering (python)
  <center class="half">
    <img src="../point_cloud_processing/ch3_clustering/hw/figure/fig3.png" width="500"/> <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.3 Benchmark of clustering algorithm. <br>Left 1: Mykmeans; Left 2: MyGMM</div>
</center>

### 4. Ground Segmentation and Clustering (python)
  <center class="half">
      <img src="../point_cloud_processing/ch4_model_fitting/figure/fig1.png" width="500"/>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">Fig.4 Ground segmentation(RANSAC + Least square optimization). Blue part is the segmented ground</div>
  </center><br>
  <center>
      <img src="../point_cloud_processing/ch4_model_fitting/figure/fig4.png" width="500"/>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">Fig.5 DBSCAN clustering algorithm of the point clouds after aplying ground filter </div>
  </center><br>

### 5. Deep Learning for Classification (python)
  <center class="half">
    <img src="../point_cloud_processing/ch5_deep_learning/PointNet/figure/fig1.png" width="350"/><img src="../point_cloud_processing/ch5_deep_learning/PointNet/figure/fig2.png" width="350"/>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.6 left: loss diagramm right: accuracy diagramm </div>
</center>

### 6. Deep Learning for Object Detection (python)
<center class="half">
    <img src="../point_cloud_processing/ch6_object_detection/PointRCNN/output/rcnn/default/eval/epoch_no_number/val/final_result/plot/car_detection_3d.png" width="300"/>
    <img src="../point_cloud_processing/ch6_object_detection/PointRCNN/output/rcnn/default/eval/epoch_no_number/val/final_result/plot/car_detection_ground.png" width="300"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
</center>

<center class="half">
    <img src="../point_cloud_processing/ch6_object_detection/PointRCNN/output/rcnn/default/eval/epoch_no_number/val/final_result/plot/car_detection.png" width="300"/>
    <img src="../point_cloud_processing/ch6_object_detection/PointRCNN/output/rcnn/default/eval/epoch_no_number/val/final_result/plot/car_orientation.png" width="300"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.7 P-R Curve of car detection and orientation
</center>

### 7. Feature Detection (C++)
<center class="half">
    <img src="../point_cloud_processing/ch7_feature_detection/fig/fig1.png" width="300"/>
    <img src="../point_cloud_processing/ch7_feature_detection/fig/fig2.png" width="315"/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.8 Feature detection of gituar and airplane
</center>