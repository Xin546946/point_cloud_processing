## Introduction
---------------

1. **PCA**
   ~~~ python
   def PCA(data, correlation=False, sort=True):
        # 作业1
        # 屏蔽开始
        if correlation:
            corr_data = np.corrcoef(data.T )
            eigenvectors, eigenvalues, _ = np.linalg.svd(corr_data)        

        else:
            cov_data = np.cov(data.T)
            eigenvectors, eigenvalues, _ = np.linalg.svd(cov_data)

        eigenvalues = np.sqrt(eigenvalues)
        # 屏蔽结束

        if sort:
            sort = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[sort]
            eigenvectors = eigenvectors[:, sort]

        return eigenvalues, eigenvectors
   ~~~

* using covariance，the visualization is shown below
  $$
  cov(X,Y) = \sum_{i=1}^{N} \frac{(X_i-\overline{X})(Y_i-\overline{Y})^T}{N -1}
  $$
<br>
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./figures/figure1.png";class ="half">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig1. pca of point cloud, red line indicates the main direction and green the second</div>
</center>
<br>

* if using correlation coefficients
  $$
  \rho(X,Y) = \frac{cov(X,Y)}{\sigma_X\sigma_Y}
  $$
<br>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./figures/2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig2. pca of point cloud, red line indicates the main direction and green the second</div>
</center>
<br>

2. **Points' Normals**
    ~~~python
    # 作业2
    # 屏蔽开始
    for i in range(points.shape[0]):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 20)
        knn_points = np.asarray(point_cloud_o3d.points)[idx, :]
        _, v_knn_points = PCA(knn_points)
        normals.append(v_knn_points[:,-1])

    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数
    # 屏蔽结束
    ~~~
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./figures/normals.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig3. Point cloud with the normals</div>
</center>
<br>

3. **Voxel Downsampling**
    ~~~python
    def hash_conflict(a,b):
        if(a[1] != b[1] or a[2] != b[2] or a[3] != b[3]):
            return True
        return False

    def voxel_filter(point_cloud, leaf_size, random_sampling = False):
        filtered_points = []
        # 作业3
        # 屏蔽开始
        point_cloud = np.asarray(point_cloud)
        print("total points : ", len(point_cloud))
        x_max, y_max, z_max = np.max(point_cloud,axis = 0)
        x_min, y_min, z_min = np.min(point_cloud,axis = 0)
    
        Dx = (x_max - x_min) // leaf_size
        Dy = (y_max - y_min) // leaf_size
        Dz = (z_max - z_min) // leaf_size
        container_size = Dx * Dy * Dz 
    
        h = list() 
        for i in range (point_cloud.shape[0]):
            x, y, z = point_cloud[i]
            hx = np.floor((x - x_min) / leaf_size)
            hy = np.floor((y - y_min) / leaf_size)
            hz = np.floor((z - z_min) / leaf_size)
            h.append([(hx + hy * Dx + hz * Dx * Dy) % container_size, hx, hy, hz, i])#storing pair
        h = np.asarray(h)
        h_index = np.lexsort((h[:,0], h[:,1], h[:,2], h[:,3]))
        H = list()
        for i in range(len(h_index)):
            H.append(h[h_index[i]])
        #now H stores all points, they are all sorted according to different dimensions(h,hx,hy,hz)
    
        filtered_points = list()
        cur_voxel = list()
        cur_voxel.append(point_cloud[int(H[0][4])])#first point can't be conflicted by definition. also avoiding empty cur_voxel
    
        for i in range(1 , len(h)):
            if (H[i][0] == H[i-1][0] and not hash_conflict(H[i],H[i-1])):
                #put point if it is not conflicted and have same voxel index.
                cur_voxel.append(point_cloud[int(H[i][4])])
            else:
                #otherwise pick sample point from currect voxel then clear it and add this point into it 
                if(random_sampling == False):
                    [x_c, y_c, z_c] = np.mean(np.asarray(cur_voxel),axis = 0)
                else:
                    index = np.random.choice(np.asarray(cur_voxel).shape[0], 1)
                    [x_c, y_c, z_c] = cur_voxel[index[0]]
                filtered_points.append([x_c,y_c,z_c])
                cur_voxel.clear()
                cur_voxel.append(point_cloud[int(H[i][4])])
        # 屏蔽结束
        # 把点云格式改成array，并对外返回
        filtered_points = np.array(filtered_points, dtype=np.float64)
        print("sample points : ",len(filtered_points))
        return filtered_points


* setting leaf_size to 100.0 results in 66 points(60 points if hash conflict not solved), roughly represents the object
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./figures/66.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig4. Point cloud after voxel down sampling, picking by centroid, 66 points in total</div>
</center>
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./figures/66_random.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig5. Point cloud after voxel down sampling, picking by random, 66 points in total</div>
</center>
<br>

* setting leaf_size to 10.0 results in 1922 points, almost reconstructs whole point cloud
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./figures/1922.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig6. Point cloud after voxel down sampling, 1922 points in total</div>
</center>
<br>

* setting leaf_size to 0.1 results in 4414 points, further amplifiying resolution doesn't add as many points, 0.01 results in 4419 points for example, only make a minor difference with 5 points.
* The reason behind it might be that, the remaining 7000ish points are too close to their neighbors to be seperated.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./figures/4414.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig7. Point cloud after voxel down sampling, 4414 points in total</div>
</center>

<br>

* How to  solve hash conflict:
    Sort points not only according to voxel index, but also according to hx,hy,hz. This way among the points whose voxel index are the same, hash conflicted points are segmented, thus every segment can be treated as points with a new voxel index(which is the same as the one of those not conflicted points but since they are already cleared so it's "new").
<br>  


4. **Upsampling**
    ~~~c++
    double computeGaussian(int sigma_squared, int x_squared){
        double res = (1 / sqrt(2 * M_PI * sigma_squared)) * exp(-x_squared / (2 * sigma_squared));
        return res;
    }

    void apply_filter(const cv::Mat& image, cv::Mat des, int halfFilterSize, int sigma1, int sigma2){//give float img
        if(image.type() != CV_32F){
            std::cout<<"please pass float images as input!";
            return;
        }
        int width = image.cols;
        int height = image.rows;
        for(int r = 0; r < height; r++){
            for(int c = 0; c < width; c++){
                if(image.at<float>(r, c) != 0.0) continue;
                 //apply double gaussian
                double sum1 = 0;
                double sum2 = 0;
                for(int r_win = -halfFilterSize; r_win <= halfFilterSize; r_win++){
                    for(int c_win = -halfFilterSize; c_win <= halfFilterSize; c_win++){
                        if(r + r_win < 0 || c + c_win < 0 || r + r_win >= height || c + c_win >= width || image.at<float>(r+r_win, c+c_win) == 0.0) continue;
                        int posDistSquared = pow(r_win, 2) + pow(c_win, 2);
                        int depthDistSquared = pow(image.at<float>(r,c) - image.at<float>(r + r_win, c + c_win), 2);
                        sum1 += computeGaussian(sigma1, posDistSquared) * computeGaussian(sigma2, depthDistSquared) * image.at<float>(r + r_win , c + c_win);
                        sum2 += computeGaussian(sigma1, posDistSquared) * computeGaussian(sigma2, depthDistSquared);
                    }
                }
                des.at<float>(r,c) = sum1 / sum2;
            }
        }
    }
    ~~~
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./figures/jet.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig8. Depth image up sampled(above) and groud truth(bottom)</div>
</center>
mean mae: 1.401772 <br>
min  mae: 0.910021 <br>
max  mae: 2.224899 <br>
mean rmse: 3.263513 <br>
min  rmse: 1.000000 <br>
max  rmse: 5.648661 <br>
mean inverse mae: 0.007154 <br>
min  inverse mae: 0.005191 <br>
max  inverse mae: 0.011504 <br>
mean inverse rmse: 0.011411 <br>
min  inverse rmse: 0.007827 <br>
max  inverse rmse: 0.019368 <br>
mean log mae: 0.082108 <br>
min  log mae: 0.063105 <br>
max  log mae: 0.108946 <br>
mean log rmse: 0.135744 <br>
min  log rmse: 0.086491 <br>
max  log rmse: 0.234225 <br>
mean scale invariant log: 0.108467 <br>
min  scale invariant log: 0.057052 <br>
max  scale invariant log: 0.210251 <br>
mean abs relative: 0.075002 <br>
min  abs relative: 0.059773 <br>
max  abs relative: 0.091497 <br>
mean squared relative: 0.014147 <br>
min  squared relative: 0.006137 <br>
max  squared relative: 0.032267<br>

<br>
authored by : Guanzhi Feng