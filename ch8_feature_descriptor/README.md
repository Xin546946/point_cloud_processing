# ch8. Feature Description

*  In this report, we implement a feature description algorithm, namely Fast Point Feature Histogram(FPFH). Due to the limited time, we have not tested the algorithm quantitively. The correctness will be tested in the next homework (Registration).

## FPFHEstimator
* The class is in reference of FPFH interface in PCL. The concrete implementation is in src/fpfh.cpp
~~~C++
class FPFHEstimator{
    public:
    FPFHEstimator() = default;
    void set_input_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr keys);
    void set_input_normal(pcl::PointCloud<pcl::Normal>::Ptr normals);
    void set_search_surface(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    void set_radius_search(float radius);
    void compute(std::vector<FPFHSignature33>& fpfh_descriptor);

    private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr keys_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
    std::vector<FPFHSignature33> fpfh_;
    float radius_;
};
~~~

* Besides, the priciple of computing descriptor for one point is a combinition of several Single Point Feature Histogram(SPFH). So two result set are needed, FPFHResultset is able to combine different histograms of key point and the neighbour of key points with weight. SPFHResultset is used to compute triplet feature and make a histogram. The header file is as follows.
~~~C++
class FPFHResultset{
public: 
    FPFHResultset(): histogram_(std::vector<float>(33,0.f)) {} 
    void add_histogram(std::vector<float> histogram, float weight); // compute histogram, set alpha_hist->phi_hist->theta_hist
    std::vector<float> get_histogram();

    private:
    std::vector<float> histogram_;

}; 
~~~
~~~C++
class SPFHResultset{
    public:
    SPFHResultset() = default;
    void set_triplet_features(std::vector<float> alpha_vec, std::vector<float> phi_vec, std::vector<float> theta_vec);
    std::vector<float> get_histogram();

    private:
    std::vector<float> compute_histogram_from_vector(std::vector<float> vec);
    // int get_bin_id(float value, float width_bin);
    std::vector<float> histogram_;
};
~~~
