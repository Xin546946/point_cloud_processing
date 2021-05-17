#include "ransac.h"
#include "fpfh.h"
#include "spfh.h"
#include <math.h>
#include <vector>

#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

void find_correspondence(
    const std::vector<FPFHSignature33> &source_descriptor,
    const std::vector<FPFHSignature33> &target_descriptor,
    std::vector<std::pair<int, int>> &correspondences_idx) {
  // make sure source descriptor and target descriptor has the same size
  assert(source_descriptor.size() == target_descriptor.size());
  std::pair<int, int> correspondence_candidate;
  for (int i = 0; i < source_descriptor.size(); i++) {
    float min_dist_source_to_target = std::numeric_limits<float>::max();
    correspondence_candidate.first = i;
    for (int j = 0; j < source_descriptor.size(); j++) {
      source_to_target_dist =
          compute_square_dist(source_descriptor[i], target_descriptor[j]);
      if (source_to_target_dist < min_dist_source_to_target) {
        correspondence_candidate.second = j;
        min_dist_source_to_target = source_to_target_dist;
      }
    }
    correspondences_idx.push_back(correspondence_candidate);
  }
}

// RANSAC
/**
 * 1. Choose four correspondence points
 * 2.
 *
 **/
