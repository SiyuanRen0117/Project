#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <chrono>   

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	
	//Read images
	Mat img_1 = imread("/media/ren/Seagate/KITTI360_DATASET/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000000.png", CV_LOAD_IMAGE_COLOR);
	Mat img_2 = imread("/media/ren/Seagate/KITTI360_DATASET/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000001.png", CV_LOAD_IMAGE_COLOR);
	assert(img_1.data != nullptr && img_2.data != nullptr);

	//Initialize
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<FeatureDetector> descriptor = ORB::create();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	//detect oriented FAST Features
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	//calculate BRIEF descriptor based on the position of corner
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	cout << "extract ORB cost = " << time_used.count() << "seconds." << endl;

	Mat outimg1;
	drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("ORB features", outimg1);

	//Using Hamming distance to matching BRIEF descriptors from two images 
	vector<DMatch> matches;
	t1 = chrono::steady_clock::now();
	matcher->match(descriptors_1, descriptors_2, matches);
	t2 = chrono::steady_clock::now();
	time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	cout << "match ORB cost = " << time_used.count() << "seconds." << endl;

	//calculate the shortest distance and longest distance and choose the matching points
	auto min_max = minmax_element(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
	double min_dist = min_max.first->distance;
	double max_dist = min_max.second->distance;

	// printf("-- Max dist : &f \n", max_dist);
	// printf("-- Min dist : &f \n", min_dist);

	//Wrong distance between desciptors, set a minimal boundary.
	std::vector<DMatch> good_matches;
	for (int i = 0; i < descriptors_1.rows; ++i) {
		if (matches[i].distance <= max(2 * min_dist, 30.0)) {
			good_matches.push_back(matches[i]);
		}
	}

	//PLot the images with matched point
	Mat img_match;
	Mat img_goodmatch;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
	imshow("all matches", img_match);
	imwrite("../Result/img_match_1.jpg", img_match);
	imshow("good matches", img_goodmatch);
	imwrite("../Result/img_goodmatch_1.jpg", img_goodmatch);
	waitKey(0);

	return 0;
}