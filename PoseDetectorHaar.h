

#include <highgui.h>
#include <cv.h>

#include <vector>

#include "PoseDetectorHaarParameters.h"

using namespace cv;
using namespace std;

namespace PoseDetectorHaar
{
	
	// if cx and cy are not set the will be interpreted as being at the centre of the image

	// output indicates success or failure	of detection
	void DetectPoseHaar(const Mat_<uchar>& intensity, Vec6d& o_poseEstimate, Matx66d& o_covariance, const Vec6d& i_poseEstimate, const Matx66d& i_covariance, float fx = 500, float fy = 500, float cx = -1, float cy = -1, const PoseDetectorHaarParameters& params = PoseDetectorHaarParameters());
	void DetectPoseHaar(const Mat_<uchar>& intensity, Vec6d& o_poseEstimate, Matx66d& o_covariance, const Vec6d& i_poseEstimate, const Matx66d& i_covariance, CascadeClassifier& classifier, float fx = 500, float fy = 500, float cx = -1, float cy = -1, const PoseDetectorHaarParameters& params = PoseDetectorHaarParameters());
	
	void DetectPoseHaar(const Mat_<uchar>& intensity, const Mat_<float>& depth, Vec6d& o_poseEstimate, Matx66d& o_covariance, const Vec6d& i_poseEstimate, const Matx66d& i_covariance, float fx = 500, float fy = 500, float cx = -1, float cy = -1, const PoseDetectorHaarParameters& params = PoseDetectorHaarParameters());
	void DetectPoseHaar(const Mat_<uchar>& intensity, const Mat_<float>& depth, Vec6d& o_poseEstimate, Matx66d& o_covariance, const Vec6d& i_poseEstimate, const Matx66d& i_covariance, CascadeClassifier& classifier, float fx = 500, float fy = 500, float cx = -1, float cy = -1, const PoseDetectorHaarParameters& params = PoseDetectorHaarParameters());	

	// Initialisation
	bool InitialisePosesHaar(const Mat_<uchar>& intensity, vector<Vec6d>& o_poseEstimates, vector<Matx66d>& o_covariances, vector<Rect>& o_regions, float fx = 500, float fy = 500, float cx = -1, float cy = -1, const PoseDetectorHaarParameters& params = PoseDetectorHaarParameters());
	bool InitialisePosesHaar(const Mat_<uchar>& intensity, vector<Vec6d>& o_poseEstimates, vector<Matx66d>& o_covariances, vector<Rect>& o_regions, CascadeClassifier& classifier, float fx = 500, float fy = 500, float cx = -1, float cy = -1, const PoseDetectorHaarParameters& params = PoseDetectorHaarParameters());
	
	bool InitialisePosesHaar(const Mat_<uchar>& intensity, const Mat_<float>& depth, vector<Vec6d>& o_poseEstimates, vector<Matx66d>& o_covariances, vector<Rect>& o_regions, float fx = 500, float fy = 500, float cx = -1, float cy = -1, const PoseDetectorHaarParameters& params = PoseDetectorHaarParameters());
	bool InitialisePosesHaar(const Mat_<uchar>& intensity, const Mat_<float>& depth, vector<Vec6d>& o_poseEstimates, vector<Matx66d>& o_covariances, vector<Rect>& o_regions, CascadeClassifier& classifier, float fx = 500, float fy = 500, float cx = -1, float cy = -1, const PoseDetectorHaarParameters& params = PoseDetectorHaarParameters());	

}

