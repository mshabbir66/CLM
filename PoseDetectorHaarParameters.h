
#include <cv.h>

using namespace cv;

namespace PoseDetectorHaar
{

struct PoseDetectorHaarParameters
{
	double HighUncertaintyVal, LowUncertaintyVal;
	Matx66d LowCovariance; // very certain
	Matx66d HighCovariance; // very uncertain
	
	// Face parameters (used to estimate the depth of the object)
	double ObjectWidth;	

	// search window around the current pose
	double SearchWindowWidth, SearchWindowHeight;

	// offsets from the center of face to region of interest in milimeters (basically the bounding box of the object in 3D)
	double XLeftOffset, XRightOffset, YTopOffset, YBottomOffset, ZNearOffset, ZFarOffset;	

	// the offsets from the estimates based on width
	double XEstimateOffset, YEstimateOffset, ZEstimateOffset;

	// there might be cases where the offsets are different if depth is used to refine the tracking
	double XDepthEstimateOffset, YDepthEstimateOffset, ZDepthEstimateOffset;

	string ClassifierLocation;

	PoseDetectorHaarParameters()
	{
		HighUncertaintyVal = 1000;
		LowUncertaintyVal = 0.000001;
		LowCovariance = Matx66d::eye() * LowUncertaintyVal;
		HighCovariance = Matx66d::eye() * HighUncertaintyVal;

		ObjectWidth = 200;
	
		SearchWindowWidth = ObjectWidth * 2;
		SearchWindowHeight = ObjectWidth * 2;

		XLeftOffset = 300;
		XRightOffset = 300;
		YTopOffset = 200;
		YBottomOffset = 200;
		ZNearOffset = 300;
		ZFarOffset = 250;		

		XEstimateOffset = 0;
		YEstimateOffset = 0;
		ZEstimateOffset = 0;

		XDepthEstimateOffset = 0;
		YDepthEstimateOffset = 0;
		ZDepthEstimateOffset = 100;

		ClassifierLocation = "..\\lib\\3rdParty\\OpenCV\\classifiers\\haarcascade_frontalface_default.xml";
	}

};

}
