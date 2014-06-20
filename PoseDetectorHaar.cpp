///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2012, Tadas Baltrusaitis, all rights reserved.
//
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
//
//     * The software is provided under the terms of this licence stricly for
//       academic, non-commercial, not-for-profit purposes.
//     * Redistributions of source code must retain the above copyright notice, 
//       this list of conditions (licence) and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright 
//       notice, this list of conditions (licence) and the following disclaimer 
//       in the documentation and/or other materials provided with the 
//       distribution.
//     * The name of the author may not be used to endorse or promote products 
//       derived from this software without specific prior written permission.
//     * As this software depends on other libraries, the user must adhere to 
//       and keep in place any licencing terms of those libraries.
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite the following work:
//
//       Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe Morency. 3D
//       Constrained Local Model for Rigid and Non-Rigid Facial Tracking.
//       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.    
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////

#include "PoseDetectorHaar.h"

namespace PoseDetectorHaar
{
	
	Rect GenerateRoiFromPose(const Vec6d& pose, const Matx66d& uncertainty, float fx, float fy, float cx, float cy, Size imgSize, double XLeftOffset, double XRightOffset, double YTopOffset, double YBottomOffset, double varianceScaling = 0.0001);

	// If no classifier specified
	void DetectPoseHaar(const Mat_<uchar>& intensity, Vec6d& o_poseEstimate, Matx66d& o_covariance, const Vec6d& i_poseEstimate, const Matx66d& i_covariance, float fx, float fy, float cx, float cy, const PoseDetectorHaarParameters& params)
	{
		
		CascadeClassifier classifier(params.ClassifierLocation);
		DetectPoseHaar(intensity, o_poseEstimate, o_covariance, i_poseEstimate, i_covariance, classifier,fx, fy, cx, cy, params);

	}

	void DetectPoseHaar(const Mat_<uchar>& intensity, const Mat_<float>& depth, Vec6d& o_poseEstimate, Matx66d& o_covariance, const Vec6d& i_poseEstimate, const Matx66d& i_covariance, float fx, float fy, float cx, float cy, const PoseDetectorHaarParameters& params)
	{
		
		CascadeClassifier classifier(params.ClassifierLocation);
		DetectPoseHaar(intensity, depth, o_poseEstimate, o_covariance, i_poseEstimate, i_covariance, classifier, fx, fy, cx, cy, params);

	}

	void DetectPoseHaar(const Mat_<uchar>& intensity, Vec6d& o_poseEstimate, Matx66d& o_covariance, const Vec6d& i_poseEstimate, const Matx66d& i_covariance, CascadeClassifier& classifier, float fx, float fy, float cx, float cy, const PoseDetectorHaarParameters& params)
	{

		if(cx == -1 || cy == -1)
		{
			cx = (float)intensity.cols /2;
			cy = (float)intensity.rows /2;
		}

		// Use the previous pose estimate to reduce the number of false positives by only searching in a specific region
		vector<Rect> regions;

		Rect roi = GenerateRoiFromPose(i_poseEstimate, i_covariance, fx, fy, cx, cy, intensity.size(), params.XLeftOffset, params.XRightOffset, params.YTopOffset, params.YBottomOffset);

		classifier.detectMultiScale(intensity(roi), regions, 1.2,2,CV_HAAR_DO_CANNY_PRUNING, Size(40, 40)); 

		// correct the regions
		for(size_t i = 0; i < regions.size(); ++i)
		{
			regions[i].x += roi.x;
			regions[i].y += roi.y;
		}
		
		// More confidence in X, Y, than in Z, and more than rotation (ad-hoc) (only trust only Y basically, and even then not much)
		// as X tends to drift when face is rotated (should ideally learn these from a larger dataset)
		o_covariance = params.HighCovariance;
		
		// Initialise
		Vec6d currentEstimate(0.0);

		// Pick the closest of the regions as the output
		for(size_t i = 0; i < regions.size(); ++i)
		{

			double Z = params.ObjectWidth * fx / regions[i].width;
			
			double centerX = regions[i].x + regions[i].width / 2.0;
			double centerY = regions[i].y + regions[i].height / 2.0;
	
			double X  = (centerX - cx) * Z / fx + params.XEstimateOffset;
			double Y  = (centerY - cy) * Z / fy + params.YEstimateOffset;
	
			Z = Z + params.ZEstimateOffset;

			// No rotation estimates from Haar classifier
			Vec6d poseEstimate(X, Y, Z, 0, 0, 0);
								
			// if the current prediction is better pick that one
			if(i == 0 || cv::norm(poseEstimate, i_poseEstimate) < cv::norm(currentEstimate, i_poseEstimate))
			{
				currentEstimate = poseEstimate;
				o_covariance(1,1) = 20;
			}
			

		}
	}

	void DetectPoseHaar(const Mat_<uchar>& intensity, const Mat_<float>& depth, Vec6d& o_poseEstimate, Matx66d& o_covariance, const Vec6d& i_poseEstimate, const Matx66d& i_covariance, CascadeClassifier& classifier, float fx, float fy, float cx, float cy, const PoseDetectorHaarParameters& params)
	{

		if(cx == -1 || cy == -1)
		{
			cx = intensity.cols /2.0f;
			cy = intensity.rows /2.0f;
		}

		// Use the previous pose estimate to reduce the number of false positives by only searching in a specific region
		vector<Rect> regions;

		Rect roi;
		if(cv::trace(Mat(i_covariance))[0]/6 < params.HighUncertaintyVal)
		{
			roi = GenerateRoiFromPose(i_poseEstimate, i_covariance, fx, fy, cx, cy, intensity.size(), params.XLeftOffset, params.XRightOffset, params.YTopOffset, params.YBottomOffset);
		}
		else
		{
			roi = Rect(0,0,intensity.cols, intensity.rows);
		}
		classifier.detectMultiScale(intensity(roi), regions, 1.2,2,CV_HAAR_DO_CANNY_PRUNING, Size(40, 40)); 
		for(size_t i = 0; i < regions.size(); ++i)
		{
			regions[i].x += roi.x;
			regions[i].y += roi.y;
		}
		
		// More confidence in X, Y, than in Z, and more than rotation (ad-hoc) (only trust only Y basically, and even then not much)
		// In the end decided not to actually use the haar detection in the Kalman update due to unreliability
		// as X tends to drift when face is rotated (should ideally learn these from a larger dataset)
		o_covariance = params.HighCovariance;
		
		// Initialise
		Vec6d currentEstimate;
		
		// OpenCV classifier does not have any orientation estimates but we can try estimating the depth based on average face size and the width of detected region
		for(size_t i = 0; i < regions.size(); ++i)
		{			

			double Z = params.ObjectWidth * fx / regions[i].width;
			
			double centerX = regions[i].x + regions[i].width / 2.0;
			double centerY = regions[i].y + regions[i].height / 2.0;
	
			double X  = (centerX - cx) * Z / fx + params.XEstimateOffset;
			double Y  = (centerY - cy) * Z / fy + params.YEstimateOffset;
	
			// No rotation estimates from openCV
			Vec6d poseEstimate(X, Y, Z, 0, 0, 0);						
				
			// only trust Y estimate
			o_covariance(1,1) = 20;
			
			if(!depth.empty())
			{
				// it is possible to refine the estimate slightly using the depth image by calculating the centers of mass (if the object to detect track is symmetric)
			
				// First get the depth estimate in the center of the roi
				Mat_<float> depthSmall = depth(Rect((int)centerX - 8, (int)centerY - 8, 16, 16));
				Z = mean(depthSmall, depthSmall > 0)[0]; // don't use illegal depth for estimate
				
				
				// Now can create a bounding box around that center of x
				Rect boundingBox = GenerateRoiFromPose(poseEstimate, params.LowCovariance, fx, fy, cx, cy, depth.size(), params.XLeftOffset, params.XRightOffset, params.YTopOffset, params.YBottomOffset);
			
				// create a mask around Z and center estimates
				Mat_<uchar> mask(depth.rows, depth.cols,(uchar)0);
				
				Mat dRoi = depth(boundingBox);
				Mat mRoi = mask(boundingBox);

				inRange(dRoi, Z - params.ZNearOffset, Z + params.ZFarOffset, mRoi);
			
				// Set the mask to 1 instead of 255 as per default openCV behaviour
				mask = mask / 255;

				// Only redefine the centres if there is enough depth information
				if(sum(mask)[0] > 200)
				{

					Z = mean(depthSmall, mask(Rect((int)centerX - 8, (int)centerY - 8, 16, 16)))[0] + params.ZDepthEstimateOffset;

					X  = (centerX - cx) * Z / fx + params.XEstimateOffset;
					Y  = (centerY - cy) * Z / fy + params.YEstimateOffset;

					// redefine the pose
					poseEstimate = Vec6d(X, Y, Z, 0, 0, 0);
					
					o_covariance(0,0) = 20;
					o_covariance(1,1) = 20;
					o_covariance(2,2) = 20;
				}		
			}
			// if the current prediction is better pick that one (always pick the first one though as well)
			if(i == 0 || cv::norm(poseEstimate, i_poseEstimate) < cv::norm(currentEstimate, i_poseEstimate))
			{
				currentEstimate = poseEstimate;
			}

		}
		o_poseEstimate = currentEstimate;
	}

	bool InitialisePosesHaar(const Mat_<uchar>& intensity, vector<Vec6d>& o_poseEstimates, vector<Matx66d>& o_covariances, vector<Rect>& o_regions, float fx, float fy, float cx, float cy, const PoseDetectorHaarParameters& params)
	{
		CascadeClassifier classifier("../../lib/3rdParty/OpenCV/classifier/haarcascade_frontalface_alt_old.xml");
		if(classifier.empty())
		{
			cout << "Couldn't load the Haar cascade classifier" << endl;
			return false;
		}
		else
		{
			return InitialisePosesHaar(intensity, o_poseEstimates, o_covariances, o_regions, classifier,fx, fy, cx, cy, params);
		}

	}

	bool InitialisePosesHaar(const Mat_<uchar>& intensity, vector<Vec6d>& o_poseEstimates, vector<Matx66d>& o_covariances, vector<Rect>& o_regions, CascadeClassifier& classifier, float fx, float fy, float cx, float cy, const PoseDetectorHaarParameters& params)
	{
		if(cx == -1 || cy == -1)
		{
			cx = intensity.cols /2.0f;
			cy = intensity.rows /2.0f;
		}

		o_poseEstimates.clear();
		o_covariances.clear();

		classifier.detectMultiScale(intensity, o_regions, 1.2,2,CV_HAAR_DO_CANNY_PRUNING, Size(40, 40)); 

		// OpenCV classifier does not have any orientation estimates but we can try estimating the depth based on average face size and the width of detected region
		for(size_t i = 0; i < o_regions.size(); ++i)
		{			

			double Z = params.ObjectWidth * fx / o_regions[i].width;
			
			double centerX = o_regions[i].x + o_regions[i].width / 2.0;
			double centerY = o_regions[i].y + o_regions[i].height / 2.0;
	
			double X  = (centerX - cx) * Z / fx + params.XEstimateOffset;
			double Y  = (centerY - cy) * Z / fy + params.YEstimateOffset;
	
			// No rotation estimates from openCV
			Vec6d poseEstimate(X, Y, Z, 0, 0, 0);						
			
			o_poseEstimates.push_back(poseEstimate);
			o_covariances.push_back(params.LowCovariance);

		}

		return o_regions.size() > 0;
	}
	
	bool InitialisePosesHaar(const Mat_<uchar>& intensity, const Mat_<float>& depth, vector<Vec6d>& o_poseEstimates, vector<Matx66d>& o_covariances, vector<Rect>& o_regions, float fx, float fy, float cx, float cy, const PoseDetectorHaarParameters& params)
	{
		CascadeClassifier classifier(params.ClassifierLocation);	
		if(classifier.empty())
		{
			cout << "Couldn't load the Haar cascade classifier" << endl;
			return false;
		}
		else
		{
			return InitialisePosesHaar(intensity, depth, o_poseEstimates, o_covariances, o_regions, classifier,fx, fy, cx, cy, params);
		}
	}

	bool InitialisePosesHaar(const Mat_<uchar>& intensity, const Mat_<float>& depth, vector<Vec6d>& o_poseEstimates, vector<Matx66d>& o_covariances, vector<Rect>& o_regions, CascadeClassifier& classifier, float fx, float fy, float cx, float cy, const PoseDetectorHaarParameters& params)
	{
		if(cx == -1 || cy == -1)
		{
			cx = intensity.cols /2.0f;
			cy = intensity.rows /2.0f;
		}

		o_poseEstimates.clear();
		o_covariances.clear();

		classifier.detectMultiScale(intensity, o_regions, 1.2,2,CV_HAAR_DO_CANNY_PRUNING, Size(40, 40)); 

		// OpenCV classifier does not have any orientation estimates but we can try estimating the depth based on average face size and the width of detected region
		for(size_t i = 0; i < o_regions.size(); ++i)
		{			

			double Z = params.ObjectWidth * fx / o_regions[i].width;
			
			double centerX = o_regions[i].x + o_regions[i].width / 2.0;
			double centerY = o_regions[i].y + o_regions[i].height / 2.0;
	
			double X  = (centerX - cx) * Z / fx + params.XEstimateOffset;
			double Y  = (centerY - cy) * Z / fy + params.YEstimateOffset;
	
			// No rotation estimates from openCV
			Vec6d poseEstimate(X, Y, Z, 0, 0, 0);						
			
			if(!depth.empty())
			{
				// First get the depth estimate in the center of the roi
				Mat_<float> depthSmall = depth(Rect((int)centerX - 8, (int)centerY - 8, 16, 16));
				Z = mean(depthSmall)[0];

				// Now can create a bounding box around that center of x
				Rect boundingBox = GenerateRoiFromPose(poseEstimate, params.LowCovariance, fx, fy, cx, cy, depth.size(), params.XLeftOffset, params.XRightOffset, params.YTopOffset, params.YBottomOffset);
			
				// create a mask around Z and center estimates
				Mat_<uchar> mask(depth.rows, depth.cols,(uchar)0);
				
				Mat dRoi = depth(boundingBox);
				Mat mRoi = mask(boundingBox);

				inRange(dRoi, Z - params.ZNearOffset, Z + params.ZFarOffset, mRoi);
			
				// Set the mask to 1 instead of 255 as per default openCV behaviour
				mask = mask / 255;

				// Only redefine the centres if there is enough depth information
				if(sum(mask)[0] > 200)
				{

					Z = mean(depthSmall, mask(Rect((int)centerX - 8, (int)centerY - 8, 16, 16)))[0];
					Z = Z + params.ZDepthEstimateOffset;
					
					X  = (centerX - cx) * Z / fx + params.XDepthEstimateOffset;
					Y  = (centerY - cy) * Z / fy + params.YDepthEstimateOffset; 

					// redefine the pose
					poseEstimate = Vec6d(X, Y, Z, 0, 0, 0);

				}
			}
			o_poseEstimates.push_back(poseEstimate);
			o_covariances.push_back(params.LowCovariance);

		}

		return o_regions.size() > 0;
	}
	
	// ROI is expanded based on the uncertainty
	Rect GenerateRoiFromPose(const Vec6d& pose, const Matx66d& uncertainty, float fx, float fy, float cx, float cy, Size imgSize, double XLeftOffset, double XRightOffset, double YTopOffset, double YBottomOffset, double varianceScaling )
	{
		if(cv::trace(Mat(uncertainty))[0]/6 > 50)
		{
			Rect roi;

			roi.width = imgSize.width;
			roi.x = 0;		
			roi.height = imgSize.height;
			roi.y = 0;

			return roi;

		}
		else
		{
			// first project the pose to the image
			double centerx, centery;

			centerx = (fx * pose[0])/pose[2] + cx;
			centery = (fy * pose[1])/pose[2] + cy;
		
			double Z = pose[2];

			// region depends on the current pose location
			double leftOffset = fx * XLeftOffset/Z;
			double rightOffset = fx * XRightOffset/Z;
			double topOffset = fx * YTopOffset/Z + fx;
			double bottomOffset = fx * YBottomOffset/Z;

			Rect roi((int)(centerx - leftOffset + 0.5), (int)(centery - topOffset + 0.5), (int)(leftOffset + rightOffset + 0.5), (int)(topOffset + bottomOffset + 0.5));

			// clamp the ROI
			roi.x = max(roi.x, 0);
			roi.y = max(roi.y, 0);
			if(roi.x + roi.width > imgSize.width)
				roi.width = imgSize.width - roi.x;
			if(roi.y + roi.height > imgSize.height)
				roi.height = imgSize.height - roi.y;

			// deal with cases where the pose estimate is wildly off
			if(roi.width == 0) 
			{
				roi.width = imgSize.width;
				roi.x = 0;
			}
			if(roi.height == 0)
			{
				roi.height = imgSize.height;
				roi.y = 0;
			}

			return roi;
		}
	}

}
