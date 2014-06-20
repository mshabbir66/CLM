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

#include "CLMTracker.h"
#include "CLMParameters.h"

#include "highgui.h"
#include "cv.h"
#include <math.h>

using namespace CLMTracker;
using namespace CLMWrapper;
using namespace cv;

//=========================================================================== Utility from CLM (renamed to avoid name clash issues)
void AddOrthRow_redef(cv::Mat &R)
{
  assert((R.rows == 3) && (R.cols == 3));
  R.at<double>(2,0) = R.at<double>(0,1)*R.at<double>(1,2) - R.at<double>(0,2)*R.at<double>(1,1);
  R.at<double>(2,1) = R.at<double>(0,2)*R.at<double>(1,0) - R.at<double>(0,0)*R.at<double>(1,2);
  R.at<double>(2,2) = R.at<double>(0,0)*R.at<double>(1,1) - R.at<double>(0,1)*R.at<double>(1,0);
  return;
}

//=========================================================================== Utility from CLM (renamed to avoid name clash issues)
void Euler2Rot_redef(cv::Mat &R,const double pitch,const double yaw,const double roll,
	       bool full = true)
{
  if(full){if((R.rows != 3) || (R.cols != 3))R.create(3,3,CV_64F);}
  else{if((R.rows != 2) || (R.cols != 3))R.create(2,3,CV_64F);}
  double sina = sin(pitch), sinb = sin(yaw), sinc = sin(roll);
  double cosa = cos(pitch), cosb = cos(yaw), cosc = cos(roll);
  R.at<double>(0,0) = cosb*cosc; R.at<double>(0,1) = -cosb*sinc; R.at<double>(0,2) = sinb;
  R.at<double>(1,0) = cosa*sinc + sina*sinb*cosc;
  R.at<double>(1,1) = cosa*cosc - sina*sinb*sinc;
  R.at<double>(1,2) = -sina*cosb; if(full)AddOrthRow_redef(R); return;
}
//=========================================================================== Utility from CLM (renamed to avoid name clash issues)
void Euler2Rot_redef(cv::Mat &R,cv::Mat &p,bool full = true)
{
  assert((p.rows == 6) && (p.cols == 1));
  Euler2Rot_redef(R,p.at<double>(1,0),p.at<double>(2,0),p.at<double>(3,0),full); return;
}


Mat_<float> CLMWrapper::GetShape(TrackerCLM& clmModel, double fx, double fy, double cx, double cy, const CLMWrapper::CLMParameters& params)
{
	int n = clmModel._shape.rows/2;

	Mat shape3d(n*3, 1, CV_64F);

	clmModel._clm._pdm.CalcShape3D(shape3d, clmModel._clm._plocal, clmModel._clm._paramsMorph);
	
	// Need to rotate the shape to get the actual 3D representation
	
	// get the rotation matrix from the euler angles
	Mat R(3,3,CV_64F);
	Euler2Rot_redef(R,clmModel._clm._pglobl);

	shape3d = shape3d.reshape(1, 3);

	shape3d = shape3d.t() * R.t();
	
	// from the weak perspective model can determine the average depth of the object
	double Zavg = fx / clmModel._clm._pglobl.at<double>(0);
	
	Mat_<float> outShape(n,3,0.0);

	// this is described in the paper in section 3.4 (equation 10)
	for(int i = 0; i < n; i++)
	{
		double Z = Zavg + shape3d.at<double>(i,2);

		double X = Z * ((clmModel._shape.at<double>(i) - cx)/fx);
		double Y = Z * ((clmModel._shape.at<double>(i + n) - cy)/fy);

		outShape.at<float>(i,0) = (float)X;
		outShape.at<float>(i,1) = (float)Y;
		outShape.at<float>(i,2) = (float)Z;

	}

	// The format is 3 rows - v cols
	return outShape.t();
	
}

Vec6d CLMWrapper::GetPoseCLM(TrackerCLM& clmModel, double fx, double fy, double cx, double cy, CLMWrapper::CLMParameters& params)
{
	if(!clmModel._shape.empty() && clmModel._clm._pglobl.at<double>(0) != 0)
	{
		double Z = fx / clmModel._clm._pglobl.at<double>(0);
	
		double X = ((clmModel._clm._pglobl.at<double>(4) - cx) * (1.0/fx)) * Z;
		double Y = ((clmModel._clm._pglobl.at<double>(5) - cy) * (1.0/fy)) * Z;
	
		return Vec6d(X, Y, Z, clmModel._clm._pglobl.at<double>(1), clmModel._clm._pglobl.at<double>(2), clmModel._clm._pglobl.at<double>(3));
	}
	else
	{
		return Vec6d(1,0,0,0,0,0);
	}
}

void GetCentreOfMass(const Mat_<uchar>& mask, double& centreX, double& centreY, const Rect& roi)
{
    if(roi.width == 0)
    {
        Moments baseMoments = moments(mask, true);
        // center of mass x = m_10/m_00, y = m_01/m_00 if using image moments
        centreX = (baseMoments.m10/baseMoments.m00);
        centreY = (baseMoments.m01/baseMoments.m00);
    }
    else
    {
        Moments baseMoments = moments(mask(roi), true);
        // center of mass x = m_10/m_00, y = m_01/m_00 if using image moments

        centreX = (baseMoments.m10/baseMoments.m00) + roi.x;
        centreY = (baseMoments.m01/baseMoments.m00) + roi.y;
    }
}

Vec6d CLMWrapper::GetPoseCLM(TrackerCLM& clmModel, const cv::Mat_<float> &depth, double fx, double fy, double cx, double cy, CLMWrapper::CLMParameters& params)
{
	if(depth.empty())
	{
		return GetPoseCLM(clmModel, fx, fy, cx, cy, params);
	}
	else
	{
		
		double Z = fx / clmModel._clm._pglobl.at<double>(0);
	
		double X = ((clmModel._clm._pglobl.at<double>(4) - cx) * (1.0/fx)) * Z;
		double Y = ((clmModel._clm._pglobl.at<double>(5) - cy) * (1.0/fy)) * Z;	

		double tx = clmModel._clm._pglobl.at<double>(4);
		double ty = clmModel._clm._pglobl.at<double>(5);

		cv::Mat_<uchar> currentFrameMask = depth > 0;

		int width = (int)(140 * clmModel._clm._pglobl.at<double>(0));			
		int height = (int)(133 * clmModel._clm._pglobl.at<double>(0));			

		Rect roi((int)tx-width/2, (int)ty-width/2, width, height);

		// clamp the ROI
		roi.x = max(roi.x, 0);
		roi.y = max(roi.y, 0);

		roi.x = min(roi.x, depth.cols-1);
		roi.y = min(roi.y, depth.rows-1);

		Vec6d currPose(X, Y, Z, clmModel._clm._pglobl.at<double>(1), clmModel._clm._pglobl.at<double>(2), clmModel._clm._pglobl.at<double>(3));

		// deal with cases where the pose estimate is wildly off
		if(roi.width <= 0) 
		{
			roi.width = depth.cols;
			roi.x = 0;
		}
		if(roi.height <= 0)
		{
			roi.height = depth.rows;
			roi.y = 0;
		}

		if(roi.x + roi.width > depth.cols)
			roi.width = depth.cols - roi.x;
		if(roi.y + roi.height > depth.rows)
			roi.height = depth.rows - roi.y;

		if(sum(currentFrameMask(roi))[0] > 200)
		{
			// Calculate the centers of mass in the mask for a new pose estimate
			double centreX, centreY;
			GetCentreOfMass(currentFrameMask, centreX, centreY, roi);

			// the center of mass gives bad results when shoulder or ponytails are visible
			Z = mean(depth(Rect((int)centreX - 8, (int)centreY - 8, 16, 16)), currentFrameMask(Rect((int)centreX - 8, (int)centreY - 8, 16, 16)))[0] + 100; // Z offset from the surface of the face
			X  = (centreX - cx) * Z / fx;
			Y  = (centreY - cy) * Z / fy; 

			// redefine the pose around witch to sample (only if it's legal)
			if(Z != 100)
			{
				currPose[0] = X;
				currPose[1] = Y;
				currPose[2] = Z;
			}
		}
		return Vec6d(X, Y, Z, clmModel._clm._pglobl.at<double>(1), clmModel._clm._pglobl.at<double>(2), clmModel._clm._pglobl.at<double>(3));
	}
}

Mat CLMWrapper::PoseToGlobal(const Vec6d& pose, double fx, double fy, double cx, double cy)
{
	Mat globl(6,1, CV_64F);
	
	double a = fx / pose[2];

	double tx = a * pose[0] + cx;
	double ty = a * pose[1] + cy;

	globl.at<double>(0) = a;
	globl.at<double>(1) = pose[3];
	globl.at<double>(2) = pose[4];
	globl.at<double>(3) = pose[5];
	globl.at<double>(4) = tx;
	globl.at<double>(5) = ty;

	return globl;
}

bool CLMWrapper::InitialiseCLM(const Mat_<uchar> &image, TrackerCLM& clmModel, const Vec6d poseEstimate, const Rect region, double fx, double fy, double cx, double cy, CLMWrapper::CLMParameters& params)
{
	
	return InitialiseCLM(image, Mat_<float>(), clmModel, poseEstimate, region, fx, fy, cx, cy, params);
}

bool CLMWrapper::InitialiseCLM(const Mat_<uchar> &image, const Mat_<float> &depthImage, TrackerCLM& clmModel, const Vec6d poseEstimate, const Rect region, double fx, double fy, double cx, double cy, CLMWrapper::CLMParameters& params)
{
	// First read in the CLM model if it hasn't been read in yet
	if(clmModel._clm._plocal.empty())
	{
		clmModel.Read(params.defaultModelLoc);
	}
	
	// Convert the ROI to CLM size
	Vec3d roiHaar(region.x, region.y, region.width);
	
	// Using the Haar transformation that was learned previously
	if(region.width > 0)
	{
		Mat_<double> roiC =  Mat(roiHaar).t() * clmModel.transformROIHaar;		
		Rect roiCLM((int)(roiC.at<double>(0) + 0.5), (int)(roiC.at<double>(1) + 0.5), (int)(roiC.at<double>(2) + 0.5), (int)(roiC.at<double>(3) + 0.5));
		//Rect roiCLM = region;

		// calculate the local and global parameters from the generated 2D shape (mapping from the 2D to 3D because camera params are unknown)
		clmModel._clm._pdm.CalcParams(clmModel._clm._pglobl, roiCLM, clmModel._clm._plocal, clmModel._clm._paramsMorph);
				
		// Optional visualisation of where the points are converging from
		bool visi = false;
		if(visi)
		{
			Mat disp;
			cv::cvtColor(image, disp, CV_GRAY2RGB);
			clmModel._clm._pdm.Draw(disp, clmModel._clm._plocal, clmModel._clm._paramsMorph, clmModel._clm._pglobl);
			imshow("Initial estimate", disp);
			//cv::waitKey(0);
		}
		
	}
	else
	{
		clmModel._clm._pglobl = PoseToGlobal(poseEstimate, fx, fy, cx, cy);
	}

	// Fit the from the initial estimate
	bool success = clmModel._clm.Fit(image, depthImage, params.wSizeInit, params.nIter, params.clamp, params.fTol, params.useMorphology, params.sigma);
		
	// convert the fit to a 2D shape
	clmModel._clm._pdm.CalcShape2D(clmModel._shape, clmModel._clm._plocal, clmModel._clm._paramsMorph, clmModel._clm._pglobl);	
	
	// Check if the actual fit converged
	if(params.fcheck && success)
	{
		Vec3d orientation;
		orientation(0) = clmModel._clm._pglobl.at<double>(1);
		orientation(1) = clmModel._clm._pglobl.at<double>(2);
		orientation(2) = clmModel._clm._pglobl.at<double>(3);

		if(!clmModel._fcheck.Check(orientation, image, depthImage, clmModel._shape, params.decisionBoundary))
		{	
			// if convergence successful we can set the pose and uncertainty properly now
			clmModel._success = false;
			return false;
		}
		else
		{
			clmModel._success = true;
			return true;
		}
	}
	else
	{
		clmModel._success = success;
		return success;
	}
}

bool CLMWrapper::TrackCLM(const Mat_<uchar> &image, TrackerCLM& clmModel, const vector<Vec6d> posePriors, const vector<Matx66d> poseUncertainties, const Rect region, float fx, float fy, float cx, float cy, CLMWrapper::CLMParameters& params)
{
	return TrackCLM(image, cv::Mat_<float>(), clmModel, posePriors, poseUncertainties, region, fx, fy, cx, cy, params);
}

bool CLMWrapper::TrackCLM(const Mat_<uchar> &image, const Mat_<float> &depth, TrackerCLM& clmModel, const vector<Vec6d> posePriors, const vector<Matx66d> poseUncertainties, const Rect region, float fx, float fy, float cx, float cy, CLMWrapper::CLMParameters& params)
{

	// try reinitialising if the tracking has failed and outside observation is available
	if(clmModel._success == false && (posePriors.size() > 0 || region.width > 0))
	{		
		Mat initGlobal = clmModel._clm._pglobl.clone();
		Mat initMorph = clmModel._clm._paramsMorph.clone();
		Mat initLocal = clmModel._clm._plocal.clone();
		
		// try from 0 non-rigid params
		clmModel._clm._plocal.setTo(0);
		clmModel._clm._paramsMorph.setTo(0);
		
		// when reinitialising don't reinitialise the morphology, as we're only tracking and not sta
		bool useOld = params.useMorphology;
		params.useMorphology = false;

		// try reinitialising using an area of interest rectangle
		if(region.width > 0)
		{
			if(InitialiseCLM(image, depth, clmModel, Vec6d(), region, fx, fy, cx, cy, params))
			{
				return true;
			}
		}
		else
		{
			for(size_t i = 0; i < posePriors.size(); ++i)
			{
				bool goodEnough = (cv::trace(Mat(poseUncertainties[i]))[0]/6 < 1000);

				if(goodEnough && InitialiseCLM(image, depth, clmModel, posePriors[i], Rect(), fx, fy, cx, cy, params))
				{
					return true;
				}
			}
		}
		params.useMorphology = useOld;

		// if reinitialisation unsuccesful carry on with usual tracking
		clmModel._clm._pglobl = initGlobal;
		clmModel._clm._plocal = initLocal;
		clmModel._clm._paramsMorph = initMorph;
	}
	
	bool success = clmModel._clm.Fit(image, depth, params.wSizeCurrent, params.nIter, params.clamp, params.fTol, false, params.sigma);
		
	clmModel._clm._pdm.CalcShape2D(clmModel._shape, clmModel._clm._plocal, clmModel._clm._paramsMorph, clmModel._clm._pglobl);	
		
	if(!success)
	{
		clmModel._success = false;
		return false;
	}

	if(params.fcheck)
	{
		Vec3d orientation;
		orientation(0) = clmModel._clm._pglobl.at<double>(1);
		orientation(1) = clmModel._clm._pglobl.at<double>(2);
		orientation(2) = clmModel._clm._pglobl.at<double>(3);

		if(clmModel._fcheck.Check(orientation, image, depth, clmModel._shape, params.decisionBoundary))
		{
			clmModel._success = true;
			return true;
		}
		else
		{
			clmModel._success = false;
			return false;
		}
	}
	else
	{
		clmModel._success = true;
		return true;
	}
}