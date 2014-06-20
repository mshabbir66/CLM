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


//  Header for all external CLM methods of interest to the user
//
//
//  Tadas Baltrusaitis
//  01/05/2012

#ifndef __CLM_TRACKER_h_
#define __CLM_TRACKER_h_

#include "CLMParameters.h"

#include <TrackerCLM.h>

#include <cv.h>

#include <iostream>

using namespace std;
using namespace cv;
using namespace CLMTracker;

namespace CLMWrapper
{

	bool InitialiseCLM(const Mat_<uchar> &image, TrackerCLM& clmModel, const Vec6d poseEstimate, const Rect region, double fx, double fy, double cx, double cy, CLMWrapper::CLMParameters& params);
	bool InitialiseCLM(const Mat_<uchar> &image, const Mat_<float> &depth, TrackerCLM& clmModel, const Vec6d poseEstimate, const Rect region, double fx, double fy, double cx, double cy, CLMWrapper::CLMParameters& params);

	bool TrackCLM(const Mat_<uchar> &image, TrackerCLM& clmModel, const vector<Vec6d> posePriors, const vector<Matx66d> poseUncertainties, const Rect region, float fx, float fy, float cx, float cy, CLMWrapper::CLMParameters& params);
	bool TrackCLM(const Mat_<uchar> &image, const Mat_<float> &depth, TrackerCLM& clmModel, const vector<Vec6d> posePriors, const vector<Matx66d> poseUncertainties, const Rect region, float fx, float fy, float cx, float cy, CLMWrapper::CLMParameters& params);

	Vec6d GetPoseCLM(TrackerCLM& clmModel, double fx, double fy, double cx, double cy, CLMWrapper::CLMParameters& params);
	Vec6d GetPoseCLM(TrackerCLM& clmModel, const Mat_<float> &depth, double fx, double fy, double cx, double cy, CLMWrapper::CLMParameters& params);

	// Gets the shape of the current instance based on the average depth of the plane (already rotated)
	Mat_<float> GetShape(TrackerCLM& clmModel, double fx, double fy, double cx, double cy, const CLMWrapper::CLMParameters& params);

	// Conversion of rectangle to pose and vice versa
	Mat PoseToGlobal(const Vec6d& pose, double fx, double fy, double cx, double cy);	

	//===========================================================================
}
#endif
