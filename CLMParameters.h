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

//  Parameters of the CLM-Z and CLM trackers
//
//  Tadas Baltrusaitis
//  01/05/2012

#ifndef __CLM_PARAM_WRAP_H
#define __CLM_PARAM_WRAP_H

#include "cv.h"
#include <highgui.h>

#include <vector>
#include <iostream>

using namespace cv;
using namespace std;
	
namespace CLMWrapper
{
	
struct CLMParameters
{

	double HighUncertaintyVal, LowUncertaintyVal;
	Matx66d LowCovariance; // very certain
	Matx66d HighCovariance; // very uncertain

	int nIter;
	double clamp, fTol; 
	bool fcheck;

	vector<int> wSizeSmall;
	vector<int> wSizeLarge;
	vector<int> wSizeVLarge;
	vector<int> wSizeInit;

	vector<int> wSizeCurrent;

	string defaultModelLoc;

	// offsets from the center of face to region of interest in milimeters (used by the tracker to convert the prior to region of interest estimate
	double XLeftOffset, XRightOffset, YTopOffset, YBottomOffset;

	double scaleToDepthFactor;
	double objectSizeFactor;

	double objectWidth;

	double Xoffset;
	double Yoffset;
	double Zoffset;

	bool useMorphology;

	// this is used for the smooting of response maps (KDE sigma)
	double sigma;

	// SVM boundary for face checking
	double decisionBoundary;

	CLMParameters()
	{
		HighUncertaintyVal = 1000000;
		LowUncertaintyVal = 0.000001;
		LowCovariance = Matx66d::eye() * LowUncertaintyVal;
		HighCovariance = Matx66d::eye() * HighUncertaintyVal;

		// number of iterations that will be performed at each clm scale
		nIter = 5;

		// how many standard deviations from should be clamped for morphology and expression
		clamp = 3;

		// the tolerance for convergence
		fTol = 0.01;

		// using an external face checker based on SVM
		fcheck = true;

		wSizeSmall = vector<int>(2);
		wSizeLarge = vector<int>(3);
		wSizeVLarge = vector<int>(4);
		wSizeInit = vector<int>(4);

		wSizeSmall[0] = 11;
		wSizeSmall[1] = 11;
		wSizeLarge.at(0) = 11;
		wSizeLarge.at(1) = 11;
		wSizeLarge.at(2) = 11;
		
		wSizeVLarge.at(0) = 11;
		wSizeVLarge.at(1) = 11;
		wSizeVLarge.at(2) = 11;
		wSizeVLarge.at(3) = 11;

		wSizeInit.at(0) = 21;
		wSizeInit.at(1) = 21;
		wSizeInit.at(2) = 21;
		wSizeInit.at(3) = 21;

		wSizeCurrent = wSizeLarge;

		defaultModelLoc = "../lib/local/CLM/model/main.txt";

		XLeftOffset = 100;
		XRightOffset = 100;
		YTopOffset = 100;
		YBottomOffset = 100;

		objectWidth = 200;

		useMorphology = true;

		sigma = 10;
		decisionBoundary = -0.5;
	}

};

}

#endif __CLM_PARAM_H