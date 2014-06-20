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

///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2010, Jason Mora Saragih, all rights reserved.
//
// This file is part of FaceTracker.
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
//       J. M. Saragih, S. Lucey, and J. F. Cohn. Face Alignment through 
//       Subspace Constrained Mean-Shifts. International Conference of Computer 
//       Vision (ICCV), September, 2009.
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

#include "FCheck.h"
#include <highgui.h>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace CLMTracker;

//===========================================================================
void FCheck::Read(ifstream &s)
{
	IO::SkipComments(s);
	s >> _b;
	IO::ReadMat(s,_w);
	
	IO::SkipComments(s);
	s >> _bDepth;
	IO::ReadMat(s,_wDepth);
	
	IO::SkipComments(s);
	s >> _bComb;
	cv::Mat_<double> wComb;
	IO::ReadMat(s,wComb);
	_wCombInt = wComb(cv::Rect(0,0,1,wComb.rows/2));
	_wCombDepth = wComb(cv::Rect(0,wComb.rows/2,1,wComb.rows/2));

	_paw.Read(s);

	crop_.create(_paw._mask.rows,_paw._mask.cols,CV_8U);
	vec_.create(_paw._nPix,1,CV_64F);

	cropDepth_.create(_paw._mask.rows,_paw._mask.cols);
	vecDepth_.create(_paw._nPix,1);
}
//===========================================================================
// Check if the fitting actually succeeded
bool FCheck::Check(const cv::Mat_<uchar>& intensityImg, const cv::Mat_<float>& depthImg, cv::Mat &s, double decisionBoundary)
{
	assert((s.type() == CV_64F) && (s.rows/2 == _paw.nPoints()) && (s.cols == 1));

	bool useIntensity = false, useDepth = false;

	
	cv::Mat cropT;	
	// the vector to be filled with paw values
	cv::MatIterator_<double> vp;	
	cv::MatIterator_<uchar>  cp;

	if(!intensityImg.empty())
	{
		_paw.Crop(intensityImg, crop_, s);	
		useIntensity = true;
		
		// the piece-wise affine image
		cropT = crop_.t();
		vp = vec_.begin<double>();
		cp = cropT.begin<uchar>();
	}
	
	cv::Mat_<double> cropTD;
	cv::MatIterator_<double> vpD;	
	cv::MatIterator_<double>  cpD;

	if(!depthImg.empty())
	{
		// First deal with missing data here

		double minX, maxX, minY, maxY;
		cv::minMaxLoc(s(cv::Rect(0, 0, 1, s.rows/2)), &minX, &maxX);
		cv::minMaxLoc(s(cv::Rect(0, s.rows/2, 1, s.rows/2)), &minY, &maxY);

		// get slightly more than the bounding box by shape, to make sure it all fits
		int minXInt = (int)(minX - 2);
		int minYInt = (int)(minY - 2);
		minXInt = minXInt < 0 ? 0 : minXInt;
		minYInt = minYInt < 0 ? 0 : minYInt;
		
		int width = (int)(maxX - minXInt + 4);
		int height = (int)(maxY - minYInt + 4);
		width = minXInt + width > depthImg.cols ? depthImg.cols - minXInt : width;
		height = minYInt + height > depthImg.rows ? depthImg.rows - minYInt : height;

		cv::Rect depthRoi(minXInt, minYInt, width, height);

		cv::Mat_<float> depthCopy = depthImg.clone();
		cv::Mat_<float> depthCopySm = depthCopy(depthRoi);
		
		cv::Mat_<uchar> maskDepth = depthCopySm > 0;

		cv::Scalar mean, stDev;
		cv::meanStdDev(depthCopySm, mean, stDev, maskDepth); 

		if(stDev[0] != 0)
		{
			depthCopySm = (depthCopySm - mean[0]) / stDev[0];
		}
		else
		{
			depthCopySm = (depthCopySm - mean[0]);
		}

		depthCopySm.setTo(0, maskDepth == 0);

		_paw.Crop(depthCopy, cropDepth_, s);
		
		useDepth = true;
		cropTD = cropDepth_.t();
		vpD = vecDepth_.begin();	
		cpD = cropTD.begin();
	}

	if((vec_.rows!=_paw._nPix)||(vec_.cols!=1)||vecDepth_.rows !=_paw._nPix || vecDepth_.cols !=1)
	{
		vec_.create(_paw._nPix,1,CV_64F);
		vecDepth_.create(_paw._nPix,1);
	}

	int wInt = crop_.cols;
	int hInt = crop_.rows;

	// the mask
	cv::Mat maskT = _paw._mask.t();

	//cv::MatIterator_<uchar>  mp = _paw._mask.begin<uchar>();
	cv::MatIterator_<uchar>  mp = maskT.begin<uchar>();

	for(int i=0;i<wInt;i++)
	{
		for(int j=0;j<hInt;j++,++mp,++cp,++cpD)
		{
			// if is within mask
			if(*mp)
			{
				if(useIntensity)
				{
					*vp++ = (double)*cp;
				}
				if(useDepth)
				{
					*vpD++ = (double)*cpD;
				}
			}
		}
	}

	if(useIntensity)
	{
		cv::Scalar mean;
		cv::Scalar std;
		cv::meanStdDev(vec_, mean, std);
		vec_-=mean[0];

		if(std[0] < 1.0e-10)
		{
			vec_.setTo(0);
		}
		else
		{
			vec_ /= std[0];
		}
	}

	double dec;

	if(useIntensity)
	{
		if(useDepth)
		{
			// combined depth and intensity
			//cv::Mat_<double> intensityPartVec = _wCom
			//cout << vecDepth_ << endl;
			dec = (_wCombDepth.dot(vecDepth_));
			dec += (_wCombInt.dot(vec_));
			dec += _bComb;
		}
		else
		{
			
			dec = (_w.dot(vec_) + _b);
			
		}
	}
	else
	{
		dec = (_wDepth.dot(vecDepth_) + _bDepth);
	}

	// be quite conservative with the decision, as don't want to stop if the tracking isn't going that badly, and in some cases the classifier is slightly too conservative
	if( dec > decisionBoundary)
		return true; 
	else 
		return false;
}

//===========================================================================
void MFCheck::Read(string location)
{

	ifstream faceCheckLoc(location);
	if(!faceCheckLoc.is_open())
	{
		cout << "WARNING: Can't find the Face checker location" << endl;
	}

	IO::SkipComments(faceCheckLoc);
	int n;
	faceCheckLoc >> n;
	
	IO::SkipComments(faceCheckLoc);

	_orientations.resize(n);
	for(int i = 0; i < n; i++)
	{
		cv::Mat orientationTmp;
		IO::ReadMat(faceCheckLoc, orientationTmp);		
		
		_orientations[i](0) = orientationTmp.at<double>(0);
		_orientations[i](1) = orientationTmp.at<double>(1);
		_orientations[i](2) = orientationTmp.at<double>(2);

		_orientations[i] = _orientations[i] * M_PI / 180.0;
	}

	_fcheck.resize(n);
	for(int i = 0; i < n; i++)
	{
		_fcheck[i].Read(faceCheckLoc);
	}
}

// Getting the closest view center based on orientation
int MFCheck::GetViewId(const cv::Vec3d& orientation)
{
	int idx = 0;
	if(this->_orientations.size() == 1)
	{
		return 0;
	}
	else
	{
		double d,dbest = -1.0;

		for(size_t i = 0; i < this->_orientations.size(); i++)
		{
	
			d = cv::norm(orientation, this->_orientations[i]);

			if(dbest < 0 || d < dbest)
			{
				dbest = d;
				idx = i;
			}
		}
		return idx;
	}
}

//===========================================================================
bool MFCheck::Check(const cv::Vec3d& orientation, const cv::Mat_<uchar>& im, const cv::Mat_<float>& depthIm, cv::Mat &s, double decisionBoundary)
{
	int id = GetViewId(orientation);

	return _fcheck[id].Check(im, depthIm, s, decisionBoundary);
}

