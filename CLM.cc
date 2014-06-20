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

#include "CLM.h"

#include <stdio.h>
#include <iostream>

#include <highgui.h>

#define it at<int>
#define db at<double>
#define SQR(x) x*x
#define PI 3.14159265
using namespace CLMTracker;
//=============================================================================
// calculating the similarity transform from the reference to the destination shape
void CalcSimT(cv::Mat &src,cv::Mat &dst, double &a,double &b,double &tx,double &ty)
{
	assert((src.type() == CV_64F) && (dst.type() == CV_64F) && (src.rows == dst.rows) && (src.cols == dst.cols) && (src.cols == 1));

	int i,n = src.rows/2;
  
	cv::Mat H(4,4,CV_64F,cv::Scalar(0));
  
	cv::Mat g(4,1,CV_64F,cv::Scalar(0));
  
	cv::Mat p(4,1,CV_64F);
  
	// source shape
	cv::MatIterator_<double> ptr1x = src.begin<double>();
	cv::MatIterator_<double> ptr1y = src.begin<double>()+n;

	// destination shape
	cv::MatIterator_<double> ptr2x = dst.begin<double>();
	cv::MatIterator_<double> ptr2y = dst.begin<double>()+n;
  
	for(i = 0; i < n; i++,++ptr1x,++ptr1y,++ptr2x,++ptr2y)
	{

		H.db(0,0) += SQR(*ptr1x) + SQR(*ptr1y);
		H.db(0,2) += *ptr1x;
		H.db(0,3) += *ptr1y;
    
		g.db(0,0) += (*ptr1x)*(*ptr2x) + (*ptr1y)*(*ptr2y);
		g.db(1,0) += (*ptr1x)*(*ptr2y) - (*ptr1y)*(*ptr2x);
		g.db(2,0) += *ptr2x;
		g.db(3,0) += *ptr2y;
	}

	H.db(1,1) = H.db(0,0);
	H.db(1,2) = H.db(2,1) = -1.0*(H.db(3,0) = H.db(0,3));
	
	H.db(1,3) = H.db(3,1) = H.db(2,0) = H.db(0,2);
	H.db(2,2) = H.db(3,3) = n;
	//std::cout << "H " << H << std::endl;
	//std::cout << "g " << g << std::endl;
	cv::solve(H,g,p,CV_CHOLESKY);
	a = p.db(0,0);
	b = p.db(1,0);
	tx = p.db(2,0);
	ty = p.db(3,0);
	return;
}
//=============================================================================
// Inverse the similarity transform from the 4 parameter style similarity transform
void invSimT(double a1,double b1,double tx1,double ty1, double& a2,double& b2,double& tx2,double& ty2)
{
  cv::Mat M = (cv::Mat_<double>(2,2) << a1, -b1, b1, a1);
  cv::Mat N = M.inv(CV_SVD);
  a2 = N.db(0,0);
  b2 = N.db(1,0);

  // rotate the translation correction
  tx2 = -1.0*(N.db(0,0)*tx1 + N.db(0,1)*ty1);
  ty2 = -1.0*(N.db(1,0)*tx1 + N.db(1,1)*ty1);
  return;
}
//=============================================================================
void SimT(cv::Mat &s,double a,double b,double tx,double ty)
{
  assert((s.type() == CV_64F) && (s.cols == 1));
  int i,n = s.rows/2; double x,y; 
  cv::MatIterator_<double> xp = s.begin<double>(),yp = s.begin<double>()+n;
  for(i = 0; i < n; i++,++xp,++yp){
    x = *xp; y = *yp; *xp = a*x - b*y + tx; *yp = b*x + a*y + ty;    
  }return;
}
//=============================================================================
//=============================================================================
//=============================================================================
//=============================================================================
//=============================================================================
//=============================================================================

void CLM::Read(string clmLocation)
{

	// n - number of views

	// PDM location 

	// Location of modules
	ifstream locations(clmLocation);

	string line;
	
	vector<string> colourPatchesLocations;
	vector<string> depthPatchesLocations;

	// The main file contains the references to other files
	while (!locations.eof())
	{ 
		
		getline(locations, line);

		stringstream lineStream(line);

		string module;
		string location;

		// figure out which module is to be read from which file
		lineStream >> module;

		getline(lineStream, location);
		location.erase(location.begin()); // remove the first space


		if (module.compare("PDM") == 0) 
		{                    			
			_pdm.Read(location);
		}
		else if (module.compare("Triangulations") == 0) 
		{                    			
			ifstream triangulationFile(location);

			IO::SkipComments(triangulationFile);

			int numViews;
			triangulationFile >> numViews;

			// read in the triangulations
			_triangulations.resize(numViews);

			for(int i = 0; i < numViews; ++i)
			{
				IO::SkipComments(triangulationFile);
				IO::ReadMat(triangulationFile, _triangulations[i]);
			}
		}
		else if(module.compare("PatchesIntensity") == 0)
		{
			colourPatchesLocations.push_back(location);
		}
		else if(module.compare("PatchesDepth") == 0)
		{
			depthPatchesLocations.push_back(location);
		}

	}
  
	_cent.resize(colourPatchesLocations.size());
	_visi.resize(colourPatchesLocations.size());
	_patch.resize(colourPatchesLocations.size());
	_patchScaling.resize(colourPatchesLocations.size());

	for(size_t i = 0; i < colourPatchesLocations.size(); ++i)
	{
		string location = colourPatchesLocations[i];
		ReadPatches(location, _cent[i], _visi[i], _patch[i], _patchScaling[i]);
	}

	_centDepth.resize(depthPatchesLocations.size());
	_visiDepth.resize(depthPatchesLocations.size());
	_patchDepth.resize(depthPatchesLocations.size());
	_patchScalingDepth.resize(depthPatchesLocations.size());

	for(size_t i = 0; i < depthPatchesLocations.size(); ++i)
	{
		string location = depthPatchesLocations[i];
		ReadPatches(location, _centDepth[i], _visiDepth[i], _patchDepth[i], _patchScalingDepth[i]);
	}
	//IO::ReadMat(s,_refs);

	// Initialising some values

	// local parameters (shape)
	_plocal.create(_pdm.NumberOfModesExpr(), 1, CV_64F);
	_paramsMorph.create(_pdm.NumberOfModesMorph(), 1);

	// global parameters (pose)
	_pglobl.create(6,1,CV_64F);

	cshape_.create(2*_pdm.NumberOfPoints() ,1, CV_64F);
	bshape_.create(2*_pdm.NumberOfPoints() ,1, CV_64F);
	oshape_.create(2*_pdm.NumberOfPoints() ,1, CV_64F);
	
	// Patch response, jacobians and hessians
	ms_.create(2*_pdm.NumberOfPoints(),1,CV_64F);
	u_.create(6+_pdm.NumberOfModesExpr(),1,CV_64F);
	g_.create(6+_pdm.NumberOfModesExpr(),1,CV_64F);
	
	J_.create(2*_pdm.NumberOfPoints(),6+_pdm.NumberOfModesExpr(),CV_64F);
	H_.create(6+_pdm.NumberOfModesExpr(),6+_pdm.NumberOfModesExpr(),CV_64F);

	Jmorph_.create(2*_pdm.NumberOfPoints(), 6+_pdm.NumberOfModesMorph());
	Hmorph_.create(6+_pdm.NumberOfModesMorph(), 6+_pdm.NumberOfModesMorph());
	uMorph_.create(6+_pdm.NumberOfModesMorph(),1);
	gMorph_.create(6+_pdm.NumberOfModesMorph(),1);

	prob_.resize(_pdm.NumberOfPoints()); pmem_.resize(_pdm.NumberOfPoints()); 
	wmem_.resize(_pdm.NumberOfPoints());
}

void CLM::ReadPatches(string patchesFileLocation, std::vector<cv::Mat>& centers, std::vector<cv::Mat>& visibility, std::vector<std::vector<MPatch> >& patches, double& patchScaling)
{

	ifstream patchesFile(patchesFileLocation);

	if(patchesFile.is_open())
	{
		IO::SkipComments(patchesFile);

		patchesFile >> patchScaling;

		IO::SkipComments(patchesFile);

		int numberViews;		

		patchesFile >> numberViews; 

		// read pdm
		centers.resize(numberViews);
		visibility.resize(numberViews);
  
		// read the patches
		patches.resize(numberViews);

		IO::SkipComments(patchesFile);

		// centers of each view (which view corresponds to which orientation)
		for(size_t i = 0; i < centers.size(); i++)
		{
			IO::ReadMat(patchesFile, centers[i]);		
			centers[i] = centers[i] * PI / 180.0;
			//cout << centers[i];
		}

		IO::SkipComments(patchesFile);

		// the visibility of points for each of the views (which verts are visible at a specific view
		for(size_t i = 0; i < visibility.size(); i++)
		{
			IO::ReadMat(patchesFile, visibility[i]);				
		}
		//cout << visibility[0] << endl;
		int numberOfPoints = visibility[0].rows;

		IO::SkipComments(patchesFile);

		// read the patches themselves
		for(size_t i = 0; i < patches.size(); i++)
		{
			// number of patches for each view
			patches[i].resize(numberOfPoints);
			// read in each patch
			for(int j = 0; j < numberOfPoints; j++)
			{
				patches[i][j].Read(patchesFile);
			}
		}
	
		Mat_<double> refGlobal(6, 1, 0.0);
		refGlobal.at<double>(0) = 1;

		_pdm.CalcShape2D(_refs, cv::Mat_<double>(_pdm._E.cols, 1, 0.0), cv::Mat_<double>(_pdm._Emorph.cols, 1, 0.0), refGlobal);
	}
	else
	{
		cout << "Can't find/open the patches file\n" << endl;
	}
}

//=============================================================================
// Getting the closest view center based on orientation
int CLM::GetViewIdx(int scale)
{
	//cout << _pglobl.db(1,0) << " " << _pglobl.db(2,0) << " " << _pglobl.db(3,0) << endl;
	int idx = 0;
	if(this->nViews() == 1)
	{
		return 0;
	}
	else
	{
		int i;
		double v1,v2,v3,d,dbest = -1.0;

		for(i = 0; i < this->nViews(scale); i++)
		{
			v1 = _pglobl.db(1,0) - _cent[scale][i].db(0,0); 
			v2 = _pglobl.db(2,0) - _cent[scale][i].db(1,0);
			v3 = _pglobl.db(3,0) - _cent[scale][i].db(2,0);
			
			d = v1*v1 + v2*v2 + v3*v3;
			if(dbest < 0 || d < dbest)
			{
				dbest = d;
				idx = i;
			}
		}
		return idx;
		//return 0;
	}
}

int CLM::GetDepthViewIdx(int scale)
{
	int idx = 0;
	if(this->nViews() == 1)
	{
		return 0;
	}
	else
	{
		int i;
		double v1,v2,v3,d,dbest = -1.0;

		for(i = 0; i < this->nViews(); i++)
		{
			v1 = _pglobl.db(1,0) - _centDepth[scale][i].db(0,0); 
			v2 = _pglobl.db(2,0) - _centDepth[scale][i].db(1,0);
			v3 = _pglobl.db(3,0) - _centDepth[scale][i].db(2,0);
			d = v1*v1 + v2*v2 + v3*v3;
			if(dbest < 0 || d < dbest)
			{
				dbest = d;
				idx = i;
			}
		}
		return idx;
	}
}

//=============================================================================
bool CLM::Fit(const cv::Mat_<uchar>& im, const cv::Mat_<float>& depthImg, const std::vector<int>& wSize, int nIter, double clamp, double fTol, bool morphology, double sigma)
{
	assert(im.type() == CV_8U);
	

	int n = _pdm.NumberOfPoints(); 
	
	// similarity and inverse similarity
	
	cv::Mat_<uchar> mask;
	cv::Mat_<float> clampedDepth;	

	
	// Background elimination
	if(!depthImg.empty())
	{

		double tx = _pglobl.at<double>(4);
		double ty = _pglobl.at<double>(5);

		// if we can't sample around tx, fail
		if(tx - 9 <= 0 || ty - 9 <= 0 || tx + 9 >= im.cols || ty + 9 >= im.rows)
		{
			//cout << "failure here tx" << endl;
			return false;
		}

		_pdm.CalcShape2D(cshape_, _plocal, _paramsMorph, _pglobl);


		double minX, maxX, minY, maxY;

		cv::minMaxLoc(cshape_(Range(0, n),Range(0,1)), &minX, &maxX);
		cv::minMaxLoc(cshape_(Range(n, n*2),Range(0,1)), &minY, &maxY);

		double width = 3 * (maxX - minX); // the area of interest: cshape minX and maxX with some scaling
		double height = 2.5 * (maxY - minY); // These scalings are fairly ad-hoc

		// getting the region of interest from the
		cv::Rect_<int> roi((int)(tx-width/2), (int)(ty - height/2), (int)width, (int)height);

		if(roi.x < 0) roi.x = 0;
		if(roi.y < 0) roi.y = 0;
		if(roi.width + roi.x >= depthImg.cols) roi.x = depthImg.cols - roi.width;
		if(roi.height + roi.y >= depthImg.rows) roi.y = depthImg.rows - roi.height;
		
		if(width > depthImg.cols)
		{
			roi.x = 0; roi.width = depthImg.cols;
		}
		if(height > depthImg.rows)
		{
			roi.y = 0; roi.height = depthImg.rows;
		}

		if(roi.width == 0) roi.width = depthImg.cols;
		if(roi.height == 0) roi.height = depthImg.rows;

		if(roi.x >= depthImg.cols) roi.x = 0;
		if(roi.y >= depthImg.rows) roi.y = 0;

		// can clamp the depth values based on shape
		mask = cv::Mat_<uchar>(depthImg.rows, depthImg.cols, (uchar)0);

		cv::Mat_<uchar> currentFrameMask = depthImg > 0;
		if(sum(currentFrameMask(cv::Rect((int)tx - 8, (int)ty - 8, 16, 16))/255)[0] > 0)
		{
			double Z = cv::mean(depthImg(cv::Rect((int)tx - 8, (int)ty - 8, 16, 16)), currentFrameMask(cv::Rect((int)tx - 8, (int)ty - 8, 16, 16)))[0]; // Z offset from the surface of the face
				
			cv::Mat dRoi = depthImg(roi);

			cv::Mat mRoi = mask(roi);

			cv::inRange(dRoi, Z - 200, Z + 200, mRoi);
			
			mask = mask / 255;
		
			Mat_<float> maskF;
			mask.convertTo(maskF, CV_32F);

			clampedDepth = depthImg.mul(maskF);
		}
		else
		{
			// this will indicate failure
			return false;
		}
	}

	//int scale = 0;
	//int depthScale = 0;
	double currScale = _pglobl.at<double>(0);

	// Find the closes depth and colour patch scales, and start wIter below, this will make sure that the last iteration is done at the best scale available
	int scale;
	int depthScale;

	double minDist = 10;
	for( size_t i = 0; i < _patchScaling.size(); ++i)
	{
		if(std::abs(_patchScaling[i] - currScale) < minDist)
		{
			minDist = std::abs(_patchScaling[i] - currScale);
			scale = i + 1;
		}

	}
	
	minDist = 10;
	for( size_t i = 0; i < _patchScalingDepth.size(); ++i)
	{
		if(std::abs(_patchScalingDepth[i] - currScale) < minDist)
		{
			minDist = std::abs(_patchScaling[i] - currScale);
			depthScale = i + 1;
		}

	}

	scale = scale - wSize.size();
	depthScale = depthScale - wSize.size();

	if(scale < 0)
		scale = 0;

	if(depthScale < 0)
		depthScale = 0;

	int numColScales = _patchScaling.size();
	int numDepthScales = _patchScalingDepth.size();

	// Go over the max number of iterations for optimisation
	for(size_t witer = 0; witer < wSize.size(); witer++)
	{
		//cout << "curr scale: " << _patchScaling[scale] << endl;
		// convert the mm scaling to the scaling that the current patches are trained on
		_pglobl.at<double>(0) = _pglobl.at<double>(0) / _patchScaling[scale];
		_plocal = _plocal * _patchScaling[scale];
		_paramsMorph = _paramsMorph * _patchScaling[scale];
		
		Mat oldE = _pdm._E.clone();
		Mat oldEmorph = _pdm._Emorph.clone();
		Mat oldM = _pdm._M.clone();
	
		_pdm._E = _pdm._E * (_patchScaling[scale] * _patchScaling[scale]);
		_pdm._Emorph = _pdm._Emorph * (_patchScaling[scale] * _patchScaling[scale]);
		_pdm._M = _pdm._M * (_patchScaling[scale]);

		_pdm.CalcShape2D(cshape_, _plocal, _paramsMorph, _pglobl);		
	

		int idx = this->GetViewIdx(scale);		
		
		cv::Mat referenceShape;
		cv::Mat globalRef = _pglobl.clone();
		globalRef.at<double>(0) = 1;

		// this will aid with some out of plane motion (as we're mapping from current global to the center of training location)
		//cout << _cent[scale][idx] << endl;
		globalRef.at<double>(1) = _cent[scale][idx].at<double>(0);
		globalRef.at<double>(2) = _cent[scale][idx].at<double>(1);
		globalRef.at<double>(3) = _cent[scale][idx].at<double>(2);
		
		globalRef.at<double>(4) = 0;
		globalRef.at<double>(5) = 0;

		_pdm.CalcShape2D(referenceShape, _plocal, _paramsMorph, globalRef);

		double a1, b1, tx1, ty1, a2, b2, tx2, ty2;
		CalcSimT(referenceShape, cshape_, a1,b1,tx1,ty1);

		// inverse the similarity transform
		invSimT(a1,b1,tx1,ty1,a2,b2,tx2,ty2);

		//currScale = _pglobl.at<double>(0);

		//double patchScalingDiff = currScale / _patchScaling[scale];

		// for visualisation
		bool visi = false;
		
		Mat disp;
		if(visi)
		{
			cv::cvtColor(im, disp, CV_GRAY2BGR);
			_pdm.Draw(disp, cshape_);
			cv::imshow("disp", disp);
		}	

		//cout << cshape_ << endl;
		//Mat resizeImgUchar;

		//cv::resize(im, resizeImgUchar, Size((int)(im.cols/patchScalingDiff), (int)(im.rows / patchScalingDiff)));

		//Mat_<float> resizeImg;
		//resizeImgUchar.convertTo(resizeImg, CV_32F);

		//Mat_<double> patchSizeShape = cshape_ / patchScalingDiff;
		
		//Mat resizeImgDisp = resizeImgUchar.clone();
		//_pdm.Draw(resizeImgDisp, patchSizeShape);
		//imshow("resized", resizeImgDisp);
		//cv::waitKey(0);

#ifdef _OPENMP
#pragma omp parallel for
#endif
		// calculate the patch responses for every vertex
		for(int i = 0; i < n; i++)
		{
			
			if(_visi[scale][idx].rows == n)
			{
				if(_visi[scale][idx].it(i,0) == 0 || _patch[scale][idx][i]._p[0]._confidence < 0.1)
				{
					continue;
				}
			}

			int w = wSize[witer]+_patch[scale][idx][i]._w - 1; 
			int h = wSize[witer]+_patch[scale][idx][i]._h - 1;

			if((w>wmem_[i].cols) || (h>wmem_[i].rows))
			{
				wmem_[i].create(h,w,CV_32F);
			}

			//Mat_<float> currentPatch;

			//cv::getRectSubPix(resizeImg, Size(w, h), Point(patchSizeShape.at<double>(i), patchSizeShape.at<double>(i+n)), currentPatch);

			//cv::imshow("patch", patch);
			//cv::waitKey(0);

			// map matrix for quadrangle extraction, which is basically scaling
			//cv::Mat sim = (cv::Mat_<float>(2,3) << _pglobl.at<double>(0) / _patchScaling[scale], 0, cshape_.db(i,0), 0, _pglobl.at<double>(0) / _patchScaling[scale], cshape_.db(i+n,0));
			cv::Mat sim = (cv::Mat_<float>(2,3) << a1,-b1, cshape_.db(i,0), b1,a1,cshape_.db(i+n,0));
			
			cv::Mat_<float> wimg = wmem_[i](cv::Rect(0,0,w,h));
			CvMat wimg_o = wimg;

			CvMat sim_o = sim;

			IplImage im_o = im;
			
			cvGetQuadrangleSubPix(&im_o,&wimg_o,&sim_o);
			
	
			//if(wSize[witer] > pmem_[i].rows)
			//{
			//	pmem_[i].create(wSize[witer],wSize[witer],CV_64F);
			//}

			// get the correct size response window
			prob_[i] = Mat_<double>(wSize[witer], wSize[witer]);//pmem_[i](cv::Rect(0,0,wSize[witer],wSize[witer]));

			_patch[scale][idx][i].Response(wimg, prob_[i]);
			
			//_patch[scale][idx][i].Response(currentPatch, prob_[i]);
			
			//cout << "images:" << endl << wimg << endl;

			//cout << "patch resp:" << endl << prob_[i] << endl;

			bool visiResp = false;

			if(visiResp)
			{
				cv::Mat window;
				cv::pyrUp(wimg, window);
				cv::imshow("quadrangle", window / 255.0);

				double minD;
				double maxD;

				cv::minMaxLoc(prob_[i], &minD, &maxD);
				Mat visiProb;
				cv::pyrUp( prob_[i]/maxD, visiProb);
				cv::imshow("col resp", visiProb);		
				
			}

			// if we have a corresponding depth patch, and we are confident in it, also don't use depth for final iteration as it's not reliable enough it seems
			if(!depthImg.empty() && (_visiDepth[depthScale][idx].it(i,0) && _patchDepth[depthScale][idx][i]._p[0]._confidence > 0.1) && witer != 2)
			{				
				
				cv::Mat dProb = prob_[i].clone();
				cv::Mat depthWindow(h,w, clampedDepth.type());

				CvMat dimg_o = depthWindow;
				cv::Mat maskWindow(h,w, CV_32F);
				CvMat mimg_o = maskWindow;
				IplImage d_o = clampedDepth;
				IplImage m_o = mask;

				cvGetQuadrangleSubPix(&d_o,&dimg_o,&sim_o);

				
				cvGetQuadrangleSubPix(&m_o,&mimg_o,&sim_o);

				depthWindow.setTo(0, maskWindow < 1);

				_patchDepth[depthScale][idx][i].ResponseDepth(depthWindow, dProb);
				
			
				prob_[i] = prob_[i] + dProb;								
				
				// Sum to one
				double sum = 0; int cols = prob_[i].cols, rows = prob_[i].rows;
				if(prob_[i].isContinuous()){cols *= rows;rows = 1;}
				for(int x = 0; x < rows; x++){
					const double* Mi = prob_[i].ptr<double>(x);
					for(int y = 0; y < cols; y++)
					{
						sum += *Mi++;
					}
				}
				if(sum == 0)
				{
					sum = 1;
				}


				prob_[i] /= sum;

				if(visiResp)
				{
					double minD;
					double maxD;

					cv::minMaxLoc(dProb, &minD, &maxD);
					Mat visiProb;
					cv::pyrUp( dProb/maxD, visiProb);
					cv::imshow("Depth resp", visiProb);	

					cv::minMaxLoc(prob_[i], &minD, &maxD);
					cv::pyrUp( prob_[i]/maxD, visiProb);
					cv::imshow("Comb resp", visiProb);		

					//cv::waitKey(100);	
				}

			}
			
			if(visi)
			{
				double minD;
				double maxD;

				cv::minMaxLoc(prob_[i], &minD, &maxD);
				//cout << "max prob " << maxD << " center prob " << prob_[i].at<double>(wSize[witer]/2, wSize[witer]/2) << endl;
				Mat currProb;
				prob_[i].convertTo(currProb, CV_8U, 255 / maxD);
				cv::resize(currProb.clone(), currProb, Size(), (_pglobl.at<double>(0)), (_pglobl.at<double>(0)));
				// now need to resize the scale

				Mat currProb3U;
				cv::cvtColor(currProb, currProb3U, CV_GRAY2BGR, 3);
				currProb3U.copyTo(disp(Rect((int)(cshape_.db(i,0)-currProb.cols/2), (int)(cshape_.db(i+n,0)-currProb.rows/2), currProb.cols, currProb.rows)));
				cv::rectangle(disp, Rect((int)(cshape_.db(i,0)-currProb.cols/2), (int)(cshape_.db(i+n,0)-currProb.rows/2), currProb.cols, currProb.rows), Scalar(0, 255, 0), 1);
				imshow("Current patches", disp);
			}
		}

		if(visi)
		{
			imshow("Current patches", disp);
			//cv::waitKey(0);
		}

		// the actual optimisation step
		
		// apply the similarity transform on the 2D shape
		SimT(cshape_,a2,b2,tx2,ty2); 

		// apply the similarity transform to global parameters
		_pdm.ApplySimT(a2,b2,tx2,ty2,_pglobl);

		cshape_.copyTo(bshape_);

		// the actual optimisation step

		//std::cout << "global before opt" << _pglobl << endl;
		
		//cout << "before" << _paramsMorph << endl;

		// rigid pose optimisation
		this->Optimize(_pglobl, _plocal, _pglobl.clone(), _plocal.clone(), bshape_, idx,wSize[witer],nIter,fTol,clamp,1, scale, sigma);
		
		// non-rigid pose optimisation
		//this->Optimize(_pglobl, _plocal, _pglobl.clone(), _plocal.clone(), bshape_, idx,wSize[witer],nIter,fTol,clamp,0, scale, witer);
		
		// apply the similarity transform to get the final optimisation

		
		if(!morphology || _pdm._Emorph.cols == 0)
		{
			this->Optimize(_pglobl, _plocal, _pglobl.clone(), _plocal.clone(), bshape_, idx,wSize[witer],nIter,fTol,clamp,0, scale, sigma);
		}
		else
		{
			this->OptimizeMorphology(_pglobl, _paramsMorph, _pglobl.clone(), _paramsMorph.clone(), bshape_, idx,wSize[witer],nIter,fTol,clamp, scale, sigma);
			
		}
		
		//cout << _paramsMorph << endl;
		
		_pdm.ApplySimT(a1,b1,tx1,ty1,_pglobl);

		_pdm._E = oldE.clone();
		_pdm._M = oldM.clone();

		_pdm._Emorph = oldEmorph.clone();

		_pglobl.at<double>(0) = _pglobl.at<double>(0) * _patchScaling[scale];
		_plocal = _plocal / _patchScaling[scale];
		_paramsMorph = _paramsMorph / _patchScaling[scale];

		double currA = _pglobl.at<double>(0);
		
		// If there are more scales to go, and we don't need to upscale too much move to next scale level
		if(scale < numColScales - 1 && 0.9 * _patchScaling[scale] < currA)
		{
			scale++;			
		}

		if(depthScale < numDepthScales - 1 && 0.9 * _patchScalingDepth[depthScale] < currA)
		{
			depthScale++;
		}
		
		// do not let the pose get out of hand
		if(_pglobl.at<double>(1) > PI / 2)
			_pglobl.at<double>(1) = PI/2;
		if(_pglobl.at<double>(1) < -PI / 2)
			_pglobl.at<double>(1) = -PI/2;
		if(_pglobl.at<double>(2) > PI / 2)
			_pglobl.at<double>(2) = PI/2;
		if(_pglobl.at<double>(2) < -PI / 2)
			_pglobl.at<double>(2) = -PI/2;
		if(_pglobl.at<double>(3) > PI / 2)
			_pglobl.at<double>(3) = PI/2;
		if(_pglobl.at<double>(3) < -PI / 2)
			_pglobl.at<double>(3) = -PI/2;

		// Can't track very small images reliably (less than ~30px across)
		if(_pglobl.at<double>(0) < 0.2)
		{
			cout << "Detection too small for CLM" << endl;
			return false;
		}
		
	}


	return true;
}
//=============================================================================
void CLM::Optimize(Mat& finalGlobal, Mat& finalLocal, const Mat_<double>& initialGlobal, const Mat_<double>& initialLocal, const Mat_<double>& baseShape, int idx, int wSize, int nIter, double fTol,double clamp, bool rigid, int scale, double sigma)
{
	int i;
	int m=_pdm.NumberOfModesExpr();
	int n=_pdm.NumberOfPoints();  

	//double var,sigma=(wSize*wSize)/36.0;
	//double sigma = 10;//3.16 works with V8;//*_patchScaling[scale];//3.16/(sqrt((double)witer+1));; // deduced from remaining sigmas, we also want to smooth less when scale is larger
	double var;

	cv::Mat u,g,J,H; 
	
	Mat_<double> currentGlobal = initialGlobal.clone();
	Mat_<double> currentLocal = initialLocal.clone();

	Mat_<double> currentShape;
	Mat_<double> previousShape;

	//cout << "base shape" << endl << baseShape << endl;
	// Correct the PDM now

	// Jacobian and Hessian placeholders
	if(rigid)
	{
		u = u_(cv::Rect(0,0,1,6));
		g = g_(cv::Rect(0,0,1,6)); 
		J = J_(cv::Rect(0,0,6,2*n));
		H = H_(cv::Rect(0,0,6,6));
	}
	else
	{
		u = u_;
		g = g_;
		J = J_;
		H = H_;
	}

	// Number of iterations
	for(int iter = 0; iter < nIter; iter++)
	{
		//double currPatchScaling = currentGlobal.at<double>(0) / _patchScaling[scale];

		// get the current estimates of x
		_pdm.CalcShape2D(currentShape, currentLocal, _paramsMorph, currentGlobal);
		
		if(iter > 0)
		{
			// if the shape hasn't changed terminate
			if(cv::norm(currentShape,previousShape) < fTol)
				break;
		}

		currentShape.copyTo(previousShape);
		
		// purely for visualisation
		Mat_<float> meanShiftPatch(wSize,wSize);		

		// calculate the appropriate Jacobians in 2D, even though the actual behaviour is in 3D, using small angle approximation and oriented shape
		if(rigid)
		{
			_pdm.CalcRigidJacob(currentLocal, _paramsMorph, currentGlobal, J);
		}
		else
		{
			_pdm.CalcJacob(currentLocal, _paramsMorph, currentGlobal, J);
		}

#ifdef _OPENMP
#pragma omp parallel for
#endif
		// for every point (patch) basically calculating v
		for(i = 0; i < n; i++)
		{
			if(_visi[scale][idx].rows == n)
			{
				// if patch unavailable for current index, or too unreliable
				if(_visi[scale][idx].it(i,0) == 0  || _patch[scale][idx][i]._p[0]._confidence < 0.1 || cv::sum(prob_[i])[0] == 0)
				{
					cv::Mat Jx = J.row(i);
					Jx = cvScalar(0);
					cv::Mat Jy = J.row(i+n);
					Jy = cvScalar(0);
					ms_.db(i,0) = 0.0;
					ms_.db(i+n,0) = 0.0;
					continue;
				}
			}

			double dx = (currentShape.at<double>(i) - baseShape.at<double>(i)) + (wSize-1)/2;
			double dy = (currentShape.at<double>(i+n) - baseShape.at<double>(i+n)) + (wSize-1)/2;

			int ii,jj;
			double v,vx,vy,mx=0.0,my=0.0,sum=0.0;

			// Iterate over the patch responses here
			cv::MatIterator_<double> p = prob_[i].begin<double>();

			for(ii = 0; ii < wSize; ii++)
			{
				vx = (dy-ii)*(dy-ii);
				for(jj = 0; jj < wSize; jj++)
				{
					vy = (dx-jj)*(dx-jj);

					// the probability at the current, xi, yi
					v = *p++;

					// the KDE evaluation of that point
					v *= exp(-0.5*(vx+vy)/(sigma));

					sum += v;

					// mean shift in x and y
					mx += v*jj;
					my += v*ii; 

					// for visualisation
					//meanShiftPatch.at<float>(ii,jj) = v;
				}
			}
			// setting the actual mean shift update, sigma is hardcoded it seems (actually could experiment with it slightly, as scale has changed)

			double msx = (mx/sum - dx);
			double msy = (my/sum - dy);

			ms_.db(i,0) = msx;
			ms_.db(i+n,0) = msy;
			
			//cout << msx << " " << msy << endl;			
			
			//Mat meanShiftPatchVis;
			//double currPatchScaling = 4;
			//cv::resize(meanShiftPatch, meanShiftPatchVis, Size((int)wSize*currPatchScaling, (int)wSize*currPatchScaling));
			//cv::cvtColor(meanShiftPatchVis,meanShiftPatchVis, CV_GRAY2BGR);

			//double min, max;
			//cv::minMaxIdx(meanShiftPatchVis, &min, &max);
			//cv::circle(meanShiftPatchVis, cv::Point((dx + msx)*currPatchScaling, (dy + msy)*currPatchScaling), 2, cv::Scalar(0,0,255), 2);
			//cv::circle(meanShiftPatchVis, cv::Point((dx)*currPatchScaling, dy*currPatchScaling), 2, cv::Scalar(255,0,0), 2);
			//imshow("mean shift vec", meanShiftPatchVis/max);
			//
			//Mat actualPatchVis;
			//cv::resize(prob_[i], actualPatchVis, Size((int)wSize*currPatchScaling, (int)wSize*currPatchScaling));
			//actualPatchVis.clone().convertTo(actualPatchVis, CV_32F);
			//cv::cvtColor(actualPatchVis, actualPatchVis, CV_GRAY2BGR);

			//double minA, maxA;
			//cv::minMaxIdx(actualPatchVis, &minA, &maxA);
			//cv::circle(actualPatchVis, cv::Point((dx + msx )*currPatchScaling, (dy + msy )*currPatchScaling), 2, cv::Scalar(0,0,255), 2);
			//cv::circle(actualPatchVis, cv::Point((dx)*currPatchScaling, dy*currPatchScaling), 2, cv::Scalar(255,0,0), 2);
			//imshow("actual prob vec", actualPatchVis/maxA);
						
			//cv::waitKey(0);
		}
		// projection of the meanshifts onto the jacobians
		g = J.t()*ms_;
		
		// calculating the Hessian approximation
		H = J.t()*J;
		
		// if the optimisation is for non-rigid motion can use regularisation
		if(!rigid)
		{
			for(i = 0; i < m; i++)
			{

				var = 0.5*sigma/(_pdm._E.db(0,i));

				H.db(6+i,6+i) += var;
				g.db(6+i,0) -= var*currentLocal.db(i,0);
			}
		}

		u_ = cvScalar(0);

		// as u is a subset of u_, this solving only applies to the update actually being performed (rigid vs. non rigid)
		// the actual solution for delta p, eq (36) in Int
		cv::solve(H,g,u,CV_CHOLESKY);

		// update the reference
		_pdm.CalcReferenceUpdate(u_,currentLocal,currentGlobal, this->_pdm._V);

		if(!rigid)
		{
			// clamp to the local parameters for valid expressions
			_pdm.Clamp(currentLocal,clamp, this->_pdm._E);
		}


	}
	finalGlobal = currentGlobal;
	finalLocal = currentLocal;
	
}

void CLM::OptimizeMorphology(Mat& finalGlobal, Mat& finalLocal, const Mat_<double>& initialGlobal, const Mat_<double>& initialLocal, const Mat_<double>& baseShape, int idx, int wSize, int nIter, double fTol,double clamp, int scale, double sigma)
{
	int i;
	int m=_pdm.NumberOfModesMorph();
	int n=_pdm.NumberOfPoints();  

	//double sigma = 10;//3.16 works with V8;//*_patchScaling[scale];//3.16/(sqrt((double)witer+1));; // deduced from remaining sigmas, we also want to smooth less when scale is larger
	double var;

	Mat_<double> u = uMorph_;
	Mat_<double> g = gMorph_;
	Mat_<double> J = Jmorph_;
	Mat_<double> H = Hmorph_;
	
	Mat_<double> currentGlobal = initialGlobal.clone();
	Mat_<double> currentMorph = initialLocal.clone();
	
	Mat_<double> currentShape;
	Mat_<double> previousShape;

	for(int iter = 0; iter < nIter; iter++)
	{
		_pdm.CalcShape2D(currentShape, _plocal, currentMorph, currentGlobal);
		
		if(iter > 0)
		{
			// if the shape hasn't changed terminate
			if(cv::norm(currentShape,previousShape) < fTol)
				break;
		}
		currentShape.copyTo(previousShape);
		
		_pdm.CalcMorphologyJacob(currentMorph, currentGlobal, J);

#ifdef _OPENMP
#pragma omp parallel for
#endif
		// for every point (patch) basically calculating v
		for(i = 0; i < n; i++)
		{
			if(_visi[scale][idx].rows == n)
			{
				// if patch unavailable for current index
				if(_visi[scale][idx].it(i,0) == 0 || _patch[scale][idx][i]._p[0]._confidence < 0.1 || cv::sum(prob_[i])[0] == 0)
				{
					cv::Mat Jx = J.row(i  );
					Jx = cvScalar(0);
					cv::Mat Jy = J.row(i+n);
					Jy = cvScalar(0);
					ms_.db(i,0) = 0.0;
					ms_.db(i+n,0) = 0.0;
					continue;
				}
			}
			
			double dx = (currentShape.at<double>(i) - baseShape.at<double>(i)) + (wSize-1)/2;
			double dy = (currentShape.at<double>(i+n) - baseShape.at<double>(i+n)) + (wSize-1)/2;

			int ii,jj; double v,vx,vy,mx=0.0,my=0.0,sum=0.0;     

			// Iterate over the patch responses here
			cv::MatIterator_<double> p = prob_[i].begin<double>();

			for(ii = 0; ii < wSize; ii++)
			{
				vx = (dy-ii)*(dy-ii);
				for(jj = 0; jj < wSize; jj++)
				{
					vy = (dx-jj)*(dx-jj);

					// the probability at the current, xi, yi
					v = *p++;

					// the KDE evaluation of that point
					v *= exp(-0.5*(vx+vy)/(sigma));

					sum += v;

					// mean shift in x and y
					mx += v*jj;
					my += v*ii; 

				}
			}
			double msx = (mx/sum - dx);
			double msy = (my/sum - dy);

			ms_.db(i,0) = msx;
			ms_.db(i+n,0) = msy;
		}

		g = J.t()*ms_;
		H = J.t()*J;

		for(i = 0; i < m; i++)
		{			
			var = 0.5*sigma/(_pdm._E.db(0,i));

			H.db(6+i,6+i) += var;
			g.db(6+i,0) -= var*currentMorph.db(i,0);
		}
		
		uMorph_.setTo(0);

		// as u is a subset of uMorph_, this solving only applies to the update actually being performed (rigid vs. non rigid)
		cv::solve(H,g,u,CV_CHOLESKY);		

		// update the reference
		_pdm.CalcReferenceUpdate(uMorph_, currentMorph, _pglobl, this->_pdm._Vmorph);
		// clamp to the local parameters for valid expressions
		_pdm.Clamp(currentMorph, clamp, this->_pdm._Emorph);

	}
		
	finalGlobal = currentGlobal;
	finalLocal = currentMorph;
}
//=============================================================================
