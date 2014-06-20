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


#include "PAW.h"

#include <cv.h>
#include <highgui.h>

#define it at<int>
#define db at<double>

using namespace CLMTracker;

//===========================================================================
void PAW::Read(ifstream &s)
{
	IO::SkipComments(s);
	s >> _nPix >> _xmin >> _ymin;

	IO::SkipComments(s);
	IO::ReadMat(s,_src);

	IO::SkipComments(s);
	IO::ReadMat(s,_tri);

	IO::SkipComments(s);
	IO::ReadMat(s,_tridx);
	
	cv::Mat tmpMask;
	IO::SkipComments(s);		
	IO::ReadMat(s, tmpMask);	
	tmpMask.convertTo(_mask, CV_8U);	

	//cout << _mask << endl;
	IO::SkipComments(s);
	IO::ReadMat(s,_alpha);

	IO::SkipComments(s);
	IO::ReadMat(s,_beta);

	_mapx.create(_mask.rows,_mask.cols,CV_32F);
	_mapy.create(_mask.rows,_mask.cols,CV_32F);
	_coeff.create(this->nTri(),6,CV_64F); _dst = _src;
}


void PAW::Crop(const cv::Mat &src, cv::Mat &dst, cv::Mat &s)
{
  assert((s.type() == CV_64F) && (s.rows == _src.rows) && (s.cols == 1) && 
	 (src.type() == dst.type()));
  
  _dst = s;

  this->CalcCoeff();
  this->WarpRegion(_mapx,_mapy);
  
  cv::remap(src,dst,_mapx,_mapy,CV_INTER_LINEAR);
  
}

//=============================================================================
void PAW::CalcCoeff()
{
	int i,j,k,l,p=this->nPoints();
	double c1,c2,c3,c4,c5,c6,*coeff,*alpha,*beta;

	for(l = 0; l < this->nTri(); l++)
	{
	  
		i = _tri.it(l,0);
		j = _tri.it(l,1);
		k = _tri.it(l,2);

		c1 = _dst.db(i  ,0);
		c2 = _dst.db(j  ,0) - c1;
		c3 = _dst.db(k  ,0) - c1;
		c4 = _dst.db(i+p,0);
		c5 = _dst.db(j+p,0) - c4;
		c6 = _dst.db(k+p,0) - c4;

		coeff = _coeff.ptr<double>(l);
		alpha = _alpha.ptr<double>(l);
		beta  = _beta.ptr<double>(l);

		coeff[0] = c1 + c2*alpha[0] + c3*beta[0];
		coeff[1] =      c2*alpha[1] + c3*beta[1];
		coeff[2] =      c2*alpha[2] + c3*beta[2];
		coeff[3] = c4 + c5*alpha[0] + c6*beta[0];
		coeff[4] =      c5*alpha[1] + c6*beta[1];
		coeff[5] =      c5*alpha[2] + c6*beta[2];
	}
}
//=============================================================================
void PAW::WarpRegion(cv::Mat &mapx,cv::Mat &mapy)
{
	assert((mapx.type() == CV_32F) && (mapy.type() == CV_32F));

	if((mapx.rows != _mask.rows) || (mapx.cols != _mask.cols))
		_mapx.create(_mask.rows,_mask.cols,CV_32F);

	if((mapy.rows != _mask.rows) || (mapy.cols != _mask.cols))
		_mapy.create(_mask.rows,_mask.cols,CV_32F);

	int x,y,j,k=-1;
	double yi,xi,xo,yo,*a=NULL,*ap;

	cv::MatIterator_<float> xp = mapx.begin<float>();
	cv::MatIterator_<float> yp = mapy.begin<float>();
	cv::MatIterator_<uchar> mp = _mask.begin<uchar>();
	cv::MatIterator_<int>   tp = _tridx.begin<int>();
	//cout << _mask << endl << _mask.type();
	//CV_8U
	for(y = 0; y < _mask.rows; y++)
	{
		yi = double(y) + _ymin;

		for(x = 0; x < _mask.cols; x++)
		{
			xi = double(x) + _xmin;

			if(*mp == 0)
			{
				*xp = -1;
				*yp = -1;
			}
			else
			{
				j = *tp;
				if(j != k)
				{
					a = _coeff.ptr<double>(j);
					k = j;
				}  	
				ap = a;
				xo = *ap++;
				xo += *ap++ * xi;
				*xp = float(xo + *ap++ * yi);
				yo = *ap++;
				yo += *ap++ * xi;
				*yp = float(yo + *ap++ * yi);
			}
			mp++; tp++; xp++; yp++;
		}
	}
}
//===========================================================================
