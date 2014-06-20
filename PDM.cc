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

#include "PDM.h"
#include <iostream>

#define db at<double>
using namespace CLMTracker;
//===========================================================================
void AddOrthRow(cv::Mat &R)
{
  assert((R.rows == 3) && (R.cols == 3));
  // Using the formulation of the rotation matrix Rx * Ry * Rz, we can work out the 3rd row when we only know the first two ones (it's fairly straightforward math, just slightly tedious)
  R.db(2,0) = R.db(0,1)*R.db(1,2) - R.db(0,2)*R.db(1,1);
  R.db(2,1) = R.db(0,2)*R.db(1,0) - R.db(0,0)*R.db(1,2);
  R.db(2,2) = R.db(0,0)*R.db(1,1) - R.db(0,1)*R.db(1,0);
  return;
}
//=============================================================================
void MetricUpgrade(cv::Mat &R)
{
  assert((R.rows == 3) && (R.cols == 3));
  cv::SVD svd(R,cv::SVD::MODIFY_A);
  cv::Mat X = svd.u*svd.vt,W = cv::Mat::eye(3,3,CV_64F); 
  W.db(2,2) = determinant(X); R = svd.u*W*svd.vt; return;
}

// Converting from Euler angles Rx * Ry * Rz rotation to Rotation matrix
// Full indicates if a 3x3 or a 2x3 matrix is needed (in 2x3 case we don't care about resulting z component)
//===========================================================================
void Euler2Rot(cv::Mat &R,const double pitch,const double yaw,const double roll,
	       bool full = true)
{
	// defining the matrices based on needed size
	if(full)
	{
		if((R.rows != 3) || (R.cols != 3))
		{
			R.create(3,3,CV_64F);
		}
	}
	else
	{
		if((R.rows != 2) || (R.cols != 3))
		{
			R.create(2,3,CV_64F);
		}
	}
	// used for the rotation matrix calculations
	double sina = sin(pitch), sinb = sin(yaw), sinc = sin(roll);
	double cosa = cos(pitch), cosb = cos(yaw), cosc = cos(roll);
	
	R.db(0,0) = cosb*cosc;
	R.db(0,1) = -cosb*sinc;
	R.db(0,2) = sinb;
	R.db(1,0) = cosa*sinc + sina*sinb*cosc;
	R.db(1,1) = cosa*cosc - sina*sinb*sinc;
	R.db(1,2) = -sina*cosb;

	// if we need a full matrix construct it using the existing bit (as there is enough info in first two rows)
	if(full)
	{
		AddOrthRow(R);
	}
}
//===========================================================================
void Euler2Rot(cv::Mat &R, const cv::Mat &p,bool full = true)
{
  assert((p.rows == 6) && (p.cols == 1));
  Euler2Rot(R,p.db(1,0),p.db(2,0),p.db(3,0),full); return;
}
//=============================================================================
void Rot2Euler(cv::Mat &R,double& pitch,double& yaw,double& roll)
{
  assert((R.rows == 3) && (R.cols == 3));
  double q[4];
  q[0] = sqrt(1+R.db(0,0)+R.db(1,1)+R.db(2,2))/2;
  q[1] = (R.db(2,1) - R.db(1,2)) / (4*q[0]) ;
  q[2] = (R.db(0,2) - R.db(2,0)) / (4*q[0]) ;
  q[3] = (R.db(1,0) - R.db(0,1)) / (4*q[0]) ;
  yaw  = asin(2*(q[0]*q[2] + q[1]*q[3]));
  pitch= atan2(2*(q[0]*q[1]-q[2]*q[3]),
	       q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3]); 
  roll = atan2(2*(q[0]*q[3]-q[1]*q[2]),
	       q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3]);
  return;
}
//=============================================================================
void Rot2Euler(cv::Mat &R,cv::Mat &p)
{
  assert((p.rows == 6) && (p.cols == 1));
  Rot2Euler(R,p.db(1,0),p.db(2,0),p.db(3,0)); return;
}

//=============================================================================
//=============================================================================
//=============================================================================
//=============================================================================
//=============================================================================
//=============================================================================
//=============================================================================
//=============================================================================
//=============================================================================
//=============================================================================
//=============================================================================

//===========================================================================
void PDM::Clamp(cv::Mat &p, double c, const Mat& E)
{
	assert((p.rows == E.cols) && (p.cols == 1) && (p.type() == CV_64F));
	cv::MatConstIterator_<double> e  = E.begin<double>();
	cv::MatIterator_<double> p1 =  p.begin<double>();
	cv::MatIterator_<double> p2 =  p.end<double>();
	double v;

	// go over all parameters
	for(; p1 != p2; ++p1,++e)
	{
		v = c*sqrt(*e);
		// if the values is too extreme clamp it
		if(fabs(*p1) > v)
		{
			if(*p1 > 0.0)
			{
				*p1=v;
			}
			else
			{
				*p1=-v;
			}
		}
	}

}
//===========================================================================
void PDM::CalcShape3D(cv::Mat &s, const Mat& plocal, const Mat_<double>& pLocalMoph)
{
	assert((s.type() == CV_64F) && (plocal.type() == CV_64F));
	assert((s.rows == _M.rows) && (s.cols = 1));
	assert((plocal.rows == _E.cols) && (plocal.cols == 1));

	s = _M + _V*plocal;

	if(!pLocalMoph.empty())
	{
		s = s + _Vmorph * pLocalMoph;
	}
}
//===========================================================================
// Get the 2D shape from global and local parameters
void PDM::CalcShape2D(cv::Mat &s, const Mat& plocal, const Mat_<double>& pLocalMoph, const Mat &pglobl)
{
	assert((plocal.type() == CV_64F) && (pglobl.type() == CV_64F));
	assert((plocal.rows == _E.cols) && (plocal.cols == 1));
	assert((pglobl.rows == 6) && (pglobl.cols == 1));

	int n = _M.rows/3;
	double a=pglobl.db(0,0); // scaling factor
	double x=pglobl.db(4,0); // x offset
	double y=pglobl.db(5,0); // y offset

	// get the rotation matrix from the euler angles
	Euler2Rot(R_,pglobl);

	// get the 3D shape of the object
	S_ = _M + _V*plocal;

	if(!pLocalMoph.empty())
	{
		S_ = S_ + _Vmorph * pLocalMoph;
	}
	
	// create the 2D shape matrix
	if((s.rows != _M.rows) || (s.cols = 1))
		s.create(2*n,1,CV_64F);

	// for every vertex
	for(int i = 0; i < n; i++)
	{
		// Transform this using the weak-perspective mapping to 2D from 3D
		s.db(i  ,0) = a*( R_.db(0,0)*S_.db(i    ,0) + R_.db(0,1)*S_.db(i+n  ,0) + R_.db(0,2)*S_.db(i+n*2,0) )+x;
		s.db(i+n,0) = a*( R_.db(1,0)*S_.db(i    ,0) + R_.db(1,1)*S_.db(i+n  ,0) + R_.db(1,2)*S_.db(i+n*2,0) )+y;
	}
	return;
}

//===========================================================================
void PDM::CalcParams(Mat& pGlobal, const Rect& roi, const Mat& pLocal, const Mat_<double>& pLocalMorph, const Vec3d& rotation)
{

	// get the shape instance based on local params
	Mat_<double> currentShape(_M.size());

	CalcShape3D(currentShape, pLocal, pLocalMorph);

	// rotate the shape
	Mat rotationMatrix;
	Euler2Rot(rotationMatrix, rotation[0], rotation[1], rotation[2]);

	Mat reshaped = currentShape.reshape(1, currentShape.rows / 3).t();

	Mat rotatedShape = (rotationMatrix * reshaped);

	double minX;
	double maxX;
	cv::minMaxLoc(rotatedShape.row(0), &minX, &maxX);	

	double minY;
	double maxY;
	cv::minMaxLoc(rotatedShape.row(1), &minY, &maxY);

	double width = abs(minX - maxX);
	double height = abs(minY - maxY);

	double scaling = ((roi.width / width) + (roi.height / height)) / 2;

	double tx = roi.x - (minX * scaling);
	double ty = roi.y - (minY * scaling);

	pGlobal.create(6, 1, CV_64F);

	pGlobal.at<double>(0) = scaling;
	pGlobal.at<double>(1) = rotation[0];
	pGlobal.at<double>(2) = rotation[1];
	pGlobal.at<double>(3) = rotation[2];
	pGlobal.at<double>(4) = tx;
	pGlobal.at<double>(5) = ty;
}

void PDM::Draw(cv::Mat img, const Mat& plocal, const Mat_<double>& pLocalMoph, const Mat& pglobl, Mat_<int>& triangulation)
{

	Mat shape2D;

	CalcShape2D(shape2D, plocal, pLocalMoph, pglobl);

	// Draw the calculated shape on top of an image
	Draw(img, shape2D);	

}

void PDM::Draw(cv::Mat img, const Mat& shape2D, Mat_<int>& triangulation)
{
	int n = shape2D.rows/2;

	if(!triangulation.empty())
	{
		//draw triangulation
		Scalar c = CV_RGB(0,0,255);
		for(int i = 0; i < triangulation.rows; i++)
		{
			Point p1 = cv::Point((int)shape2D.at<double>(triangulation.at<int>(i,0),0), (int)shape2D.at<double>(triangulation.at<int>(i,0)+n,0));
			Point p2 = cv::Point((int)shape2D.at<double>(triangulation.at<int>(i,1),0), (int)shape2D.at<double>(triangulation.at<int>(i,1)+n,0));
			cv::line(img,p1,p2,c);

			p1 = cv::Point((int)shape2D.at<double>(triangulation.at<int>(i,0),0), (int)shape2D.at<double>(triangulation.at<int>(i,0)+n,0));
			p2 = cv::Point((int)shape2D.at<double>(triangulation.at<int>(i,2),0), (int)shape2D.at<double>(triangulation.at<int>(i,2)+n,0));
			cv::line(img,p1,p2,c);

			p1 = cv::Point((int)shape2D.at<double>(triangulation.at<int>(i,2),0), (int)shape2D.at<double>(triangulation.at<int>(i,2)+n,0));
			p2 = cv::Point((int)shape2D.at<double>(triangulation.at<int>(i,1),0), (int)shape2D.at<double>(triangulation.at<int>(i,1)+n,0));
			cv::line(img,p1,p2,c);
		}
	}

	for( int i = 0; i < n; ++i)
	{

		Point featurePoint((int)shape2D.at<double>(i), (int)shape2D.at<double>(i +n));

		cv::circle(img, featurePoint, 2, Scalar(0,0,255), 1);
	}
	
}

void PDM::Identity(Mat& plocal, Mat_<double>& pLocalMoph, Mat& pglobl)
{
  plocal = Mat::zeros(_V.cols,1,CV_64F);
  pLocalMoph = Mat_<double>::zeros(_Vmorph.cols, 1);

  pglobl = (Mat_<double>(6,1) << 1, 0, 0, 0, 0, 0);
  
}
//===========================================================================
// Calculate the PDM's Jacobian over rigid parameters (rotation, translation and scaling) (ideally this would be over rotation and location instead, actual 3D parameters)
void PDM::CalcRigidJacob(const Mat& plocal, const Mat_<double>& pLocalMoph, const Mat& pglobl,cv::Mat &Jacob)
{
  int i;
  
  // number of verts
  int n = _M.rows/3;
  
  // number of principal components
  int m = _V.cols;
  
  double X,Y,Z;

  assert((plocal.rows == m)  && (plocal.cols == 1) && (pglobl.rows == 6)  && (pglobl.cols == 1) && (Jacob.rows == 2*n) && (Jacob.cols == 6));
  
  double rx[3][3] = {{0,0,0},{0,0,-1},{0,1,0}};
  cv::Mat Rx(3,3,CV_64F,rx);

  double ry[3][3] = {{0,0,1},{0,0,0},{-1,0,0}};
  cv::Mat Ry(3,3,CV_64F,ry);
  
  double rz[3][3] = {{0,-1,0},{1,0,0},{0,0,0}};
  cv::Mat Rz(3,3,CV_64F,rz);

  double s = pglobl.db(0,0);

  // Get the current 3D shape (not affected by global transform)
  this->CalcShape3D(S_, plocal, pLocalMoph);

  // get the rotation matrix
  Euler2Rot(R_,pglobl); 

  // Somehow this is a linearised approximation of rotation
  P_ = s*R_(cv::Rect(0,0,3,2)); // ignoring the z - component when calculating the jacobian

  Px_ = P_*Rx; 
  Py_ = P_*Ry;
  Pz_ = P_*Rz;

  assert(R_.isContinuous() && Px_.isContinuous() && Py_.isContinuous() && Pz_.isContinuous());

  const double* px = Px_.ptr<double>(0);
  const double* py = Py_.ptr<double>(0);
  const double* pz = Pz_.ptr<double>(0);
  const double* r  =  R_.ptr<double>(0);

  cv::MatIterator_<double> Jx = Jacob.begin<double>();
  cv::MatIterator_<double> Jy = Jx + n*6;

  for(i = 0; i < n; i++)
  {
    
	X=S_.db(i,0);
	Y=S_.db(i+n,0);
	Z=S_.db(i+n*2,0);    

    *Jx++ =  r[0]*X +  r[1]*Y +  r[2]*Z;
    *Jy++ =  r[3]*X +  r[4]*Y +  r[5]*Z;
    *Jx++ = px[0]*X + px[1]*Y + px[2]*Z;
    *Jy++ = px[3]*X + px[4]*Y + px[5]*Z;
    *Jx++ = py[0]*X + py[1]*Y + py[2]*Z;
    *Jy++ = py[3]*X + py[4]*Y + py[5]*Z;
    *Jx++ = pz[0]*X + pz[1]*Y + pz[2]*Z;
    *Jy++ = pz[3]*X + pz[4]*Y + pz[5]*Z;
    
	*Jx++ = 1.0;
	*Jy++ = 0.0;
	*Jx++ = 0.0;
	*Jy++ = 1.0;

  }
  
  //Jacob = Jacob / 2.0;

}

void PDM::CalcMorphologyJacob(const Mat_<double>& pLocalMorph, const Mat& pglobl, Mat_<double> &Jacob)
{
	int i,j;

	int n = _M.rows/3;
	int m = _Vmorph.cols;
	double X,Y,Z;
	assert((pLocalMorph.rows == m)  && (pLocalMorph.cols == 1) && (pglobl.rows == 6)  && (pglobl.cols == 1) && (Jacob.rows == 2*n) && (Jacob.cols == 6+m));

	double s = pglobl.db(0,0);

	double rx[3][3] = {{0,0,0},{0,0,-1},{0,1,0}}; cv::Mat Rx(3,3,CV_64F,rx);
	double ry[3][3] = {{0,0,1},{0,0,0},{-1,0,0}}; cv::Mat Ry(3,3,CV_64F,ry);
	double rz[3][3] = {{0,-1,0},{1,0,0},{0,0,0}}; cv::Mat Rz(3,3,CV_64F,rz);

	Mat plocal = Mat::zeros(this->NumberOfModesExpr(), 1, CV_64F);

	this->CalcShape3D(S_, plocal, pLocalMorph);
	
	Euler2Rot(R_, pglobl); 

	P_ = s*R_(cv::Rect(0,0,3,2)); Px_ = P_*Rx; Py_ = P_*Ry; Pz_ = P_*Rz;

	assert(R_.isContinuous() && Px_.isContinuous() && Py_.isContinuous() && Pz_.isContinuous() && P_.isContinuous());

	const double* px = Px_.ptr<double>(0);
	const double* py = Py_.ptr<double>(0);
	const double* pz = Pz_.ptr<double>(0);
	const double* p  =  P_.ptr<double>(0);
	const double* r  =  R_.ptr<double>(0);

	cv::MatIterator_<double> Jx =  Jacob.begin();
	cv::MatIterator_<double> Jy =  Jx + n*(6+m);
	cv::MatIterator_<double> Vx =  _Vmorph.begin();
	cv::MatIterator_<double> Vy =  Vx + n*m;
	cv::MatIterator_<double> Vz =  Vy + n*m;

	for(i = 0; i < n; i++)
	{
    
		X=S_.db(i,0);
		Y=S_.db(i+n,0);
		Z=S_.db(i+n*2,0);    
    
		*Jx++ =  r[0]*X +  r[1]*Y +  r[2]*Z;
		*Jy++ =  r[3]*X +  r[4]*Y +  r[5]*Z;
		*Jx++ = px[0]*X + px[1]*Y + px[2]*Z;
		*Jy++ = px[3]*X + px[4]*Y + px[5]*Z;
		*Jx++ = py[0]*X + py[1]*Y + py[2]*Z;
		*Jy++ = py[3]*X + py[4]*Y + py[5]*Z;
		*Jx++ = pz[0]*X + pz[1]*Y + pz[2]*Z;
		*Jy++ = pz[3]*X + pz[4]*Y + pz[5]*Z;

		*Jx++ = 1.0;
		*Jy++ = 0.0;
		*Jx++ = 0.0;
		*Jy++ = 1.0;

		for(j = 0; j < m; j++,++Vx,++Vy,++Vz)
		{
			*Jx++ = p[0]*(*Vx) + p[1]*(*Vy) + p[2]*(*Vz);
			*Jy++ = p[3]*(*Vx) + p[4]*(*Vy) + p[5]*(*Vz);
		}
	}
}
//===========================================================================
void PDM::CalcJacob(const Mat& plocal, const Mat_<double>& pLocalMorph, const Mat& pglobl, cv::Mat &Jacob)
{ 
	int i,j,n = _M.rows/3,m = _V.cols; double X,Y,Z;
	assert((plocal.rows == m)  && (plocal.cols == 1) && (pglobl.rows == 6)  && (pglobl.cols == 1) && (Jacob.rows == 2*n) && (Jacob.cols == 6+m));

	double s = pglobl.db(0,0);

	double rx[3][3] = {{0,0,0},{0,0,-1},{0,1,0}}; cv::Mat Rx(3,3,CV_64F,rx);
	double ry[3][3] = {{0,0,1},{0,0,0},{-1,0,0}}; cv::Mat Ry(3,3,CV_64F,ry);
	double rz[3][3] = {{0,-1,0},{1,0,0},{0,0,0}}; cv::Mat Rz(3,3,CV_64F,rz);

	this->CalcShape3D(S_, plocal, pLocalMorph);
	
	Euler2Rot(R_, pglobl); 

	P_ = s*R_(cv::Rect(0,0,3,2)); Px_ = P_*Rx; Py_ = P_*Ry; Pz_ = P_*Rz;

	assert(R_.isContinuous() && Px_.isContinuous() && Py_.isContinuous() && Pz_.isContinuous() && P_.isContinuous());

	const double* px = Px_.ptr<double>(0);
	const double* py = Py_.ptr<double>(0);
	const double* pz = Pz_.ptr<double>(0);
	const double* p  =  P_.ptr<double>(0);
	const double* r  =  R_.ptr<double>(0);

	cv::MatIterator_<double> Jx =  Jacob.begin<double>();
	cv::MatIterator_<double> Jy =  Jx + n*(6+m);
	cv::MatIterator_<double> Vx =  _V.begin<double>();
	cv::MatIterator_<double> Vy =  Vx + n*m;
	cv::MatIterator_<double> Vz =  Vy + n*m;

	for(i = 0; i < n; i++)
	{
    
		X=S_.db(i,0);
		Y=S_.db(i+n,0);
		Z=S_.db(i+n*2,0);    
    
		*Jx++ =  r[0]*X +  r[1]*Y +  r[2]*Z;
		*Jy++ =  r[3]*X +  r[4]*Y +  r[5]*Z;
		*Jx++ = px[0]*X + px[1]*Y + px[2]*Z;
		*Jy++ = px[3]*X + px[4]*Y + px[5]*Z;
		*Jx++ = py[0]*X + py[1]*Y + py[2]*Z;
		*Jy++ = py[3]*X + py[4]*Y + py[5]*Z;
		*Jx++ = pz[0]*X + pz[1]*Y + pz[2]*Z;
		*Jy++ = pz[3]*X + pz[4]*Y + pz[5]*Z;

		*Jx++ = 1.0;
		*Jy++ = 0.0;
		*Jx++ = 0.0;
		*Jy++ = 1.0;

		for(j = 0; j < m; j++,++Vx,++Vy,++Vz)
		{
			*Jx++ = p[0]*(*Vx) + p[1]*(*Vy) + p[2]*(*Vz);
			*Jy++ = p[3]*(*Vx) + p[4]*(*Vy) + p[5]*(*Vz);
		}
	}
}

//===========================================================================
// Updating the parameters
void PDM::CalcReferenceUpdate(cv::Mat &dp,cv::Mat &plocal,cv::Mat &pglobl, cv::Mat& V)
{
  assert((dp.rows == 6+V.cols) && (dp.cols == 1));

  plocal += dp(cv::Rect(0,6,1,V.cols));

  pglobl.db(0,0) += dp.db(0,0);
  pglobl.db(4,0) += dp.db(4,0);
  pglobl.db(5,0) += dp.db(5,0);

  Euler2Rot(R1_,pglobl);
  R2_ = cv::Mat::eye(3,3,CV_64F);

  R2_.db(1,2) = -1.0*(R2_.db(2,1) = dp.db(1,0));
  R2_.db(2,1) = -1.0*(R2_.db(0,2) = dp.db(2,0));
  R2_.db(0,1) = -1.0*(R2_.db(1,0) = dp.db(3,0));
  MetricUpgrade(R2_);
  R3_ = R1_*R2_;
  Rot2Euler(R3_,pglobl);
  return;
}
//===========================================================================
void PDM::ApplySimT(double a, double b, double tx, double ty, cv::Mat &pglobl)
{
	assert((pglobl.rows == 6) && (pglobl.cols == 1) && (pglobl.type()==CV_64F));

	// roll
	double angle = atan2(b,a);

	// scale
	double scale = a/cos(angle);

	double ca = cos(angle);
	double sa = sin(angle);
	double xc = pglobl.db(4,0);
	double yc = pglobl.db(5,0);

	// Create a rotation matrix for roll
	R1_ = cv::Scalar(0);
	R1_.db(2,2) = 1.0;
	R1_.db(0,0) =  ca;
	R1_.db(0,1) = -sa;
	R1_.db(1,0) =  sa;
	R1_.db(1,1) =  ca;
	
	// combine it with the existing pglobl matrix
	Euler2Rot(R2_,pglobl);
	R3_ = R1_*R2_; 

	pglobl.db(0,0) *= scale;

	// apply that matrix to the pglobl
	Rot2Euler(R3_,pglobl);

	// rotation corrected tx and ty
	pglobl.db(4,0) = a*xc - b*yc + tx; 
	pglobl.db(5,0) = b*xc + a*yc + ty;

	return;
}

void PDM::Read(string location)
{
  	
	ifstream pdmLoc(location);

	IO::SkipComments(pdmLoc);

	// Reading mean values
	IO::ReadMat(pdmLoc,_M);
	
	IO::SkipComments(pdmLoc);

	// Reading principal components
	IO::ReadMat(pdmLoc,_V);
	
	IO::SkipComments(pdmLoc);
	
	// Reading eigenvalues	
	IO::ReadMat(pdmLoc,_E);	

	// Check if morphology is defined here as well
	IO::SkipComments(pdmLoc);
	
	if(!pdmLoc.eof())
	{
		IO::ReadMat(pdmLoc,_Vmorph);
		IO::SkipComments(pdmLoc);
		IO::ReadMat(pdmLoc,_Emorph);
	}

	// The 3D instance of a model
	S_.create(_M.rows,1,CV_64F);  

	// Rotation matrix describing the model
	R_.create(3,3,CV_64F); 
	
	//s_.create(_M.rows,1,CV_64F);
	
	P_.create(2,3,CV_64F);
	Px_.create(2,3,CV_64F);
	Py_.create(2,3,CV_64F);
	Pz_.create(2,3,CV_64F);
	R1_.create(3,3,CV_64F);
	R2_.create(3,3,CV_64F);
	R3_.create(3,3,CV_64F);

}
