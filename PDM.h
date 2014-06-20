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

#ifndef __PDM_h_
#define __PDM_h_
#include "IO.h"

using namespace cv;

namespace CLMTracker
{
  //===========================================================================
  /** 
      A 3D Point Distribution Model
  */
class PDM{
	public:    
    
		cv::Mat _M; /**< mean 3D shape vector [x1,..,xn,y1,...yn]      */
  
		cv::Mat _V; /**< basis of variation                            */
		cv::Mat _E; /**< vector of eigenvalues (row vector)            */

		cv::Mat_<double> _Vmorph; /** basis of variation of the morphology of the face (static per video, so fitting on this is only done once) */
		cv::Mat_<double> _Emorph; /** eigenvalues */
	
		PDM(){;}
		
		void Read(string location);

		// Number of vertices
		inline int NumberOfPoints(){return _M.rows/3;}
		
		// Listing the number of modes of variation
		inline int NumberOfModesExpr(){return _V.cols;}
		inline int NumberOfModesMorph(){return _Vmorph.cols;}


		void Clamp(Mat& p, double c, const Mat& E);
		void Identity(Mat& plocal, Mat_<double>& pLocalMoph, Mat& pglobl);
		void CalcShape3D(Mat& s, const Mat& plocal, const Mat_<double>& pLocalMoph);
		void CalcShape2D(Mat& s, const Mat& plocal, const Mat_<double>& pLocalMoph, const Mat& pglobl);
    
		//void CalcParams(cv::Mat &s,cv::Mat &plocal,cv::Mat &pglobl);
		// provided the region of interest, and the local parameters (with optional rotation), generates the global parameters that can generate the face in there
		void CalcParams(Mat& pGlobal, const Rect& roi, const Mat& pLocal, const Mat_<double>& pLocalMoph, const Vec3d& rotation = Vec3d(0.0));

		void CalcRigidJacob(const Mat& plocal, const Mat_<double>& pLocalMoph, const Mat& pglobl, Mat &Jacob);
		void CalcMorphologyJacob(const Mat_<double>& pLocalMoph, const Mat& pglobl, Mat_<double> &Jacob);
		void CalcJacob(const Mat& plocal, const Mat_<double>& pLocalMoph, const Mat& pglobl, Mat &Jacob);

		void CalcReferenceUpdate(cv::Mat &dp, cv::Mat &plocal, cv::Mat &pglobl, cv::Mat& V);
		void ApplySimT(double a,double b,double tx,double ty,cv::Mat &pglobl);
    	
		// Helper visualisation functions
		void Draw(cv::Mat img, const Mat& plocal, const Mat_<double>& pLocalMoph, const Mat& pglobl, Mat_<int>& triangulation = Mat_<int>());
		void Draw(cv::Mat img, const Mat& shape, Mat_<int>& triangulation = Mat_<int>());

  private:
    
	cv::Mat S_; // The 3D instance of a model
	cv::Mat R_; // Rotation matrix describing the current rotation

	cv::Mat P_,Px_,Py_,Pz_,R1_,R2_,R3_; // the jacobian approximations (I think)
  };
  //===========================================================================
}
#endif
