cmake_minimum_required(VERSION 2.8)
project( CLM )
find_package( OpenCV REQUIRED )

#file(GLOB PoseDetectorHaar_SOURSES	
#	PoseDetectorHaar.*
#	PoseDetectorHaarParameters.*
#)
#add_library(PoseDetectorHaar STATIC ${PoseDetectorHaar_SOURSES} ${OpenCV_LIBS})
#target_link_libraries( PoseDetectorHaar ${OpenCV_LIBS} )

set(KINECT_LIB "C:/Program Files/Microsoft SDKs/Kinect/v2.0-DevPreview1404" CACHE FILEPATH "description")
FIND_LIBRARY(Kinect20_LIBRARIES NAMES Kinect20 Kinect20.lib PATHS "${KINECT_LIB}/lib/x64" DOC "Freetype library")
FIND_PATH(Kinect20_INCLUDE_DIRS Kinect.h "${KINECT_LIB}/inc" DOC "Freetype includes")
INCLUDE_DIRECTORIES(${Kinect20_INCLUDE_DIRS})

file(GLOB K4W_OPENCV_SOURSES	
	K4Wv2OpenCVModule.*
)

file(GLOB PoseDetectorHaar_SOURSES	
	PoseDetectorHaar.*
	PoseDetectorHaarParameters.*
	CLM.*
	FCheck.*
	IO.*
	Patch.*
	PAW.*
	PDM.*
	TrackerCLM.*
	CLMTracker.*
	CLMParameters.*
)

include_directories( ./ )

add_library(KinecToOpenCV ${K4W_OPENCV_SOURSES} ${Kinect20_LIBRARIES})
add_library( FaceTrackerLib STATIC ${PoseDetectorHaar_SOURSES} )

add_executable( FaceTrackerSample SimpleCLM.cpp )
target_link_libraries( FaceTrackerSample ${OpenCV_LIBS} FaceTrackerLib KinecToOpenCV ${Kinect20_LIBRARIES})

