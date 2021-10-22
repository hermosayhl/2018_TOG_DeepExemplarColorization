
mkdir -p ./bin


OPENCV_DIR="/usr/local/opencv/opencv-4.5.2"

CAFFE_DIR="/home/liuchang/tools/C++/caffe"


nvcc ./src/linux/deep_image_analogy/source/*.cpp ./src/linux/deep_image_analogy/source/*.cu -I${OPENCV_DIR}/include/opencv4 -I${CAFFE_DIR}/include -I${CAFFE_DIR}/build/src -L${CAFFE_DIR}/build/lib \
	-o ./bin/deep_image_analogy -std=c++11\
    -I/usr/include/eigen3 -L${OPENCV_DIR}/lib -lopencv_core -lopencv_highgui -lopencv_imgproc \
    -lopencv_imgcodecs -lboost_system -lboost_filesystem -lcublas -lcaffe -lglog \
    -Wno-deprecated-gpu-targets


nvcc ./src/linux/similarity_combo/source/*.cpp ./src/linux/similarity_combo/source/*.cu -I${OPENCV_DIR}/include/opencv4 -I${CAFFE_DIR}/include -I${CAFFE_DIR}/build/src -L${CAFFE_DIR}/.build_release/lib \
	-o ./bin/similarity_combo -std=c++11\
    -I/usr/include/eigen3 -L${OPENCV_DIR}/lib -lopencv_core -lopencv_highgui -lopencv_imgproc \
    -lopencv_imgcodecs -lboost_system -lboost_filesystem -lcublas -lcaffe -lglog \
    -Wno-deprecated-gpu-targets


