# #!/bin/bash

# mkdir -p /src/app/demo/bin

# # Make Similarity Combo
# SOURCE=/src/app/similarity_subnet/linux/similarity_combo/source

# nvcc $SOURCE/*.cpp $SOURCE/*.cu -I/src/app/caffe/include -I/src/app/caffe/build/src -L/src/app/caffe/build/lib -o /src/app/demo/bin/similarity_combo -std=c++11\
#     -I/usr/include/eigen3 -lopencv_core -lopencv_highgui -lopencv_imgproc \
#     -lopencv_imgcodecs -lboost_system -lboost_filesystem -lcublas -lcaffe -lglog \
#     -Wno-deprecated-gpu-targets

# # Make Deep Image Analogy
# SOURCE=/src/app/similarity_subnet/linux/deep_image_analogy/source

# nvcc $SOURCE/*.cpp $SOURCE/*.cu -I/src/app/caffe/include -I/src/app/caffe/build/src -L/src/app/caffe/build/lib -o /src/app/demo/bin/deep_image_analogy -std=c++11\
#     -I/usr/include/eigen3 -lopencv_core -lopencv_highgui -lopencv_imgproc \
#     -lopencv_imgcodecs -lboost_system -lboost_filesystem -lcublas -lcaffe -lglog \
#     -Wno-deprecated-gpu-targets


#!/bin/bash

mkdir -p ./bin



nvcc ../linux/similarity_combo/source/*.cpp ../linux/similarity_combo/source/*.cu -I/usr/local/opencv/opencv-4.5.2/include/opencv4 -I/home/liuchang/tools/C++/caffe/include -I/home/liuchang/tools/C++/caffe/build/src -L/home/liuchang/tools/C++/caffe/.build_release/lib -o ./bin/similarity_combo -std=c++11\
    -I/usr/include/eigen3 -L/usr/local/opencv/opencv-4.5.2/lib -lopencv_core -lopencv_highgui -lopencv_imgproc \
    -lopencv_imgcodecs -lboost_system -lboost_filesystem -lcublas -lcaffe -lglog \
    -Wno-deprecated-gpu-targets

export LD_LIBRARY_PATH=/home/liuchang/tools/C++/caffe/.build_release/lib:$LD_LIBRARY_PATH
# ./bin/similarity_combo


nvcc ../linux/deep_image_analogy/source/*.cpp ../linux/deep_image_analogy/source/*.cu -I/usr/local/opencv/opencv-4.5.2/include/opencv4 -I/home/liuchang/tools/C++/caffe/include -I/home/liuchang/tools/C++/caffe/build/src -L/home/liuchang/tools/C++/caffe/build/lib -o ./bin/deep_image_analogy -std=c++11\
    -I/usr/include/eigen3 -L/usr/local/opencv/opencv-4.5.2/lib -lopencv_core -lopencv_highgui -lopencv_imgproc \
    -lopencv_imgcodecs -lboost_system -lboost_filesystem -lcublas -lcaffe -lglog \
    -Wno-deprecated-gpu-targets
