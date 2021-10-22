# middlebury - noise07 - level1T

INPUT_DIR="./example/middlebury/input"
REFER_DIR="./example/middlebury/reference"

RESULT_DIR="./results/noise07/level1T/middlebury"

./bin/deep_image_analogy 	models/deep_image_analogy/	 ${INPUT_DIR}	 ${REFER_DIR} 	${RESULT_DIR}/flow
./bin/similarity_combo	 models/similarity_subnet/	 ${INPUT_DIR} ${REFER_DIR}	 ${RESULT_DIR}/flow 	${RESULT_DIR}/combo
python3 ../colorization_subnet/test.py --input_dir ${INPUT_DIR} --refer_dir ${REFER_DIR} --combo_dir ${RESULT_DIR}/combo --out_dir ${RESULT_DIR}/colored_result  --short_size 256 --test_model models/colorization_subnet/example_net.pth --gpu_id 0
