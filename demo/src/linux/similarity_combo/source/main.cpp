// Copyright (c) Microsoft. All rights reserved.

// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#include "Combo.cuh"
#include "Classifier.h"
#include <boost/filesystem.hpp>

#define MAX_LEN 1024

#include <vector>

bool make_full_dir(const string& path) {
	if (path.empty()) {
		return false;
	}
	size_t uPos = path.find_first_of("/");
	while (uPos != string::npos) {
		string dir = path.substr(0, uPos + 1);
		boost::system::error_code ec;
		boost::filesystem::create_directory(dir, ec);
		uPos = path.find_first_of("/", uPos + 1);
	}
	boost::system::error_code ec;
	boost::filesystem::create_directory(path, ec);
	return true;
}


inline std::string get_file_name(const std::string& path) {
	string::size_type iPos = path.find_last_of('/') + 1;
	const auto result = path.substr(iPos, path.length() - iPos);
	return result.substr(0, result.rfind("."));;
}

std::vector<std::string> get_filenames(const std::string& dir) {

	std::vector<std::string> filenames;

	boost::filesystem::path path(dir);
	if (!boost::filesystem::exists(path)) {
		std::cout << dir << " doesn't exists !\n";
		return filenames;
	}
	boost::filesystem::directory_iterator end_iter;
	for (boost::filesystem::directory_iterator iter(path); iter!=end_iter; ++iter) {
		if (boost::filesystem::is_regular_file(iter->status())) {
			const auto path = iter->path().string();
			string::size_type iPos = path.find_last_of('/') + 1;
			string filename = path.substr(iPos, path.length() - iPos);
			filenames.emplace_back(filename);
		}
		if (boost::filesystem::is_directory(iter->status())) {
			get_filenames(iter->path().string());
		}
	}
	return filenames;
}


int ComputeCombo(int argc, char** argv){

	std::string pretrained_model_dir(argv[1]);
	std::string input_images_dir(argv[2]);
	std::string refer_images_dir(argv[3]);
	std::string flow_results_dir(argv[4]);
	std::string combo_results_dir(argv[5]);
	
	if(input_images_dir.back() != '/') input_images_dir += '/';
	if(refer_images_dir.back() != '/') refer_images_dir += '/';
	if(flow_results_dir.back() != '/') flow_results_dir += '/';
	if(combo_results_dir.back() != '/') combo_results_dir += '/';


	printf("pretrained_model_dir  :  %s\ninput_images_dir  :  %s\nrefer_images_dir  :  %s\nflow_results_dir  :  %s\ncombo_results_dir  :  %s\n", pretrained_model_dir.c_str(), input_images_dir.c_str(), refer_images_dir.c_str(), flow_results_dir.c_str(), combo_results_dir.c_str());

	if (not boost::filesystem::exists(combo_results_dir)) {
		// boost::filesystem::create_directory(combo_results_dir);
		make_full_dir(combo_results_dir);
	}

	::google::InitGoogleLogging("ComputeCombo");

	Combo dp;
	dp.SetGPU(0);

	string model_file = "/vgg_19_gray_bn/deploy.prototxt";
	string trained_file = "/vgg_19_gray_bn/vgg19_bn_gray_ft_iter_150000.caffemodel";
	Classifier classifier_A(pretrained_model_dir + model_file, pretrained_model_dir + trained_file);
	Classifier classifier_B(pretrained_model_dir + model_file, pretrained_model_dir + trained_file);


	const auto filenames = get_filenames(input_images_dir);

	const int total_size = int(filenames.size());

	for(int cnt = 0;cnt < total_size; ++cnt) {
		printf("%d/%d  ===>  processing  %s\n", cnt + 1, total_size, filenames[cnt].c_str());

		const auto input_image = input_images_dir + filenames[cnt];
		const auto refer_image = refer_images_dir + filenames[cnt];
		const auto flow_file = flow_results_dir + filenames[cnt];

		char fileName0[100];
		char fileName1[100];
		char ffileName0[100];
		char ffileName1[100];

		// load images
		bool isOKA = dp.LoadA(input_image.c_str());
		bool isOKB = dp.LoadBP(refer_image.c_str());
		if (!isOKA) {
			printf("Error: Fail reading image1: %s!\n", input_image.c_str());
			continue;
		}
		if (!isOKB) {
			printf("Error: Fail reading image2: %s!\n", refer_image.c_str());
			continue;
		}

		int aw, ah, bw, bh;
		dp.GetASize(aw, ah);
		dp.GetBPSize(bw, bh);

		if (aw > MAX_LEN || ah > MAX_LEN) {
			printf("Error: Unsupported image1's size (long edge > 1024): w = %d, h = %d!\n", aw, ah);
			continue;
		}
		if (bw > MAX_LEN || bh > MAX_LEN) {
			printf("Error: Unsupported image size (long edge > 1024): w = %d, h = %d!\n", bw, bh);
			continue;
		}

		// first detect if flow exits
		const auto input_image_id = get_file_name(input_image);
		const auto refer_image_id = get_file_name(refer_image);
		sprintf(ffileName0, "%sinput_%s_refer_%s.txt", flow_results_dir.c_str(), input_image_id.c_str(), refer_image_id.c_str());

		sprintf(ffileName1, "%srefer_%s_input_%s.txt", flow_results_dir.c_str(), refer_image_id.c_str(), input_image_id.c_str());

		if (!boost::filesystem::exists(ffileName0)) {
			printf("Error: Flow %s does not exist!\n", ffileName0);
			continue;
		}
		if (!boost::filesystem::exists(ffileName1)) {
			printf("Error: Flow %s does not exist!\n", ffileName1);
			continue;
		}

		sprintf(fileName0, "%sinput_%s_refer_%s.combo", combo_results_dir.c_str(), input_image_id.c_str(), refer_image_id.c_str());
		sprintf(fileName1, "%srefer_%s_input_%s.combo", combo_results_dir.c_str(), refer_image_id.c_str(), input_image_id.c_str());

		FILE* fp_a = fopen(fileName0, "wb");
		FILE* fp_b = fopen(fileName1, "wb");

		dp.ComputeDist(classifier_A, classifier_B, fp_a, fp_b, ffileName0, ffileName1);

		fclose(fp_a);
		fclose(fp_b);

		// printf("Info: Create %s and %s.\n\n", fileName0, fileName1);
	}

	// for (int i = sid; i < eid; ++i)
	// {
	// 	int val = fscanf(fp, "%s %s %f\n", name0, name1, &score);

	// 	printf("Info: Read line #%d, image1 = %s, image2 = %s.\n", i, name0, name1);
	// 	if (val == EOF) break;

	// 	string name0Str(name0);
	// 	string name1Str(name1);

	// 	int pos0 = name0Str.find_last_of(".");
	// 	int pos1 = name1Str.find_last_of(".");
	// 	if (name0Str.length() - pos0 <= 4)
	// 	{
	// 		name0Str = name0Str.substr(0, pos0) + ".jpg";
	// 		name1Str = name1Str.substr(0, pos1) + ".jpg";
	// 	}

	// 	string A = inputDir + name0Str;
	// 	string BP = inputDir + name1Str;

	// 	pos0 = name0Str.find_last_of(".");
	// 	pos1 = name1Str.find_last_of(".");
	// 	string input_image = name0Str.substr(0, pos0);
	// 	string refer_image = name1Str.substr(0, pos1);

	// 	char fileName0[260];
	// 	char fileName1[260];
	// 	char ffileName0[260];
	// 	char ffileName1[260];

	// 	// load images
	// 	bool isOKA = dp.LoadA(A.c_str());
	// 	bool isOKB = dp.LoadBP(BP.c_str());
	// 	if (!isOKA) {
	// 		printf("Error: Fail reading image1: %s!\n", A.c_str());
	// 		continue;
	// 	}
	// 	if (!isOKB) {
	// 		printf("Error: Fail reading image2: %s!\n", BP.c_str());
	// 		continue;
	// 	}

	// 	int aw, ah, bw, bh;
	// 	dp.GetASize(aw, ah);
	// 	dp.GetBPSize(bw, bh);

	// 	if (aw > MAX_LEN || ah > MAX_LEN) {
	// 		printf("Error: Unsupported image1's size (long edge > 1024): w = %d, h = %d!\n", aw, ah);
	// 		continue;
	// 	}
	// 	if (bw > MAX_LEN || bh > MAX_LEN) {
	// 		printf("Error: Unsupported image size (long edge > 1024): w = %d, h = %d!\n", bw, bh);
	// 		continue;
	// 	}

	// 	// first detect if flow exits
	// 	sprintf(ffileName0, "%s/%s_%s.txt", flow_results_dir.c_str(), input_image.c_str(), refer_image.c_str());

	// 	sprintf(ffileName1, "%s/%s_%s.txt", flow_results_dir.c_str(), refer_image.c_str(), input_image.c_str());

	// 	if (!boost::filesystem::exists(ffileName0)) {
	// 		printf("Error: Flow %s does not exist!\n", ffileName0);
	// 		continue;
	// 	}
	// 	if (!boost::filesystem::exists(ffileName1)) {
	// 		printf("Error: Flow %s does not exist!\n", ffileName1);
	// 		continue;
	// 	}

	// 	// detect if flow is valid
	// 	printf("Info: Flows exist.\n", i);

	// 	sprintf(fileName0, "%s/%s_%s.combo", combo_results_dir.c_str(), input_image.c_str(), refer_image.c_str());
	// 	sprintf(fileName1, "%s/%s_%s.combo", combo_results_dir.c_str(), refer_image.c_str(), input_image.c_str());

	// 	FILE* fp_a = fopen(fileName0, "wb");
	// 	FILE* fp_b = fopen(fileName1, "wb");

	// 	dp.ComputeDist(classifier_A, classifier_B, fp_a, fp_b, ffileName0, ffileName1);

	// 	fclose(fp_a);
	// 	fclose(fp_b);

	// 	printf("Info: Create %s and %s.\n\n", fileName0, fileName1);
	// }

	google::ShutdownGoogleLogging();

	classifier_B.DeleteNet();
	classifier_A.DeleteNet();

	return 0;
}

int main(int argc, char** argv)  {
	ComputeCombo(argc, argv);
	return 0;
}
