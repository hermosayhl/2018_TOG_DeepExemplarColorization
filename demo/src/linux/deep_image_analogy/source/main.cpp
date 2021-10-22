#include "DeepAnalogy.cuh"
#include "Classifier.h"
#include <boost/filesystem.hpp>
#include <vector>



bool make_full_dir(const string& path) {
	if (path.empty())
	{
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


void run_flow(int argc, char** argv) {

	std::string pretrained_model_dir(argv[1]);
	std::string input_images_dir(argv[2]);
	std::string refer_images_dir(argv[3]);
	std::string flow_results_dir(argv[4]);

	if(input_images_dir.back() != '/') input_images_dir += '/';
	if(refer_images_dir.back() != '/') refer_images_dir += '/';
	if(flow_results_dir.back() != '/') flow_results_dir += '/';

	printf("pretrained_model_dir  :  %s\ninput_images_dir  :  %s\nrefer_images_dir  :  %s\nflow_results_dir  :  %s\n", pretrained_model_dir.c_str(), input_images_dir.c_str(), refer_images_dir.c_str(), flow_results_dir.c_str());
	
	// 创建文件夹
	if (not boost::filesystem::exists(flow_results_dir)){
		// boost::filesystem::create_directory(flow_results_dir);
		make_full_dir(flow_results_dir);
	}

	// 获取所有文件名字
	const auto filenames = get_filenames(input_images_dir);
	// for(const auto& it : filenames)
	// 	std::cout << it << std::endl;

	// return;

	::google::InitGoogleLogging("deepanalogy");

	DeepAnalogy dp;
	dp.SetModel(pretrained_model_dir);
	dp.SetGPU(0);

	string model_file = "vgg19/VGG_ILSVRC_19_layers_deploy.prototxt";
	string trained_file = "vgg19/VGG_ILSVRC_19_layers.caffemodel";

	Classifier classifier_A(pretrained_model_dir + model_file, pretrained_model_dir + trained_file);
	Classifier classifier_B(pretrained_model_dir + model_file, pretrained_model_dir + trained_file);

	const int total_size = int(filenames.size());
	for(int cnt = 0;cnt < total_size; ++cnt) {

		printf("%d/%d  ===>  processing  %s\n", cnt + 1, total_size, filenames[cnt].c_str());

		const auto input_image = input_images_dir + filenames[cnt];
		const auto refer_image = refer_images_dir + filenames[cnt];

		dp.SetA(input_image);
		dp.SetBPrime(refer_image);
		dp.SetOutputDir(flow_results_dir);
		dp.SetRatio(1.0);
		dp.SetBlendWeight(2);
		dp.UsePhotoTransfer(false);
		dp.LoadInputs();
		dp.ComputeAnn(classifier_A, classifier_B);
	}

	// for (int i = sid; i < eid; ++i)
	// {
	// 	int val = fscanf(fp, "%s %s %f\n", name0, name1, &score);
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

	// 	printf("A   :  %s\n", A.c_str());
	// 	printf("BP  :  %s\n", BP.c_str());

	// 	// pos0 = name0Str.find_last_of(".");
	// 	// pos1 = name1Str.find_last_of(".");
	// 	// string name_A = name0Str.substr(0, pos0);
	// 	// string name_B = name1Str.substr(0, pos1);

	// 	// string flow1 = flow_results_dir + name_A + "_" + name_B + ".txt";
	// 	// string flow2 = flow_results_dir + name_B + "_" + name_A + ".txt";

	// 	dp.SetA(A);
	// 	dp.SetBPrime(BP);
	// 	dp.SetOutputDir(flow_results_dir);
	// 	dp.SetRatio(1.0);
	// 	dp.SetBlendWeight(2);
	// 	dp.UsePhotoTransfer(false);
	// 	dp.LoadInputs();
	// 	dp.ComputeAnn(classifier_A, classifier_B);
	// }

	google::ShutdownGoogleLogging();

	classifier_A.DeleteNet();
	classifier_B.DeleteNet();
}

int main(int argc, char** argv) 
{
	run_flow(argc, argv);
	return 0;
}



// #include "DeepAnalogy.cuh"
// #include "Classifier.h"
// #include <boost/filesystem.hpp>

// void run_flow(int argc, char** argv)
// {
// 	string pretrained_model_dir = argv[1];
// 	string rootDir = argv[2];

// 	int sid = atoi(argv[3]);
// 	int eid = atoi(argv[4]);
// 	int gid = atoi(argv[5]);

// 	printf("sid = %d\neid = %d\ngid = %d\n", sid, eid, gid);

// 	string postfix = "";
// 	if (argc >= 8)
// 	{
// 		postfix = "_" + string(argv[7]);
// 	}

// 	string fname = rootDir + "/pairs" + postfix + ".txt";
// 	FILE* fp = fopen(fname.c_str(), "r");
// 	char name0[260], name1[260];
// 	float score = 0.f;
// 	int val = 1;
// 	for (int i = 0; i < sid; ++i)
// 	{
// 		val = fscanf(fp, "%s %s %f\n", name0, name1, &score);
// 		if (val == EOF) break;
// 	}

// 	if (val == EOF)
// 		return;

// 	string inputDir = rootDir + "/input" + postfix + "/";
// 	string flow_results_dir = rootDir + "/flow" + postfix + "/";

// 	printf("inputDir :  %s\noutputDir  :  %s\n", inputDir.c_str(), flow_results_dir.c_str());

// 	if (!boost::filesystem::exists(flow_results_dir))
// 	{
// 		boost::filesystem::create_directory(flow_results_dir);
// 	}

// 	::google::InitGoogleLogging("deepanalogy");

// 	DeepAnalogy dp;
// 	dp.SetModel(pretrained_model_dir);
// 	dp.SetGPU(gid);

// 	string model_file = "vgg19/VGG_ILSVRC_19_layers_deploy.prototxt";
// 	string trained_file = "vgg19/VGG_ILSVRC_19_layers.caffemodel";

// 	Classifier classifier_A(pretrained_model_dir + model_file, pretrained_model_dir + trained_file);
// 	Classifier classifier_B(pretrained_model_dir + model_file, pretrained_model_dir + trained_file);

// 	for (int i = sid; i < eid; ++i)
// 	{
// 		int val = fscanf(fp, "%s %s %f\n", name0, name1, &score);
// 		if (val == EOF) break;

// 		string name0Str(name0);
// 		string name1Str(name1);

// 		int pos0 = name0Str.find_last_of(".");
// 		int pos1 = name1Str.find_last_of(".");
// 		if (name0Str.length() - pos0 <= 4)
// 		{
// 			name0Str = name0Str.substr(0, pos0) + ".jpg";
// 			name1Str = name1Str.substr(0, pos1) + ".jpg";
// 		}

// 		string A = inputDir + name0Str;
// 		string BP = inputDir + name1Str;

// 		printf("A   :  %s\n", A.c_str());
// 		printf("BP  :  %s\n", BP.c_str());

// 		// pos0 = name0Str.find_last_of(".");
// 		// pos1 = name1Str.find_last_of(".");
// 		// string name_A = name0Str.substr(0, pos0);
// 		// string name_B = name1Str.substr(0, pos1);

// 		// string flow1 = flow_results_dir + name_A + "_" + name_B + ".txt";
// 		// string flow2 = flow_results_dir + name_B + "_" + name_A + ".txt";

// 		dp.SetA(A);
// 		dp.SetBPrime(BP);
// 		dp.SetOutputDir(flow_results_dir);
// 		dp.SetRatio(1.0);
// 		dp.SetBlendWeight(2);
// 		dp.UsePhotoTransfer(false);
// 		dp.LoadInputs();
// 		dp.ComputeAnn(classifier_A, classifier_B);
// 	}

// 	google::ShutdownGoogleLogging();

// 	classifier_A.DeleteNet();
// 	classifier_B.DeleteNet();
// 	fclose(fp);
// }

// int main(int argc, char** argv) 
// {
// 	run_flow(argc, argv);
// 	return 0;
// }
