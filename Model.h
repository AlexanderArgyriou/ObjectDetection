#pragma once
#include "pch.h"

using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::AI::MachineLearning;
using namespace Windows::Foundation::Collections;
using namespace Windows::Graphics::Imaging;
using namespace Windows::Media;
using namespace Windows::Storage;
using namespace std;

class Model
{
	private:
		LearningModel                    M;
		hstring                  ModelPath;
		hstring                    ImgPath;
		string                     LblPath;
		VideoFrame                   Image;
		LearningModelSession       Session;
		LearningModelBinding       Binding;
		LearningModelDeviceKind DeviceKind;
		vector<string>              Labels;

		inline void PrintResults(const IVectorView<float> &Results) noexcept;       // PrintResults
		inline void LoadLabels() noexcept;                                          // LoadLabels

	public:
		inline Model(const hstring &newModelPath, 
			const hstring &newImgPath, 
			const string &newLblPath) noexcept;             // Constructor
		inline void LoadModel() noexcept;                   // LoadModel
		inline void LoadImg() noexcept;                     // LoadImage
		inline void BindModel() noexcept;                   // BindModel
		inline void EvaluateModel() noexcept;               // Evaluate
};	// Model

inline Model::Model(const hstring &newModelPath, const hstring &newImgPath,
	const string &newLblPath) noexcept
	: ModelPath(newModelPath), ImgPath(newImgPath),
	  LblPath(newLblPath),
	  M(nullptr), Image(nullptr),
	  DeviceKind(LearningModelDeviceKind::Default),
	  Session(nullptr),
	  Binding(nullptr),
	  Labels(1'000)
{}	// Constructor

inline void Model::LoadModel() noexcept
{
	auto start = chrono::steady_clock::now();

	M = LearningModel::LoadFromFilePath(ModelPath);

	auto end = chrono::steady_clock::now();

	cout << "Time to load SqueezeNet pretrained model : "
		<< chrono::duration_cast<chrono::milliseconds>(end - start).count()
		<< " msec" << endl;
}	// LoadModel

inline void Model::LoadImg() noexcept
{
	auto start = chrono::steady_clock::now();

	StorageFile File = StorageFile::GetFileFromPathAsync(ImgPath).get();
	auto Stream = File.OpenAsync(FileAccessMode::Read).get();
	BitmapDecoder Decoder = BitmapDecoder::CreateAsync(Stream).get();
	SoftwareBitmap Bitmap = Decoder.GetSoftwareBitmapAsync().get();
	Image = VideoFrame::CreateWithSoftwareBitmap(Bitmap);
	Stream.Close();

	auto end = chrono::steady_clock::now();

	cout << "Time to load Image : "
		<< chrono::duration_cast<chrono::milliseconds>(end - start).count()
		<< " msec" << endl;
}	// LoadImg

inline void Model::BindModel() noexcept
{
	// input feature is named "data_0" 
	// output feature is named "softmaxout_1".
	// You can see these properties for any model by opening them in Netron
	// Re-Construct a Session and a Binding.
	auto start = chrono::steady_clock::now();

	Session = LearningModelSession( M, LearningModelDevice(DeviceKind) );
	Binding = LearningModelBinding( Session );
	Binding.Bind(L"data_0", ImageFeatureValue::CreateFromVideoFrame(Image));
	vector<int64_t> shape({ 1, 1000, 1, 1 });
	Binding.Bind(L"softmaxout_1", TensorFloat::Create(shape));

	auto end = chrono::steady_clock::now();

	cout << "Time to Bind Model : "
		<< chrono::duration_cast<chrono::milliseconds>(end - start).count()
		<< " msec" << endl;
}	// BindModel()

inline void Model::EvaluateModel() noexcept
{
	auto start = chrono::steady_clock::now();

	auto Results = Session.Evaluate(Binding, L"RunId");	// Run

	auto end = chrono::steady_clock::now();

	cout << "Model run took : "
		<< chrono::duration_cast<chrono::milliseconds>(end - start).count()
		<< " msec" << endl;

	auto ResultTensor = Results.Outputs().Lookup(L"softmaxout_1").as<TensorFloat>();
	auto ResultVector = ResultTensor.GetAsVectorView();
	PrintResults(ResultVector);
}	// Evaluate Model

inline void Model::PrintResults(const IVectorView<float> &Results) noexcept
{
	this->LoadLabels();
	vector<float> Top3(3);
	vector<int>   Top3Index(3);

	// 1000 options, with probabilities for each. (How SqueezeNet Works)
	for (int i = 0; i < Results.Size(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (Results.GetAt(i) > Top3[j])
			{
				Top3Index[j] = i;
				Top3[j] = Results.GetAt(i);
				break;
			}	// if
		}	// for
	}	// for

	for (int i = 0; i < 3; i++)
		cout << "-> Guess " << i + 1 << " : "<< Labels[Top3Index[i]].c_str() 
		<< ", with probability : " << Top3[i] * 100 << "%" << endl;
	cout << "-----------------------------------------------------";
	cout << "-----------------------------------------------------" << endl;
}	// PrintResults

inline void Model::LoadLabels() noexcept
{
	ifstream File(LblPath, ifstream::in);
	string S;
	int i = 0;

	while (getline(File, S, ',')) // tokenize
	{
		std::getline(File, S);
		Labels[i++] = S;
	}	// while
}	// LoadLabels