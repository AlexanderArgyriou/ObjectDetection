#include "pch.h"
#include "Model.h"

void ProcessModel(const hstring &IPath) noexcept;

int main()
{
    hstring IMG_Path1(L"C:\\Users\\alex\\Desktop\\ML_Object_Detection\\kitten_224.png");
    hstring IMG_Path2(L"C:\\Users\\alex\\Desktop\\ML_Object_Detection\\Rat.jpg");
    hstring IMG_Path3(L"C:\\Users\\alex\\Desktop\\ML_Object_Detection\\fish.png");
    hstring IMG_Path4(L"C:\\Users\\alex\\Desktop\\ML_Object_Detection\\Koala.jpg");
    hstring IMG_Path5(L"C:\\Users\\alex\\Desktop\\ML_Object_Detection\\spoon.png");
    hstring IMG_Path6(L"C:\\Users\\alex\\Desktop\\ML_Object_Detection\\fork.jpg");
    hstring IMG_Path7(L"C:\\Users\\alex\\Desktop\\ML_Object_Detection\\glass.jpg");

    ProcessModel(IMG_Path1);
    ProcessModel(IMG_Path2);
    ProcessModel(IMG_Path3);
    ProcessModel(IMG_Path4);
    ProcessModel(IMG_Path5);
    ProcessModel(IMG_Path6);
    ProcessModel(IMG_Path7);

    return 0;
}   // main

void ProcessModel(const hstring &IPath) noexcept
{
    hstring MD_Path(L"C:\\Users\\alex\\Desktop\\ML_Object_Detection\\SqueezeNet.onnx");
    string LBL_Path("C:\\Users\\alex\\Desktop\\ML_Object_Detection\\Labels.txt");
    Model *MD;

    MD = new Model(MD_Path, IPath, LBL_Path);
    MD->LoadModel();
    MD->LoadImg();
    MD->BindModel();
    MD->EvaluateModel();

    delete MD;
}
