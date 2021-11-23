#include "hyperbolic/MicromorphicLinearElasticMaterial.hpp"

namespace Plato
{

//*********************************************************************************
//**************************** NEXT: 1D Implementation ****************************
//*********************************************************************************

template<>
void MicromorphicLinearElasticMaterial<1>::initialize()
{
    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellStiffnessCe(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumSkwTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumSkwTerms; tIndexJ++)
        {
            mCellStiffnessCc(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellStiffnessCm(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        mReferenceStrain(tIndexI) = 0.0;
    }
}

template<>
void MicromorphicLinearElasticMaterial<1>::setReferenceStrainTensor(const Teuchos::ParameterList& aParamList)
{
    if(aParamList.isType<Plato::Scalar>("e11"))
        mReferenceStrain(0) = aParamList.get<Plato::Scalar>("e11");
}

template<>
MicromorphicLinearElasticMaterial<1>::MicromorphicLinearElasticMaterial() :
        mRayleighB(0.0)
{
    this->initialize();
}

template<>
MicromorphicLinearElasticMaterial<1>::MicromorphicLinearElasticMaterial(const Teuchos::ParameterList& aParamList) :
        mRayleighB(0.0)
{
    this->initialize();
    this->setReferenceStrainTensor(aParamList);

    if(aParamList.isType<Plato::Scalar>("RayleighB"))
    {
        mRayleighB = aParamList.get<Plato::Scalar>("RayleighB");
    }
}

//*********************************************************************************
//**************************** NEXT: 2D Implementation ****************************
//*********************************************************************************

template<>
void MicromorphicLinearElasticMaterial<2>::initialize()
{
    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellStiffnessCe(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumSkwTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumSkwTerms; tIndexJ++)
        {
            mCellStiffnessCc(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellStiffnessCm(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        mReferenceStrain(tIndexI) = 0.0;
    }
}

template<>
void MicromorphicLinearElasticMaterial<2>::setReferenceStrainTensor(const Teuchos::ParameterList& aParamList)
{
    if(aParamList.isType<Plato::Scalar>("e11"))
        mReferenceStrain(0) = aParamList.get<Plato::Scalar>("e11");
    if(aParamList.isType<Plato::Scalar>("e22"))
        mReferenceStrain(1) = aParamList.get<Plato::Scalar>("e22");
    if(aParamList.isType<Plato::Scalar>("e12"))
        mReferenceStrain(2) = aParamList.get<Plato::Scalar>("e12");
}

template<>
MicromorphicLinearElasticMaterial<2>::MicromorphicLinearElasticMaterial() :
        mRayleighB(0.0)
{
    this->initialize();
}

template<>
MicromorphicLinearElasticMaterial<2>::MicromorphicLinearElasticMaterial(const Teuchos::ParameterList& aParamList) :
        mRayleighB(0.0)
{
    this->initialize();
    this->setReferenceStrainTensor(aParamList);

    if(aParamList.isType<Plato::Scalar>("RayleighB"))
    {
        mRayleighB = aParamList.get<Plato::Scalar>("RayleighB");
    }
}


//*********************************************************************************
//**************************** NEXT: 3D Implementation ****************************
//*********************************************************************************

template<>
void MicromorphicLinearElasticMaterial<3>::initialize()
{
    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellStiffnessCe(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumSkwTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumSkwTerms; tIndexJ++)
        {
            mCellStiffnessCc(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellStiffnessCm(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        mReferenceStrain(tIndexI) = 0.0;
    }
}

template<>
void MicromorphicLinearElasticMaterial<3>::setReferenceStrainTensor(const Teuchos::ParameterList& aParamList)
{
    if(aParamList.isType<Plato::Scalar>("e11"))
        mReferenceStrain(0) = aParamList.get<Plato::Scalar>("e11");
    if(aParamList.isType<Plato::Scalar>("e22"))
        mReferenceStrain(1) = aParamList.get<Plato::Scalar>("e22");
    if(aParamList.isType<Plato::Scalar>("e33"))
        mReferenceStrain(2) = aParamList.get<Plato::Scalar>("e33");
    if(aParamList.isType<Plato::Scalar>("e23"))
        mReferenceStrain(3) = aParamList.get<Plato::Scalar>("e23");
    if(aParamList.isType<Plato::Scalar>("e13"))
        mReferenceStrain(4) = aParamList.get<Plato::Scalar>("e13");
    if(aParamList.isType<Plato::Scalar>("e12"))
        mReferenceStrain(5) = aParamList.get<Plato::Scalar>("e12");
}

template<>
MicromorphicLinearElasticMaterial<3>::MicromorphicLinearElasticMaterial() :
        mRayleighB(0.0)
{
    this->initialize();
}

template<>
MicromorphicLinearElasticMaterial<3>::MicromorphicLinearElasticMaterial(const Teuchos::ParameterList& aParamList) :
        mRayleighB(0.0)
{
    this->initialize();
    this->setReferenceStrainTensor(aParamList);

    if(aParamList.isType<Plato::Scalar>("RayleighB"))
    {
        mRayleighB = aParamList.get<Plato::Scalar>("RayleighB");
    }
}

} // namespace Plato
