#include "hyperbolic/micromorphic/MicromorphicInertiaMaterial.hpp"

namespace Plato
{

//*********************************************************************************
//**************************** NEXT: 1D Implementation ****************************
//*********************************************************************************

template<>
void MicromorphicInertiaMaterial<1>::initialize()
{
    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellInertiaTe(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumSkwTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumSkwTerms; tIndexJ++)
        {
            mCellInertiaTc(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellInertiaJm(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumSkwTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumSkwTerms; tIndexJ++)
        {
            mCellInertiaJc(tIndexI, tIndexJ) = 0.0;
        }
    }
}

template<>
MicromorphicInertiaMaterial<1>::MicromorphicInertiaMaterial() :
        mCellMacroscopicDensity(1.0),
        mPressureScaling(1.0),
        mRayleighA(0.0)
{
    this->initialize();
}

template<>
MicromorphicInertiaMaterial<1>::MicromorphicInertiaMaterial(const Teuchos::ParameterList& aParamList) :
        mCellMacroscopicDensity(1.0),
        mPressureScaling(1.0),
        mRayleighA(0.0)
{
    this->initialize();

    if(aParamList.isType<Plato::Scalar>("RayleighA"))
    {
        mRayleighA = aParamList.get<Plato::Scalar>("RayleighA");
    }
}

//*********************************************************************************
//**************************** NEXT: 2D Implementation ****************************
//*********************************************************************************

template<>
void MicromorphicInertiaMaterial<2>::initialize()
{
    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellInertiaTe(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumSkwTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumSkwTerms; tIndexJ++)
        {
            mCellInertiaTc(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellInertiaJm(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumSkwTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumSkwTerms; tIndexJ++)
        {
            mCellInertiaJc(tIndexI, tIndexJ) = 0.0;
        }
    }
}

template<>
MicromorphicInertiaMaterial<2>::MicromorphicInertiaMaterial() :
        mCellMacroscopicDensity(1.0),
        mPressureScaling(1.0),
        mRayleighA(0.0)
{
    this->initialize();
}

template<>
MicromorphicInertiaMaterial<2>::MicromorphicInertiaMaterial(const Teuchos::ParameterList& aParamList) :
        mCellMacroscopicDensity(1.0),
        mPressureScaling(1.0),
        mRayleighA(0.0)
{
    this->initialize();

    if(aParamList.isType<Plato::Scalar>("RayleighA"))
    {
        mRayleighA = aParamList.get<Plato::Scalar>("RayleighA");
    }
}

//*********************************************************************************
//**************************** NEXT: 3D Implementation ****************************
//*********************************************************************************

template<>
void MicromorphicInertiaMaterial<3>::initialize()
{
    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellInertiaTe(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumSkwTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumSkwTerms; tIndexJ++)
        {
            mCellInertiaTc(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellInertiaJm(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumSkwTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumSkwTerms; tIndexJ++)
        {
            mCellInertiaJc(tIndexI, tIndexJ) = 0.0;
        }
    }
}

template<>
MicromorphicInertiaMaterial<3>::MicromorphicInertiaMaterial() :
        mCellMacroscopicDensity(1.0),
        mPressureScaling(1.0),
        mRayleighA(0.0)
{
    this->initialize();
}

template<>
MicromorphicInertiaMaterial<3>::MicromorphicInertiaMaterial(const Teuchos::ParameterList& aParamList) :
        mCellMacroscopicDensity(1.0),
        mPressureScaling(1.0),
        mRayleighA(0.0)
{
    this->initialize();

    if(aParamList.isType<Plato::Scalar>("RayleighA"))
    {
        mRayleighA = aParamList.get<Plato::Scalar>("RayleighA");
    }
}

} 
