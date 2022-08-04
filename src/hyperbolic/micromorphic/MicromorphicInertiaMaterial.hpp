#pragma once

#include "PlatoMathTypes.hpp"
#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

template<Plato::OrdinalType SpatialDim>
class MicromorphicInertiaMaterial
{
protected:
    static constexpr auto mNumVoigtTerms   = (SpatialDim == 3) ? 6 :
                                             ((SpatialDim == 2) ? 3 :
                                            (((SpatialDim == 1) ? 1 : 0)));
    static constexpr auto mNumSkwTerms     = (SpatialDim == 3) ? 3 :
                                             ((SpatialDim == 2) ? 1 :
                                            (((SpatialDim == 1) ? 1 : 0)));


    static_assert(mNumVoigtTerms, "SpatialDim must be 1, 2, or 3."); 

    Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellInertiaTe;  
    Plato::Matrix<mNumSkwTerms,mNumSkwTerms> mCellInertiaTc;  
    Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellInertiaJm;  
    Plato::Matrix<mNumSkwTerms,mNumSkwTerms> mCellInertiaJc;   

    Plato::Scalar mCellMacroscopicDensity; 
    Plato::Scalar mPressureScaling; 
    Plato::Scalar mRayleighA; 

public:
    MicromorphicInertiaMaterial();

    MicromorphicInertiaMaterial(const Teuchos::ParameterList& aParamList);

    decltype(mCellMacroscopicDensity)     getMacroscopicMassDensity()     const {return mCellMacroscopicDensity;}

    decltype(mCellInertiaTe)   getInertiaMatrixTe() const {return mCellInertiaTe;}

    decltype(mCellInertiaTc)   getInertiaMatrixTc() const {return mCellInertiaTc;}

    decltype(mCellInertiaJm)   getInertiaMatrixJm() const {return mCellInertiaJm;}

    decltype(mCellInertiaJc)   getInertiaMatrixJc() const {return mCellInertiaJc;}

    decltype(mPressureScaling) getPressureScaling() const {return mPressureScaling;}

    decltype(mRayleighA)       getRayleighA()       const {return mRayleighA;}

private:
    void initialize();

};

}

