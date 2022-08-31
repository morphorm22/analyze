#pragma once

#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

template<Plato::OrdinalType SpatialDim>
class MicromorphicLinearElasticMaterial
{
protected:
    static constexpr auto mNumVoigtTerms   = (SpatialDim == 3) ? 6 :
                                             ((SpatialDim == 2) ? 3 :
                                            (((SpatialDim == 1) ? 1 : 0)));
    static constexpr auto mNumSkwTerms     = (SpatialDim == 3) ? 3 :
                                             ((SpatialDim == 2) ? 1 :
                                            (((SpatialDim == 1) ? 1 : 0)));


    static_assert(mNumVoigtTerms, "SpatialDim must be 1, 2, or 3."); 

    Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffnessCe;   
    Plato::Matrix<mNumSkwTerms,mNumSkwTerms> mCellStiffnessCc;   
    Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffnessCm;   
    Plato::Array<mNumVoigtTerms> mReferenceStrain;                

    Plato::Scalar mRayleighB; 

public:
    MicromorphicLinearElasticMaterial();

    MicromorphicLinearElasticMaterial(const Teuchos::ParameterList& aParamList);

    decltype(mCellStiffnessCe)   getStiffnessMatrixCe() const {return mCellStiffnessCe;}

    decltype(mCellStiffnessCc)   getStiffnessMatrixCc() const {return mCellStiffnessCc;}

    decltype(mCellStiffnessCm)   getStiffnessMatrixCm() const {return mCellStiffnessCm;}

    decltype(mReferenceStrain) getReferenceStrain() const {return mReferenceStrain;}

    decltype(mRayleighB)       getRayleighB()       const {return mRayleighB;}

private:
    void initialize();

    void setReferenceStrainTensor(const Teuchos::ParameterList& aParamList);
};

}
