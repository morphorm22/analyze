#pragma once

#include "LinearElasticMaterial.hpp"

#include "PlatoMathTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Stress functor.

 given a strain, compute the stress.
 stress tensor in Voigt notation = {s_xx, s_yy, s_zz, s_yz, s_xz, s_xy}
 */
/******************************************************************************/
template< typename EvaluationType, typename ElementType >
class RayleighStress : public ElementType
{
private:

    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using ElementType::mNumVoigtTerms; /*!< number of stress/strain terms */

    const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness; /*!< material stiffness matrix */
    Plato::Array<mNumVoigtTerms> mReferenceStrain;                       /*!< reference voigt strain tensor */
    Plato::Scalar mRayleighB;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aCellStiffness material element stiffness matrix
    **********************************************************************************/
    RayleighStress(const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> aCellStiffness) :
            mCellStiffness(aCellStiffness)
    {
        for(Plato::OrdinalType tIndex = 0; tIndex < mNumVoigtTerms; tIndex++)
        {
            mReferenceStrain(tIndex) = 0.0;
        }
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model interface
    **********************************************************************************/
    RayleighStress(const Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> aMaterialModel) :
            mCellStiffness(aMaterialModel->getStiffnessMatrix()),
            mReferenceStrain(aMaterialModel->getReferenceStrain()),
            mRayleighB(aMaterialModel->getRayleighB())
    {
    }

    /******************************************************************************//**
     * \brief Compute Cauchy stress tensor
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
     * \param [in]  aVelGrad Velocity gradient tensor
    **********************************************************************************/
    template<typename StressScalarType, typename StrainScalarType, typename VelGradScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()( Plato::Array<mNumVoigtTerms, StressScalarType>        & aCauchyStress,
                const Plato::Array<mNumVoigtTerms, StrainScalarType>  & aSmallStrain,
                const Plato::Array<mNumVoigtTerms, VelGradScalarType> & aVelGrad
    ) const 
    {
      for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
        aCauchyStress(iVoigt) = 0.0;
        for( Plato::OrdinalType jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          aCauchyStress(iVoigt) += (aSmallStrain(jVoigt)-mReferenceStrain(jVoigt))*mCellStiffness(iVoigt, jVoigt)
                                     +  aVelGrad(jVoigt)*mCellStiffness(iVoigt, jVoigt)*mRayleighB;
        }
      }
    }
};
// class RayleighStress

}// namespace Plato
