#pragma once

#include <Teuchos_RCPDecl.hpp>

#include "PlatoMathTypes.hpp"
#include "LinearElasticMaterial.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Hatching
{

/******************************************************************************/
/*! Stress functor.

 given a strain, compute the stress.
 stress tensor in Voigt notation = {s_xx, s_yy, s_zz, s_yz, s_xz, s_xy}
 */
/******************************************************************************/
template<typename ElementType>
class LinearStress : ElementType
{
private:

    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;

    const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness;

    Plato::Array<mNumVoigtTerms> mReferenceStrain;


public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aCellStiffness material element stiffness matrix
    **********************************************************************************/
    LinearStress(const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> aCellStiffness) :
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
    LinearStress(const Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> aMaterialModel) :
      mCellStiffness  (aMaterialModel->getStiffnessMatrix()),
      mReferenceStrain(aMaterialModel->getReferenceStrain()) {}

#ifdef WHAT_USES_THIS_
    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor
     * \param [in]  aCellOrdinal element ordinal
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
    **********************************************************************************/
    KOKKOS_INLINE_FUNCTION void
    operator()(Plato::OrdinalType aCellOrdinal,
               Plato::ScalarMultiVectorT<ResultT> const& aCauchyStress,
               Plato::ScalarMultiVectorT<StrainT> const& aSmallStrain) const
    {
        // Method used to compute the stress and called from within a
        // Kokkos parallel_for.
        for(Plato::OrdinalType tVoigtIndex_I = 0; tVoigtIndex_I < mNumVoigtTerms; tVoigtIndex_I++)
        {
            aCauchyStress(aCellOrdinal, tVoigtIndex_I) = 0.0;

            for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
            {
                aCauchyStress(aCellOrdinal, tVoigtIndex_I) +=
                  (aSmallStrain(aCellOrdinal, tVoigtIndex_J) - mReferenceStrain(tVoigtIndex_J)) *
                  mCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
            }
        }
    }
#endif

    /******************************************************************************//**
     * \brief Compute Cauchy stress tensor
     * \param [in]  aCellOrdinal element ordinal
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aStrainInc Incremental strain tensor
     * \param [in]  aPrevStrain Reference strain tensor
    **********************************************************************************/
    template<typename ResultT, typename StrainT, typename LocalStateT>
    KOKKOS_INLINE_FUNCTION void
    operator()(
      Plato::OrdinalType                            aCellOrdinal,
      Plato::OrdinalType                            aGpOrdinal,
      Plato::Array<mNumVoigtTerms, ResultT>       & aCauchyStress,
      Plato::Array<mNumVoigtTerms, StrainT> const & aStrainInc,
      Plato::ScalarArray3DT<LocalStateT>    const & aPrevStrain
    ) const
    {
        // compute stress
        //
        for(Plato::OrdinalType tVoigtIndex_I = 0; tVoigtIndex_I < mNumVoigtTerms; tVoigtIndex_I++)
        {
            aCauchyStress(tVoigtIndex_I) = 0.0;

            for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
            {
                aCauchyStress(tVoigtIndex_I) +=
                    (aStrainInc(tVoigtIndex_J) - mReferenceStrain(tVoigtIndex_J) +
                     aPrevStrain(aCellOrdinal, aGpOrdinal, tVoigtIndex_J)) * mCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
            }
        }
    }
};
// class LinearStress

}// namespace Hatching

}// namespace Elliptic

}// namespace Plato
