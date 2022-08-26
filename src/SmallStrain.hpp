#pragma once

#include "PlatoStaticsTypes.hpp"

#include "PlatoMathTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Infinitesimal strain functor.

 Given a gradient matrix and displacement array, compute the strain.
 strain tensor in Voigt notation = {e_xx, e_yy, e_zz, e_yz, e_xz, e_xy}

 */
/******************************************************************************/
template<typename ElementType>
class SmallStrain : public ElementType
{
private:

    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumNodesPerCell;

public:

    template<typename StrainScalarType, typename DispScalarType, typename GradientScalarType>
    KOKKOS_INLINE_FUNCTION void operator()(
              Plato::OrdinalType                                                     aCellOrdinal,
              Plato::Array<mNumVoigtTerms, StrainScalarType>                       & aStrain,
        const Plato::ScalarMultiVectorT<DispScalarType>                            & aState,
        const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType> & aGradient) const
    {
        /***************************************************************************//**
         * \brief Compute Cauchy strain tensor - Voigt notation used herein.
         * \param [in/out] aStrain      Cauchy strain tensor
         * \param [in]     aState       state workset
         * \param [in]     aGradient    spatial gradient matrix
        *******************************************************************************/
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mNumSpatialDims; tDimIndex++)
        {
            aStrain(tVoigtTerm) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                auto tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDimIndex;
                aStrain(tVoigtTerm) +=
                        aState(aCellOrdinal, tLocalOrdinal) * aGradient(tNodeIndex, tDimIndex);
            }
            tVoigtTerm++;
        }

        for(Plato::OrdinalType tDofIndexJ = mNumSpatialDims - 1; tDofIndexJ >= 1; tDofIndexJ--)
        {
            for(Plato::OrdinalType tDofIndexI = tDofIndexJ - 1; tDofIndexI >= 0; tDofIndexI--)
            {
                for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
                {
                    auto tLocalOrdinalI = tNodeIndex * mNumDofsPerNode + tDofIndexI;
                    auto tLocalOrdinalJ = tNodeIndex * mNumDofsPerNode + tDofIndexJ;
                    aStrain(tVoigtTerm) += (aState(aCellOrdinal, tLocalOrdinalJ) * aGradient(tNodeIndex, tDofIndexI)
                            + aState(aCellOrdinal, tLocalOrdinalI) * aGradient(tNodeIndex, tDofIndexJ));
                }
                tVoigtTerm++;
            }
        }
    }
};
// class Strain

}
// namespace Plato
