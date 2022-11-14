#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************/
/*! Two-field Mechanical kinematics functor.

 Given a gradient matrix and state array, compute the pressure gradient
 and symmetric gradient of the displacement.
 */
/******************************************************************************/
template<typename ElementType>
class Kinematics : ElementType
{
private:
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumSpatialDims;
    using ElementType::mPressureDofOffset;

public:
    /***********************************************************************************
     * \brief Compute deviatoric stress, volume flux, cell stabilization
     * \param [in] aCellOrdinal cell ordinal
     * \param [in/out] aStrain displacement strains workset on H^{1}(\Omega)
     * \param [in/out] aPressureGrad pressure gradient workset on L^2(\Omega)
     * \param [in] aState state workset
     * \param [in] aGradient configuration gradient workset
     **********************************************************************************/
    template<typename StrainScalarType, typename StateScalarType, typename GradientScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        Plato::OrdinalType                                      aCellOrdinal,
        Plato::Array<mNumVoigtTerms, StrainScalarType>        & aStrain,
        Plato::Array<mNumSpatialDims, StrainScalarType>       & aPressureGrad,
        Plato::ScalarMultiVectorT<StateScalarType>      const & aState,
        Plato::Matrix<mNumNodesPerCell,
                     mNumSpatialDims,
                     GradientScalarType>                const & aGradient) const
    {
        // compute strain
        //
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumSpatialDims; tDofIndex++)
        {
            aStrain(tVoigtTerm) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDofIndex;
                aStrain(tVoigtTerm) += aState(aCellOrdinal, tLocalOrdinal)
                                     * aGradient(tNodeIndex, tDofIndex);
            }
            tVoigtTerm++;
        }

        for(Plato::OrdinalType tDofJ = mNumSpatialDims - 1; tDofJ >= 1; tDofJ--)
        {
            for(Plato::OrdinalType tDofI = tDofJ - 1; tDofI >= 0; tDofI--)
            {
                for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
                {
                    Plato::OrdinalType tLocalOrdinalI = tNodeIndex * mNumDofsPerNode + tDofI;
                    Plato::OrdinalType tLocalOrdinalJ = tNodeIndex * mNumDofsPerNode + tDofJ;
                    aStrain(tVoigtTerm) +=
                            ( aState(aCellOrdinal, tLocalOrdinalJ) * aGradient(tNodeIndex, tDofI)
                            + aState(aCellOrdinal, tLocalOrdinalI) * aGradient(tNodeIndex, tDofJ) );
                }
                tVoigtTerm++;
            }
        }

        // compute pressure gradient
        //
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumSpatialDims; tDofIndex++)
        {
            aPressureGrad(tDofIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + mPressureDofOffset;
                aPressureGrad(tDofIndex) += aState(aCellOrdinal, tLocalOrdinal)
                                          * aGradient(tNodeIndex, tDofIndex);
            }
        }
    }
};
// class Kinematics

} // namespace Stabilized
} // namespace Plato
