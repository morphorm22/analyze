#pragma once

#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************/
/*! Two-field thermomechanical kinematics functor.

 Given a gradient matrix and state array, compute the pressure gradient,
 temperature gradient, and symmetric gradient of the displacement.
 */
/******************************************************************************/
template<typename ElementType>
class TMKinematics : ElementType
{
private:

    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mPressureDofOffset;
    using ElementType::mTDofOffset;

public:

    template<typename StrainScalarType, typename StateScalarType, typename GradientScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        Plato::OrdinalType                                                           aCellOrdinal,
        Plato::Array<mNumVoigtTerms,  StrainScalarType>                            & aStrain,
        Plato::Array<mNumSpatialDims, StrainScalarType>                            & aPressureGrad,
        Plato::Array<mNumSpatialDims, StrainScalarType>                            & aTempGrad,
        Plato::ScalarMultiVectorT<StateScalarType>                           const & aState,
        Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType> const & aGradient) const
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
                aStrain(tVoigtTerm) += aState(aCellOrdinal, tLocalOrdinal) * aGradient(tNodeIndex, tDofIndex);
            }
            tVoigtTerm++;
        }
        for(Plato::OrdinalType tDofIndexJ = mNumSpatialDims - 1; tDofIndexJ >= 1; tDofIndexJ--)
        {
            for(Plato::OrdinalType tDofIndexI = tDofIndexJ - 1; tDofIndexI >= 0; tDofIndexI--)
            {
                for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
                {
                    Plato::OrdinalType tLocalOrdinalI = tNodeIndex * mNumDofsPerNode + tDofIndexI;
                    Plato::OrdinalType tLocalOrdinalJ = tNodeIndex * mNumDofsPerNode + tDofIndexJ;
                    aStrain(tVoigtTerm) += (aState(aCellOrdinal, tLocalOrdinalJ)
                            * aGradient(tNodeIndex, tDofIndexI)
                            + aState(aCellOrdinal, tLocalOrdinalI) * aGradient(tNodeIndex, tDofIndexJ));
                }
                tVoigtTerm++;
            }
        }

        // compute pgrad
        //
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumSpatialDims; tDofIndex++)
        {
            aPressureGrad(tDofIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + mPressureDofOffset;
                aPressureGrad(tDofIndex) += aState(aCellOrdinal, tLocalOrdinal) * aGradient(tNodeIndex, tDofIndex);
            }
        }

        // compute tgrad
        //
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumSpatialDims; tDofIndex++)
        {
            aTempGrad(tDofIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + mTDofOffset;
                aTempGrad(tDofIndex) += aState(aCellOrdinal, tLocalOrdinal) * aGradient(tNodeIndex, tDofIndex);
            }
        }
    }
};

} // namespace Stabilized
} // namespace Plato
