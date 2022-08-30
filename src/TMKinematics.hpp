#ifndef TMKINEMATICS_HPP
#define TMKINEMATICS_HPP

#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Thermomechanical kinematics functor.

 Given a gradient matrix and displacement array, compute the strain
 and temperature gradient.
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

public:

    template<typename StrainScalarType, typename StateScalarType, typename GradientScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        Plato::OrdinalType                                                           aCellOrdinal,
        Plato::Array<mNumVoigtTerms,  StrainScalarType>                            & aStrain,
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

        // compute tgrad
        //
        Plato::OrdinalType tDofOffset = mNumSpatialDims;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumSpatialDims; tDofIndex++)
        {
            aTempGrad(tDofIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDofOffset;
                aTempGrad(tDofIndex) += aState(aCellOrdinal, tLocalOrdinal) * aGradient(tNodeIndex, tDofIndex);
            }
        }
    }
}; // class TMKinematics
} // namespace Plato

#endif
