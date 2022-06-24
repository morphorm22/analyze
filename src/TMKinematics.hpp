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
    DEVICE_TYPE inline void
    operator()(
        Plato::OrdinalType                                  aCellOrdinal,
        Plato::ScalarMultiVectorT<StrainScalarType> const & aStrain,
        Plato::ScalarMultiVectorT<StrainScalarType> const & aTempGrad,
        Plato::ScalarMultiVectorT<StateScalarType>  const & aState,
        Plato::ScalarArray3DT<GradientScalarType>   const & aGradient) const
    {

        // compute strain
        //
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumSpatialDims; tDofIndex++)
        {
            aStrain(aCellOrdinal, tVoigtTerm) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDofIndex;
                aStrain(aCellOrdinal, tVoigtTerm) += aState(aCellOrdinal, tLocalOrdinal) * aGradient(aCellOrdinal, tNodeIndex, tDofIndex);
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
                    aStrain(aCellOrdinal, tVoigtTerm) += (aState(aCellOrdinal, tLocalOrdinalJ)
                            * aGradient(aCellOrdinal, tNodeIndex, tDofIndexI)
                            + aState(aCellOrdinal, tLocalOrdinalI) * aGradient(aCellOrdinal, tNodeIndex, tDofIndexJ));
                }
                tVoigtTerm++;
            }
        }

        // compute tgrad
        //
        Plato::OrdinalType tDofOffset = mNumSpatialDims;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumSpatialDims; tDofIndex++)
        {
            aTempGrad(aCellOrdinal, tDofIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDofOffset;
                aTempGrad(aCellOrdinal, tDofIndex) += aState(aCellOrdinal, tLocalOrdinal) * aGradient(aCellOrdinal, tNodeIndex, tDofIndex);
            }
        }
    }

    template<typename StrainScalarType, typename StateScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void
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
};
// class TMKinematics

#ifdef NOPE
/******************************************************************************/
/*! Two-field thermomechanical kinematics functor.

 Given a gradient matrix and state array, compute the pressure gradient,
 temperature gradient, and symmetric gradient of the displacement.
 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class StabilizedTMKinematics : public Plato::SimplexStabilizedThermomechanics<SpaceDim>
{
private:

    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mNumDofsPerNode;
    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mPressureDofOffset;
    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mTDofOffset;

public:

    template<typename StrainScalarType, typename StateScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void operator()(Plato::OrdinalType aCellOrdinal,
                                       Plato::ScalarMultiVectorT<StrainScalarType> const& aStrain,
                                       Plato::ScalarMultiVectorT<StrainScalarType> const& aPressureGrad,
                                       Plato::ScalarMultiVectorT<StrainScalarType> const& aTempGrad,
                                       Plato::ScalarMultiVectorT<StateScalarType> const& aState,
                                       Plato::ScalarArray3DT<GradientScalarType> const& aGradient) const
    {

        // compute strain
        //
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < SpaceDim; tDofIndex++)
        {
            aStrain(aCellOrdinal, tVoigtTerm) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDofIndex;
                aStrain(aCellOrdinal, tVoigtTerm) += aState(aCellOrdinal, tLocalOrdinal) * aGradient(aCellOrdinal, tNodeIndex, tDofIndex);
            }
            tVoigtTerm++;
        }
        for(Plato::OrdinalType tDofIndexJ = SpaceDim - 1; tDofIndexJ >= 1; tDofIndexJ--)
        {
            for(Plato::OrdinalType tDofIndexI = tDofIndexJ - 1; tDofIndexI >= 0; tDofIndexI--)
            {
                for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
                {
                    Plato::OrdinalType tLocalOrdinalI = tNodeIndex * mNumDofsPerNode + tDofIndexI;
                    Plato::OrdinalType tLocalOrdinalJ = tNodeIndex * mNumDofsPerNode + tDofIndexJ;
                    aStrain(aCellOrdinal, tVoigtTerm) += (aState(aCellOrdinal, tLocalOrdinalJ)
                            * aGradient(aCellOrdinal, tNodeIndex, tDofIndexI)
                            + aState(aCellOrdinal, tLocalOrdinalI) * aGradient(aCellOrdinal, tNodeIndex, tDofIndexJ));
                }
                tVoigtTerm++;
            }
        }

        // compute pgrad
        //
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < SpaceDim; tDofIndex++)
        {
            aPressureGrad(aCellOrdinal, tDofIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + mPressureDofOffset;
                aPressureGrad(aCellOrdinal, tDofIndex) += aState(aCellOrdinal, tLocalOrdinal) * aGradient(aCellOrdinal, tNodeIndex, tDofIndex);
            }
        }

        // compute tgrad
        //
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < SpaceDim; tDofIndex++)
        {
            aTempGrad(aCellOrdinal, tDofIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + mTDofOffset;
                aTempGrad(aCellOrdinal, tDofIndex) += aState(aCellOrdinal, tLocalOrdinal) * aGradient(aCellOrdinal, tNodeIndex, tDofIndex);
            }
        }
    }
};
#endif

} // namespace Plato

#endif
