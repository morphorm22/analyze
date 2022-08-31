#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Pressure gradient functor.

 Given a gradient matrix, b, and state array, s, compute the pressure gradient, g.

 g_{i} = s_{e,I} b_{e,I,i}

 e:  element index
 I:  local node index
 i:  dimension index

 */
/******************************************************************************/
template<typename ElementType>
class PressureGradient : public ElementType
{
private:
    using ElementType::mNumSpatialDims;
    using ElementType::mNumNodesPerCell;

    Plato::Scalar mPressureScaling; /*!< Pressure scaling term used to improve condition number, which may be high due to poor scaling */

public:
    /******************************************************************************//**
     * \brief Default constructor
     * \param [in] aInput pressure scaling
    **********************************************************************************/
    PressureGradient(Plato::Scalar aInput = 1.0) :
            mPressureScaling(aInput)
    {
    }

    /******************************************************************************//**
     * \brief Compute pressure gradient at cubature point, i.e. integration point
     * \param [in] aCellOrdinal    cell (i.e. element) ordinal
     * \param [out] aPressureGrad  pressure gradient workset
     * \param [in] aPressure       pressure state workset
     * \param [in] aGradient       configuration gradient workset
    **********************************************************************************/
    template<typename ResultScalarType, typename StateScalarType, typename GradientScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        Plato::OrdinalType                                      aCellOrdinal,
        Plato::Array<mNumSpatialDims, ResultScalarType>       & aPressureGrad,
        Plato::ScalarMultiVectorT<StateScalarType>      const & aPressure,
        Plato::Matrix<mNumNodesPerCell,
                     mNumSpatialDims,
                     GradientScalarType>                const & aGradient
    ) const
    {
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mNumSpatialDims; tDimIndex++)
        {
            aPressureGrad(tDimIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                aPressureGrad(tDimIndex) += mPressureScaling * aPressure(aCellOrdinal, tNodeIndex)
                                                             * aGradient(tNodeIndex, tDimIndex);
            }
        }
    }
};
// class PressureGradient

} // namespace Plato
