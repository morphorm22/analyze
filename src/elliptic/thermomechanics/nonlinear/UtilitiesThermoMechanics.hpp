/*
 * UtilitiesThermoMechanics.hpp
 *
 *  Created on: June 17, 2023
 */

#pragma once

#include "PlatoMathTypes.hpp"

namespace Plato
{

namespace Elliptic
{

/// @brief get cell second piola-kirchhoff stress tensor at integration point
/// @tparam ScalarType     data scalar type
/// @tparam NumSpatialDims number of spatial dimensions
/// @param aCellOrdinal   cell/element ordinal
/// @param aIntgPtOrdinal integration point ordinal
/// @param aIn2PKS        second piola-kirchhoff stress workset
/// @param aOut2PKS       cell second piola-kirchhoff stress
template<Plato::OrdinalType NumSpatialDims, typename ScalarType>
KOKKOS_INLINE_FUNCTION
void 
getCell2PKS(
  const Plato::OrdinalType                                      & aCellOrdinal,
  const Plato::OrdinalType                                      & aIntgPtOrdinal,
  const Plato::ScalarArray4DT<ScalarType>                       & aIn2PKS,
        Plato::Matrix<NumSpatialDims,NumSpatialDims,ScalarType> & aOut2PKS
)
{
  for(Plato::OrdinalType tDimI = 0; tDimI < NumSpatialDims; tDimI++){
    for(Plato::OrdinalType tDimJ = 0; tDimJ < NumSpatialDims; tDimJ++){
      aOut2PKS(tDimI,tDimJ) = aIn2PKS(aCellOrdinal,aIntgPtOrdinal,tDimI,tDimJ);
    }
  }
}

} // namespace Elliptic

} // namespace Plato