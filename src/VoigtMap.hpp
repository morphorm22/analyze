/*
 * VoigtMap.hpp
 *
 *  Created on: Jun 11, 2020
 */

#ifndef VOIGTMAP_HPP
#define VOIGTMAP_HPP

#include "PlatoTypes.hpp"

namespace Plato {
  template<int SpaceDim> struct VoigtMap {};
  template<>
  struct VoigtMap<1> {
    static constexpr int cNumVoigtTerms = 1;
    Plato::OrdinalType I[1];
    Plato::OrdinalType J[1];
    Plato::OrdinalType V[1][1];
    KOKKOS_INLINE_FUNCTION VoigtMap() : I{0}, J{0}, V{{0}} {}
  };
  template<>
  struct VoigtMap<2> {
    static constexpr int cNumVoigtTerms = 3;
    Plato::OrdinalType I[3];
    Plato::OrdinalType J[3];
    Plato::OrdinalType V[2][2];
    KOKKOS_INLINE_FUNCTION VoigtMap() :
      I{0,1,0},
      J{0,1,1},
      V{{0,2},{2,1}} {}
  };
  template<>
  struct VoigtMap<3> {
    static constexpr int cNumSpaceDims = 3;
    static constexpr int cNumVoigtTerms = 6;
    Plato::OrdinalType I[6];
    Plato::OrdinalType J[6];
    Plato::OrdinalType V[3][3];
    KOKKOS_INLINE_FUNCTION VoigtMap() :
      I{0,1,2,1,0,0},
      J{0,1,2,2,2,1},
      V{{0,5,4},{5,1,3},{4,3,2}} {}
  };
}

#endif
