#pragma once

#include <Sacado.hpp>

#include "PlatoTypes.hpp"

namespace Plato {

namespace Geometry {

  template<typename ElementType>
  struct FadTypes {

    using ControlFad   = Sacado::Fad::SFad<Plato::Scalar,
                                           ElementType::mNumNodesPerCell>;
    using ConfigFad    = Sacado::Fad::SFad<Plato::Scalar,
                                           ElementType::mNumSpatialDims*
                                           ElementType::mNumNodesPerCell>;
  };
} // namespace Geometry

} // namespace Plato
