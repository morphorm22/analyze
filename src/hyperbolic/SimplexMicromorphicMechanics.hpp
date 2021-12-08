#pragma once

#include "Simplex.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for simplex-based micromorphic mechanics
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexMicromorphicMechanics : public Plato::Simplex<SpaceDim>
{
  public:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;

    static constexpr Plato::OrdinalType mNumFullTerms    = (SpaceDim == 3) ? 9 :
                                             ((SpaceDim == 2) ? 4 :
                                            (((SpaceDim == 1) ? 1 : 0)));
    static constexpr Plato::OrdinalType mNumVoigtTerms   = (SpaceDim == 3) ? 6 :
                                             ((SpaceDim == 2) ? 3 :
                                            (((SpaceDim == 1) ? 1 : 0)));
    static constexpr Plato::OrdinalType mNumSkwTerms     = (SpaceDim == 3) ? 3 :
                                             ((SpaceDim == 2) ? 1 :
                                            (((SpaceDim == 1) ? 1 : 0)));

    static constexpr Plato::OrdinalType mNumDofsPerNode  = SpaceDim + mNumFullTerms;
    static constexpr Plato::OrdinalType mNumDofsPerCell  = mNumDofsPerNode*mNumNodesPerCell;


    static constexpr Plato::OrdinalType mNumControl = NumControls;

    static constexpr Plato::OrdinalType mNumNodeStatePerNode = 0;
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;

};
} // namespace Plato

