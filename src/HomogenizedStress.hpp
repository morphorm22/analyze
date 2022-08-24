#ifndef PLATO_HOMOGENIZED_STRESS_HPP
#define PLATO_HOMOGENIZED_STRESS_HPP

#include "PlatoMathTypes.hpp"
#include "SimplexMechanics.hpp"

namespace Plato
{

/******************************************************************************/
/*! Homogenized stress functor.
  
    given a characteristic strain, compute the homogenized stress.
*/
/******************************************************************************/
template<int SpaceDim>
class HomogenizedStress : public Plato::SimplexMechanics<SpaceDim>
{
  private:

    using Plato::SimplexMechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexMechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexMechanics<SpaceDim>::mNumDofsPerNode;
    using Plato::SimplexMechanics<SpaceDim>::mNumDofsPerCell;

    const Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;
    const int mColumnIndex;

  public:

    HomogenizedStress( const Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> aCellStiffness, int aColumnIndex) :
            mCellStiffness(aCellStiffness), 
            mColumnIndex(aColumnIndex) {}

    template<typename StressScalarType, typename StrainScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()( int cellOrdinal,
                Kokkos::View<StressScalarType**, Plato::Layout, Plato::MemSpace> const& stress,
                Kokkos::View<StrainScalarType**, Plato::Layout, Plato::MemSpace> const& strain) const {

      // compute stress
      //
      for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
        stress(cellOrdinal,iVoigt) = mCellStiffness(mColumnIndex, iVoigt);
        for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          stress(cellOrdinal,iVoigt) -= strain(cellOrdinal,jVoigt)*mCellStiffness(jVoigt, iVoigt);
        }
      }
    }
};

}

#endif
