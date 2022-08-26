#ifndef PLATO_HOMOGENIZED_STRESS_HPP
#define PLATO_HOMOGENIZED_STRESS_HPP

#include "PlatoMathTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Homogenized stress functor.
  
    given a characteristic strain, compute the homogenized stress.
*/
/******************************************************************************/
template<typename ElementType>
class HomogenizedStress : public ElementType
{
  private:

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;

    const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness;
    const int mColumnIndex;

  public:

    HomogenizedStress( const Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> aCellStiffness, int aColumnIndex) :
            mCellStiffness(aCellStiffness), 
            mColumnIndex(aColumnIndex) {}

    template<typename StressScalarType, typename StrainScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()( int cellOrdinal,
                Plato::Array<mNumVoigtTerms, StressScalarType> & tStress,
                Plato::Array<mNumVoigtTerms, StrainScalarType> & tStrain) const {

      // compute stress
      //
      for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
        tStress(iVoigt) = mCellStiffness(mColumnIndex, iVoigt);
        for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          tStress(iVoigt) -= tStrain(jVoigt)*mCellStiffness(jVoigt, iVoigt);
        }
      }
    }
};

}

#endif
