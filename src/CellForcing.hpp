#ifndef PLATO_CELL_FORCING_HPP
#define PLATO_CELL_FORCING_HPP

#include "PlatoMathTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Add forcing for homogenization cell problem.
  
    given a view, subtract the forcing column.
*/
/******************************************************************************/
template<typename ElementType>
class CellForcing : public ElementType
{
  private:

    using ElementType::mNumVoigtTerms;

    Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness;
    int mColumnIndex;

  public:
    CellForcing() : mCellStiffness(0), mColumnIndex(-1) {}
    CellForcing( const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> aCellStiffness, int aColumnIndex) :
            mCellStiffness(aCellStiffness), 
            mColumnIndex(aColumnIndex) {}

    void setCellStiffness(const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> & aCellStiffness) {mCellStiffness = aCellStiffness;}
    void setColumnIndex(int aColumnIndex) {mColumnIndex = aColumnIndex;}

    template<typename StressScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        Plato::Array<mNumVoigtTerms, StressScalarType> & aTensor
    ) const
    {
        // add forcing
        //
        for(Plato::OrdinalType tTermIndex = 0; tTermIndex < mNumVoigtTerms; tTermIndex++)
        {
            aTensor(tTermIndex) -= mCellStiffness(tTermIndex, mColumnIndex);
        }
    }
};

}

#endif
