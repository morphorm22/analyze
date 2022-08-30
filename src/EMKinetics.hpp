#ifndef PLATO_EMKINETICS_HPP
#define PLATO_EMKINETICS_HPP

#include "LinearElectroelasticMaterial.hpp"

/******************************************************************************/
/*! Electroelastics functor.
  
    given a strain and electric field, compute the stress and electric displacement

    IMPORTANT NOTE:  This model is scaled to make the coupling better conditioned. 
    The second equation is multiplied by a:

     i.e., this:     | T |   |   C     -e  |  |  S  |
                     |   | = |             |  |     |
                     | D |   |   e      p  |  |  E  |
              
     becomes this:   | T |   |   C   -a*e  |  |  S  |
                     |   | = |             |  |     |
                     |a*D|   |  a*e  a*a*p |  | E/a |
              
     A typical value for a is 1e9.  So, this model computes (T  a*D) from 
     (S E/a) which means that electrical quantities in the simulation are scaled:
 
            Electric potential:      phi/a
            Electric field:          E/a
            Electric displacement:   D*q
            Electric charge density: a*q

     and should be 'unscaled' before writing output.  Further, boundary conditions
     must be scaled.

    IMPORTANT NOTE 2:  This model is not positive definite!

*/
/******************************************************************************/

namespace Plato
{

template<typename ElementType>
class EMKinetics : public ElementType
{
  private:

    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;

    const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness;
    const Plato::Matrix<mNumSpatialDims, mNumVoigtTerms> mCellPiezoelectricCoupling;
    const Plato::Matrix<mNumSpatialDims, mNumSpatialDims> mCellPermittivity;
 
    const Plato::Scalar mAlpha;
    const Plato::Scalar mAlpha2;

  public:

    EMKinetics( const Teuchos::RCP<Plato::LinearElectroelasticMaterial<mNumSpatialDims>> materialModel ) :
            mCellStiffness(materialModel->getStiffnessMatrix()),
            mCellPiezoelectricCoupling(materialModel->getPiezoMatrix()),
            mCellPermittivity(materialModel->getPermittivityMatrix()),
            mAlpha(materialModel->getAlpha()),
            mAlpha2(mAlpha*mAlpha) { }

    template<typename KineticsScalarType, typename KinematicsScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        Plato::Array<mNumVoigtTerms,  KineticsScalarType>         & aStress,
        Plato::Array<mNumSpatialDims, KineticsScalarType>         & aEDisp,
        Plato::Array<mNumVoigtTerms,  KinematicsScalarType> const & tStrain,
        Plato::Array<mNumSpatialDims, KinematicsScalarType> const & tEField
    ) const
    {

      // compute stress
      //
      for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
        aStress(iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          aStress(iVoigt) += tStrain(jVoigt)*mCellStiffness(iVoigt, jVoigt);
        }
        for( int jDim=0; jDim<mNumSpatialDims; jDim++){
          aStress(iVoigt) -= mAlpha*tEField(jDim)*mCellPiezoelectricCoupling(jDim, iVoigt);
        }
      }

      // compute edisp
      //
      for( int iDim=0; iDim<mNumSpatialDims; iDim++){
        aEDisp(iDim) = 0.0;
        for( int jDim=0; jDim<mNumSpatialDims; jDim++){
          aEDisp(iDim) += mAlpha2*tEField(jDim)*mCellPermittivity(iDim, jDim);
        }
        for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          aEDisp(iDim) += mAlpha*tStrain(jVoigt)*mCellPiezoelectricCoupling(iDim, jVoigt);
        }
      }
    }
};

} // namespace Plato

#endif
