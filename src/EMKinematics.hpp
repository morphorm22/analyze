#ifndef EMKINEMATICS_HPP
#define EMKINEMATICS_HPP

#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Electromechanical kinematics functor.
  
    Given a gradient matrix and displacement array, compute the strain 
    and electric field.
*/
/******************************************************************************/
template<typename ElementType>
class EMKinematics : ElementType
{
  private:

    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;

  public:

    template<typename StrainScalarType, typename StateScalarType, typename GradientScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        Plato::OrdinalType                                                           aCellOrdinal,
        Plato::Array<mNumVoigtTerms,  StrainScalarType>                            & aStrain,
        Plato::Array<mNumSpatialDims, StrainScalarType>                            & aEField,
        Plato::ScalarMultiVectorT<StateScalarType>                           const & aState,
        Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType> const & aGradient) const
    {

      // compute strain
      //
      Plato::OrdinalType voigtTerm=0;
      for(Plato::OrdinalType iDof=0; iDof<mNumSpatialDims; iDof++){
        aStrain(voigtTerm)=0.0;
        for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*mNumDofsPerNode+iDof;
          aStrain(voigtTerm) += aState(aCellOrdinal,localOrdinal)*aGradient(iNode,iDof);
        }
        voigtTerm++;
      }
      for (Plato::OrdinalType jDof=mNumSpatialDims-1; jDof>=1; jDof--){
        for (Plato::OrdinalType iDof=jDof-1; iDof>=0; iDof--){
          for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
            Plato::OrdinalType iLocalOrdinal = iNode*mNumDofsPerNode+iDof;
            Plato::OrdinalType jLocalOrdinal = iNode*mNumDofsPerNode+jDof;
            aStrain(voigtTerm) +=(aState(aCellOrdinal,jLocalOrdinal)*aGradient(iNode, iDof)
                                 +aState(aCellOrdinal,iLocalOrdinal)*aGradient(iNode, jDof));
          }
          voigtTerm++;
        }
      }
 
      // compute efield
      //
      Plato::OrdinalType dofOffset = mNumSpatialDims;
      for(Plato::OrdinalType iDof=0; iDof<mNumSpatialDims; iDof++){
        aEField(iDof) = 0.0;
        for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*mNumDofsPerNode+dofOffset;
          aEField(iDof) -= aState(aCellOrdinal,localOrdinal)*aGradient(iNode, iDof);
        }
      }
    }
};

} // namespace Plato

#endif
