/*
 * StateGradient.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

namespace Plato
{

/// @class StateGradient
/// @brief Computes state gradient:
/// \f[ 
///     \nabla\mathbf{U}=\frac{\partial\mathbf{U}}{\partial\mathbf{X}}
/// \f]
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class StateGradient
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types associated with input evaluation template type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  /// @brief number of spatial dimensions 
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;

public:
  /// @fn operator()()
  /// @brief compute trial state gradient
  /// @param [in]     aCellIndex local element ordinal
  /// @param [in]     aStates    2D state workset
  /// @param [in]     aGradient  gradient functions
  /// @param [in,out] aStateGrad state gradient
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::OrdinalType                                               & aCellIndex,
    const Plato::ScalarMultiVectorT<StateScalarType>                       & aStates,
    const Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> & aGradient,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStateGrad
  ) const
  {
    for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
    {
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
      {
        Plato::OrdinalType tDof = (tNode * mNumDofsPerNode) + tDimI;
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
        {
          aStateGrad(tDimI,tDimJ) += aStates(aCellIndex,tDof) * aGradient(tNode,tDimJ);
        }
      }
    }
  }

  /// @fn operator()()
  /// @brief compute test state gradient
  /// @param [in]     aCellIndex local element ordinal
  /// @param [in]     aGradient  gradient functions
  /// @param [in,out] aStateGrad state gradient
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::OrdinalType                                               & aCellIndex,
    const Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> & aGradient,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStateGrad
  ) const
  {
    constexpr Plato::Scalar tOne(1.0);
    for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
    {
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
      {
        Plato::OrdinalType tDof = (tNode * mNumDofsPerNode) + tDimI;
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
        {
          aStateGrad(tDimI,tDimJ) += tOne * aGradient(tNode,tDimJ);
        }
      }
    }
  }

};

}