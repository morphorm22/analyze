/*
 * CriterionKirchhoffEnergyPotential_decl.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "MaterialModel.hpp"
#include "base/CriterionBase.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

/// @class CriterionKirchhoffEnergyPotential
/// @brief Evaluate elastic strain energy potential for a Kirchhoff material
///  \f[
///      \Phi=\frac{1}{2}C_{ijkl}E_{ij}E_{kl}
///  \f]
/// where \f$\mathbf{C}\f$ is the fourth order tensor of elastic moduli and \f$\mathbf{E}\f$
/// is the Green-Lagrange strain tensor.
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class CriterionKirchhoffEnergyPotential : public Plato::CriterionBase
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of integration points
  static constexpr auto mNumGaussPoints  = ElementType::mNumGaussPoints;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief local typename for base class
  using FunctionBaseType = typename Plato::CriterionBase;
  /// @brief contains mesh and model information
  using FunctionBaseType::mSpatialDomain;
  /// @brief output database
  using FunctionBaseType::mDataMap;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterial;

public:
  /// @brief class constructor
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output database
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aFuncName      name of criterion parameter list
  CriterionKirchhoffEnergyPotential(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aParamList,
    const std::string            & aFuncName
  );

  /// @brief class destructor
  ~CriterionKirchhoffEnergyPotential(){}

  /// @fn isLinear
  /// @brief returns true if criterion is linear
  /// @return boolean
  bool 
  isLinear() 
  const;

  /// @fn evaluateConditional
  /// @brief evaluate internal energt potential criterion
  /// @param [in,out] aWorkSets function domain and range workset database
  /// @param [in]     aCycle    scalar 
  void 
  evaluateConditional(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  ) const;
};

} // namespace Elliptic

} // namespace Plato
