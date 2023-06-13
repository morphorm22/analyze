/*
 * CriterionNeoHookeanEnergyPotential_decl.hpp
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

/// @class CriterionNeoHookeanEnergyPotential
/// @brief Evaluate stored energy potential for a Neo-Hookean material
///  \f[
///    \Psi(\mathbf{C})=\frac{1}{2}\lambda(\ln(J))^2 - \mu\ln(J) + \frac{1}{2}\mu(\mbox{trace}(\mathbf{C})-3),
///  \f]
/// where \f$\mathbf{C}\f$ is the right deformation tensor, \f$J=\det(\mathbf{F})\f$, \f$F\f$ 
/// is the deformation gradient, and \f$\lambda\f$ and \f$\mu\f$ are the Lame constants. 
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class CriterionNeoHookeanEnergyPotential : public Plato::CriterionBase
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
  CriterionNeoHookeanEnergyPotential(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aParamList,
    const std::string            & aFuncName
  );

  /// @brief class destructor
  ~CriterionNeoHookeanEnergyPotential(){}

  /// @fn isLinear
  /// @brief returns true if criterion is linear
  /// @return boolean
  virtual 
  bool 
  isLinear() 
  const;

  /// @fn evaluate_conditional
  /// @brief evaluate criterion on each cell
  /// @param [in]     aState   2D state workset 
  /// @param [in]     aControl 2D control workset
  /// @param [in]     aConfig  3D configuration workset
  /// @param [in,out] aResult  1D result workset
  /// @param [in]     aCycle   scalar 
  void 
  evaluateConditional(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  ) const;
}; 

} // namespace Elliptic

} // namespace Plato
