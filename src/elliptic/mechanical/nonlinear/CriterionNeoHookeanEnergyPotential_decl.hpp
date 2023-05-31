/*
 * CriterionNeoHookeanEnergyPotential_decl.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "MaterialModel.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"

namespace Plato
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
class CriterionNeoHookeanEnergyPotential : public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief local typename for base class
  using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;
  /// @brief contains mesh and model information
  using FunctionBaseType::mSpatialDomain;
  /// @brief output database
  using FunctionBaseType::mDataMap;
/// @brief scalar types for an evaluation type
  using StateT   = typename EvaluationType::StateScalarType;
  using ConfigT  = typename EvaluationType::ConfigScalarType;
  using ResultT  = typename EvaluationType::ResultScalarType;
  using ControlT = typename EvaluationType::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
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

  /// @fn evaluate_conditional
  /// @brief evaluate criterion on each cell
  /// @param [in]     aState   2D state workset 
  /// @param [in]     aControl 2D control workset
  /// @param [in]     aConfig  3D configuration workset
  /// @param [in,out] aResult  1D result workset
  /// @param [in]     aCycle   scalar 
  void 
  evaluate_conditional(
      const Plato::ScalarMultiVectorT <StateT>   & aState,
      const Plato::ScalarMultiVectorT <ControlT> & aControl,
      const Plato::ScalarArray3DT     <ConfigT>  & aConfig,
            Plato::ScalarVectorT      <ResultT>  & aResult,
            Plato::Scalar                          aCycle
  ) const;
}; 

} // namespace Plato
