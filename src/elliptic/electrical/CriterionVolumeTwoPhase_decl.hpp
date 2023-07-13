/*
 *  CriterionVolumeTwoPhase_decl.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "base/CriterionBase.hpp"
#include "materials/MaterialModel.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

/// @class CriterionVolumeTwoPhase
/// @brief volume (v) criterion for two-phase material use cases. the  
///   criterion is defined as: \f$v = \int_{\Omega}d_{alloy}\, d\Omega\f$, 
///   where \f$d_{alloy}=d_2 + (d_1 - d_2)\theta^{\rho}\f$ is the out-of-plane
///   thickness interpolation function, \f$\theta\f$ is the ersatz material 
///   density value, \f$\rho\f$ is an exponent used to penalized the ersatz 
///   density material and the subscript indices denote the material phases  
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class CriterionVolumeTwoPhase : public Plato::CriterionBase
{
/// @private member data
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions 
  static constexpr int mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of degrees of freedom per node
  static constexpr int mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  /// @brief number of degrees of freedom per cell
  static constexpr int mNumDofsPerCell  = ElementType::mNumDofsPerCell;
  /// @brief number of nodes per cell
  static constexpr int mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief contains mesh and model information
  using Plato::CriterionBase::mSpatialDomain;
  /// @brief output data map
  using Plato::CriterionBase::mDataMap;
  /// @brief scalar types associated with the evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  /// @brief typename for base class
  using FunctionBaseType = typename Plato::CriterionBase;  
  /// @brief penalty exponent for material penalty model
  Plato::Scalar mPenaltyExponent = 3.0;  
  /// @brief list of out-of-plane material thickness
  std::vector<Plato::Scalar> mOutofPlaneThickness;

/// @public functions
public:
  /// @brief class constructor
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output data map
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aFuncName      name of criterion parameter list
  CriterionVolumeTwoPhase(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap,
            Teuchos::ParameterList & aParamList,
      const std::string            & aFuncName
  );

  /// @brief class destructor
  ~CriterionVolumeTwoPhase();

  /// @fn isLinear
  /// @brief returns true if criterion is linear
  /// @return boolean
  bool 
  isLinear() 
  const;

  /// @fn evaluate_conditional
  /// @brief evaluate volume criterion 
  /// @param [in,out] aWorkSets function domain and range workset database
  /// @param [in]     aCycle    scalar 
  void
  evaluateConditional(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  ) const;

/// @private functions
private:
  /// @fn initialize
  /// @brief initialize material constitutive model
  /// @param [in] aParamList input problem parameters
  void 
  initialize(
    Teuchos::ParameterList & aParamList
  );

  /// @fn setOutofPlaneThickness
  /// @brief iset list out-of-plane thicknesses
  /// @param [in] aMaterialModel material model interface
  void 
  setOutofPlaneThickness(
    Plato::MaterialModel<EvaluationType> & aMaterialModel
  );
};
    
}