/*
 *  ResidualSteadyStateCurrent_def.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "NaturalBCs.hpp"
#include "base/ResidualBase.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/electrical/SourceEvaluator.hpp"
#include "elliptic/electrical/CurrentDensityEvaluator.hpp"

namespace Plato
{

/// @class ResidualSteadyStateCurrent
///
/// @brief evaluate steady state current residual, defined as: 
///   \f$R=\int_{\Omega}\phi_{,i}V_{,j}\ d\Omega + \int_{\Gamma}\phi\tau\ d\Gamma - \int_{\Omega}\phi Q\ d\Omega = 0,
/// where /f$\phi/f$ are the interpolation functions, \f$\tau\f$ is an external surface source, \f$Q\f$ is an external
/// volume source, and \f$V\f$ is the electric potential. 
///
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class ResidualSteadyStateCurrent : public Plato::ResidualBase
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of integration points
  static constexpr auto mNumGaussPoints  = ElementType::mNumGaussPoints;
  /// @brief number of spatial dimensions 
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  /// @brief number of degrees of freedom per cell
  static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief typename for base class
  using FunctionBaseType = Plato::ResidualBase;  
  /// @brief contains mesh and model information
  using FunctionBaseType::mSpatialDomain;
  /// @brief contains output data
  using FunctionBaseType::mDataMap;
  /// @brief degrees of freedom names for steady state current physics
  using FunctionBaseType::mDofNames;  
  /// @brief scalar types associated with the evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using GradScalarType    = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief source evaluator interface
  std::shared_ptr<Plato::SourceEvaluator<EvaluationType>> mSourceEvaluator; 
  /// @brief surface boundary condition (Neumann) interface
  std::shared_ptr<Plato::NaturalBCs<ElementType,mNumDofsPerNode>> mSurfaceLoads;  
  /// @brief current density evaluator
  std::shared_ptr<Plato::CurrentDensityEvaluator<EvaluationType>> mCurrentDensityEvaluator;  

public:
  /// @brief class constructor
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output data map
  /// @param [in] aParamList     input problem parameters
  ResidualSteadyStateCurrent(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap,
            Teuchos::ParameterList & aParamList
  );

  /// @brief class destructor
  ~ResidualSteadyStateCurrent();

  /// @fn type
  /// @brief get residual type
  /// @return residual_t enum
  Plato::Elliptic::residual_t
  type() 
  const
  { return Plato::Elliptic::residual_t::LINEAR_ELECTRICAL; }

  /// @fn getSolutionStateOutputData
  /// @brief populate state solution map
  /// @param [in] aSolutions state solution map
  /// @return updated solutions
  Plato::Solutions 
  getSolutionStateOutputData(
      const Plato::Solutions & aSolutions
  ) const;

  /// @fn evaluate
  /// @brief evaluate inner (volume) steady state current residual
  /// @param [in,out] aWorkSets     domain and range workset database
  /// @param [in]     aCycle        cycle scalar
  void
  evaluate(
    Plato::WorkSets & aWorkSets,
    Plato::Scalar     aCycle = 0.0
  ) const;

  /// @fn evaluateBoundary
  /// @brief evaluate outer (boundary) steady state current residual
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in,out] aWorkSets     domain and range workset database
  /// @param [in]     aCycle        cycle scalar
  void
  evaluateBoundary(
    const Plato::SpatialModel & aSpatialModel,
          Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0
  ) const;

private:
  /// @fn initialize
  /// @brief initialize material constitutive model
  /// @param [in] aParamList input problem parameters
  void initialize(
    Teuchos::ParameterList & aParamList
  );
    
};

}