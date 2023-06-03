/*
 *  ResidualSteadyStateCurrent_def.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "NaturalBCs.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/AbstractVectorFunction.hpp"
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
class ResidualSteadyStateCurrent : public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
{
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
  /// @brief typename for base class
  using FunctionBaseType = Plato::Elliptic::AbstractVectorFunction<EvaluationType>;  
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
  std::shared_ptr<Plato::NaturalBCs<ElementType, mNumDofsPerNode>> mSurfaceLoads;  
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

  // @fn getSolutionStateOutputData
  /// @brief populate state solution map
  /// @param [in] aSolutions state solution map
  /// @return updated solutions
  Plato::Solutions 
  getSolutionStateOutputData(
      const Plato::Solutions &aSolutions
  ) const override;

  /// @fn evaluate
  /// @brief evaluate inner (volume) steady state current residual
  /// @param [in]     aState   2D state workset
  /// @param [in]     aControl 2D control workset
  /// @param [in]     aConfig  3D configuration workset
  /// @param [in,out] aResult  2D result workset
  /// @param [in]     aCycle   scalar
  void
  evaluate(
      const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
      const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
      const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
            Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
            Plato::Scalar                                   aCycle = 1.0
  ) const override;

  /// @fn evaluate_boundary
  /// @brief evaluate outer (boundary) steady state current residual
  /// @param [in]     aState   2D state workset
  /// @param [in]     aControl 2D control workset
  /// @param [in]     aConfig  3D configuration workset
  /// @param [in,out] aResult  2D result workset
  /// @param [in]     aCycle   scalar
  void
  evaluate_boundary(
      const Plato::SpatialModel                           & aSpatialModel,
      const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
      const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
      const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
            Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
            Plato::Scalar                                   aCycle = 0.0
  ) const override;

private:
  /// @fn initialize
  /// @brief initialize material constitutive model
  /// @param [in] aParamList input problem parameters
  void initialize(
    Teuchos::ParameterList & aParamList
  );
    
};

}