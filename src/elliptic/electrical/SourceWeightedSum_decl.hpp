/*
 *  SourceWeightedSum_decl.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

/// @include standard cpp includes
#include <memory>
#include <vector>

/// @include analyze includes
#include "elliptic/electrical/SourceEvaluator.hpp"
#include "elliptic/electrical/CurrentDensitySourceEvaluator.hpp"

namespace Plato
{

/// @class SourceWeightedSum
/// @brief weighted sum of source terms; i.e., \f$Q = \sum_{i=1}^{N_Q}\hat{Q}_i\f$
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class SourceWeightedSum : public Plato::SourceEvaluator<EvaluationType>
{
private:
  /// @brief scalar types associated with the evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;  
  /// @brief input material model parameter list
  std::string mMaterialName = "";
  /// @brief list of source functions
  std::vector<std::string>   mFunctions;
  /// @brief list of source weights
  std::vector<Plato::Scalar> mFunctionWeights;
  /// @brief list of current density evaluators
  std::vector<std::shared_ptr<Plato::CurrentDensitySourceEvaluator<EvaluationType>>> mCurrentDensitySourceEvaluators;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName name of material parameter list in the input file
  /// @param [in] aParamList    input parameters
  SourceWeightedSum(
    const std::string            & aMaterialName,
          Teuchos::ParameterList & aParamList
  );

  /// @brief class destructor  
  ~SourceWeightedSum();

  /// @fn evaluate
  /// @brief evaluate weighted sum of source terms
  /// @param [in]     aSpatialDomain contains meshed model information
  /// @param [in]     aState         2D state workset
  /// @param [in]     aControl       2D control workset
  /// @param [in]     aConfig        3D configuration workset
  /// @param [in,out] aResult        2D result workset
  /// @param [in]     aScale         scalar
  void 
  evaluate(
      const Plato::SpatialDomain                         & aSpatialDomain,
      const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
      const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
      const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
      const Plato::Scalar                                & aScale
  ) const;

private:
  /// @fn initialize
  /// @brief initialize material constitutive model
  /// @param [in] aParamList input problem parameters
  void initialize(
    Teuchos::ParameterList & aParamList
  );

  /// @fn parseFunctions
  /// @brief parse input source functions
  /// @param [in] aParamList input problem parameters
  void parseFunctions(
    Teuchos::ParameterList & aParamList
  );

  /// @fn parseWeights
  /// @brief parse weights associated with source terms 
  /// @param [in] aParamList input problem parameters
  void parseWeights(
    Teuchos::ParameterList & aParamList      
  );

  /// @fn createCurrentDensitySourceEvaluators
  /// @brief create current density evaluators
  /// @param [in] aParamList input problem parameters
  void createCurrentDensitySourceEvaluators(
    Teuchos::ParameterList & aParamList  
  );

};

}