/*
 * FactoryStressEvaluator_decl.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "SpatialModel.hpp"

#include "elliptic/mechanical/SupportedParamOptions.hpp"
#include "elliptic/mechanical/nonlinear/StressEvaluator.hpp"

namespace Plato
{
    
/// @class FactoryStressEvaluator
/// @brief Factory of mechanical stress evaluators
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class FactoryStressEvaluator
{
private:
  /// @brief name of input material parameter list
  std::string mMaterialName; 
  /// @brief supported mechanical materials interface
  Plato::mechanical::MaterialEnum mS2E;

public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  FactoryStressEvaluator(
    const std::string & aMaterialName
  );

  /// @brief class destructor
  ~FactoryStressEvaluator(){}
  
  /// @fn create 
  /// @brief create stress evaluator
  /// @param [in] aMaterialName  name of input material parameter list
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output database
  /// @return shared pointer
  std::shared_ptr<Plato::StressEvaluator<EvaluationType>> 
  create(
          Teuchos::ParameterList & aParamList,
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap
  );
};

} // namespace Plato
