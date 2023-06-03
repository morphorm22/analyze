/*
 *  FactoryCurrentDensityEvaluator_decl.hpp
 *
 *  Created on: June 2, 2023
 */

#pragma once

#include "elliptic/electrical/SupportedParamOptions.hpp"
#include "elliptic/electrical/CurrentDensityEvaluator.hpp"

namespace Plato
{

/// @class FactoryElectricalMaterial
/// @brief Factory for creating electrical material constitutive models
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class FactoryCurrentDensityEvaluator
{
private:
  /// @brief input material parameter list name
  std::string mMaterialName;
  /// @brief map from input material constitutive model string to supported material constitutive model enum
  Plato::electrical::MaterialEnum mS2E;
  /// @brief reference to input problem parameter database
  Teuchos::ParameterList & mParamList;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName input material parameter list name
  /// @param [in] aParamList    input problem parameters
  FactoryCurrentDensityEvaluator(
    const std::string            & aMaterialName,
          Teuchos::ParameterList & aParamList
  );

  /// @brief class destructor
  ~FactoryCurrentDensityEvaluator(){}

  /// @fn create
  /// @brief create current density evaluator
  /// @param aSpatialDomain contains mesh and model information
  /// @param aDataMap       output database
  /// @return shared pointer
  std::shared_ptr<Plato::CurrentDensityEvaluator<EvaluationType>> 
  create(
    const Plato::SpatialDomain & aSpatialDomain,
          Plato::DataMap       & aDataMap
  );
};

} // namespace Plato
