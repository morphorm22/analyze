/*
 * FactoryNonlinearElasticMaterial_decl.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include <memory>

#include "MaterialModel.hpp"
#include "elliptic/mechanical/SupportedParamOptions.hpp"

namespace Plato
{

/// @class FactoryNonlinearElasticMaterial
/// @brief factroy for hyperelastic material models
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class FactoryNonlinearElasticMaterial
{
private:
    /// @brief const reference to input problem parameter list
    const Teuchos::ParameterList& mParamList;
    /// @brief supported mechanical materials interface
    Plato::Elliptic::mechanical::MaterialEnum mS2E;

public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  FactoryNonlinearElasticMaterial(
    Teuchos::ParameterList& aParamList
  );
  
  /// @brief class destructor
  ~FactoryNonlinearElasticMaterial(){}

  /// @brief create material model
  /// @param [in] aMaterialName name of input material parameter list
  /// @return shared pointed
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> 
  create(std::string aMaterialName);
};

}