/*
 * FactoryMechanicalMaterials_decl.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "materials/MaterialModel.hpp"

namespace Plato
{

/// @class FactoryMechanicalMaterials
/// @brief factory for mechanical material constitutive models
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class FactoryMechanicalMaterials
{
public:
  /// @brief class constructor
  FactoryMechanicalMaterials(){}

  /// @brief class destructor
  ~FactoryMechanicalMaterials(){} 

  /// @fn create
  /// @brief create mechanical material constitutive model
  /// @param [in] aMaterialName user assigned name for mechanical material constitutive model
  /// @param [in] aParamList    input problem parameters
  /// @return standard shared pointer to material model
  std::shared_ptr<Plato::MaterialModel<EvaluationType>>
  create(
    const std::string            & aMaterialName,
          Teuchos::ParameterList & aParamList
  );
};

} // namespace Plato
