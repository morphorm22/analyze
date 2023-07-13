/*
 * FactoryElasticMaterial_decl.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include <memory>

#include "materials/MaterialModel.hpp"

namespace Plato
{

/// @class FactoryElasticMaterial
/// @brief factory for elastic material constitutive models
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class FactoryElasticMaterial
{
private:
  /// @brief const reference to input problem parameters
  const Teuchos::ParameterList& mParamList;
  /// @brief supported elastic material constitutive models
  std::vector<std::string> mSupportedMaterials =
    {"isotropic linear elastic"};

public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  FactoryElasticMaterial(
    const Teuchos::ParameterList& aParamList
  );
  
  /// @fn create
  /// @brief create elastic material constitutive model
  /// @param [in] aModelName user assigned name for material model 
  /// @return standard shared pointer
  std::shared_ptr<Plato::MaterialModel<EvaluationType>>
  create(
    std::string aModelName
  ) const;

private:
  /// @fn getErrorMsg
  /// @brief error message if requested elastic material constitutive model is not supported
  /// @return string
  std::string
  getErrorMsg()
  const;
};

} // namespace Plato
