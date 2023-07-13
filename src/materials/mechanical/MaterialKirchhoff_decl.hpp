/*
 * MaterialKirchhoff_decl.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include <vector>

#include <Teuchos_ParameterList.hpp>

#include "materials/MaterialModel.hpp"
#include "elliptic/mechanical/SupportedParamOptions.hpp"

namespace Plato
{

/// @class MaterialKirchhoff
/// @brief material interface for a kirchhoff material model
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class MaterialKirchhoff : public Plato::MaterialModel<EvaluationType>
{
private:
    /// @brief topological element typename
    using ElementType = typename EvaluationType::ElementType;
    /// @brief number of spatial dimensions 
    static constexpr int mNumSpatialDims = ElementType::mNumSpatialDims;
    /// @brief map from input string to supported mechanical property
    Plato::Elliptic::mechanical::PropEnum mS2E;
    /// @brief map from mechanical property enum to list of property values in string format
    std::unordered_map<Plato::Elliptic::mechanical::property,std::vector<std::string>> mProperties;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName input material parameter list name
  /// @param [in] aParamList    input material parameter list
  MaterialKirchhoff(
      const std::string            & aMaterialName,
      const Teuchos::ParameterList & aParamList
  );

  /// @brief class destructor
  ~MaterialKirchhoff(){}

  /// @fn property
  /// @brief return list of property values
  /// @param [in] aPropertyID 
  /// @return standard vector of strings
  std::vector<std::string> 
  property(const std::string & aPropertyID)
  const;
};

}