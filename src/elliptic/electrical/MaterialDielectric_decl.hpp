/*
 * MaterialDielectric_decl.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

/// @include standard cpp includes
#include <vector>
#include <string>
#include <unordered_map>

/// @include analyze includes
#include "materials/MaterialModel.hpp"
#include "elliptic/electrical/SupportedParamOptions.hpp"

namespace Plato
{

/// @class MaterialDielectric
/// @brief class for dielectric material constitutive models
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class MaterialDielectric : public MaterialModel<EvaluationType>
/******************************************************************************/
{
private:
    /// @brief topological element type
    using ElementType = typename EvaluationType::ElementType;
    /// @brief number of spatial dimensions 
    static constexpr int mNumSpatialDims = ElementType::mNumSpatialDims;
    /// @brief map from input material property to supported material property enum
    Plato::Elliptic::electrical::PropEnum mS2E;
    /// @brief map from supported material property enum to list of input material property values in string format
    std::unordered_map<Plato::Elliptic::electrical::property,std::vector<std::string>> mProperties;

public:
    /// @brief class constructor
    /// @param [in] aMaterialName name of material parameter list in the input file
    /// @param [in] aParamList    input parameters
    MaterialDielectric(
        const std::string            & aMaterialName, 
              Teuchos::ParameterList & aParamList
    );

    ~MaterialDielectric();

    /// @fn property
    /// @brief return list of material property values in string format
    /// @param [in] aPropertyID input dielectric material property name
    /// @return standard vector of strings
    std::vector<std::string> 
    property(const std::string & aPropertyID)
    const override;

private:   
    /// @fn initialize
    /// @brief initialize material constitutive model
    /// @param [in] aParamList input problem parameters
    void initialize(
        Teuchos::ParameterList & aParamList
    );
};

}
// namespace Plato