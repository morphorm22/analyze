/*
 * MaterialElectricalConductivity_decl.hpp
 *
 *  Created on: May 23, 2023
 */

#pragma once

#include "materials/MaterialModel.hpp"
#include "elliptic/electrical/SupportedParamOptions.hpp"

namespace Plato
{

/// @class MaterialElectricalConductivity
/// @brief material model class for electrical conductivity material constitutive models
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class MaterialElectricalConductivity : public MaterialModel<EvaluationType>
{
private:
    /// @brief topological element typename
    using ElementType = typename EvaluationType::ElementType;
    /// @brief number of spatial dimensions 
    static constexpr int mNumSpatialDims = ElementType::mNumSpatialDims;
    /// @brief map from input string to supported electrical material property
    Plato::Elliptic::electrical::PropEnum mS2E;
    /// @brief map from electrical material property enum to list of material property values saved as a string
    std::unordered_map<Plato::Elliptic::electrical::property,std::vector<std::string>> mProperties;

public:
    /// @brief class constructor
    /// @param [in] aMaterialName user defined parameter list name from input file
    /// @param [in] aParamList    problem inputs
    MaterialElectricalConductivity(
        const std::string            & aMaterialName,
        const Teuchos::ParameterList & aParamList
    );
    /// @brief class destructor
    ~MaterialElectricalConductivity(){}

    /// @brief return list of material property values in string format
    /// @param [in] aPropertyID input electrical material property name
    /// @return standard vector of strings
    std::vector<std::string> 
    property(const std::string & aPropertyID)
    const override;
};

}