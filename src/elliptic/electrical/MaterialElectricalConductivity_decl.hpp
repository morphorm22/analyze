/*
 * MaterialElectricalConductivity_decl.hpp
 *
 *  Created on: May 23, 2023
 */

#pragma once

#include "MaterialModel.hpp"
#include "elliptic/electrical/SupportedOptionEnums.hpp"

namespace Plato
{

/// @class MaterialElectricalConductivity
/// @brief material model class for electrical conductivity material constitutive models
/// @tparam EvaluationType scalar evaluation automatic differentiation types
template<typename EvaluationType>
class MaterialElectricalConductivity : public MaterialModel<EvaluationType>
{
private:
    /// @brief topological element typename
    using ElementType = typename EvaluationType::ElementType;
    /// @brief number of spatial dimensions 
    static constexpr int mNumSpatialDims = ElementType::mNumSpatialDims;
    
    /// @brief map from input string to supported electrical material property
    Plato::electrical::PropEnum mS2E;
    /// @brief map from electrical material property enum to list of material property values saved as a string
    std::unordered_map<Plato::electrical::property,std::vector<std::string>> mProperties;

public:
    /// @brief class constructor
    /// @param aMaterialName user defined parameter list name from input file
    /// @param aParamList    problem inputs
    MaterialElectricalConductivity(
        const std::string            & aMaterialName,
        const Teuchos::ParameterList & aParamList
    );
    /// @brief class destructor
    ~MaterialElectricalConductivity(){}

    /// @brief return list of material property values saved as a string
    /// @param aPropertyID electrical material property enum
    /// @return standard vector of strings
    std::vector<std::string> 
    property(const std::string & aPropertyID)
    const override;
};

}