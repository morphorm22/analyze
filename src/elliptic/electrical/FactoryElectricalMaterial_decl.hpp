/*
 * FactoryElectricalMaterial_decl.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

/// @include standard cpp includes
#include <vector>
#include <string>

/// @include analyze includes
#include "materials/MaterialModel.hpp"

namespace Plato
{

/// @class FactoryElectricalMaterial
/// @brief Factory for creating electrical material constitutive models
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class FactoryElectricalMaterial
{
private:
    /// @brief const reference to input problem parameter list
    const Teuchos::ParameterList& mParamList;
    /// @brief list of supported input material options
    std::vector<std::string> mSupportedMaterials = 
      {"Electrical Conductivity", "Dielectric", "Two Phase Conductive"};
      
public:
    /// @brief class constructor
    /// @param [in] aParamList input parameters
    FactoryElectricalMaterial(
        Teuchos::ParameterList& aParamList
    );
    
    /// @brief class destructor
    ~FactoryElectricalMaterial();

    /// @fn create
    /// @brief create shared pointer to electrical material constitutive model
    /// @param [in] aModelName input material parameter list name
    /// @return standard shared pointer
    std::shared_ptr<MaterialModel<EvaluationType>> 
    create(
        std::string aModelName
    );

private:
    /// @fn getErrorMsg
    /// @brief return error message
    /// @return string
    std::string
    getErrorMsg()
    const;
};

}
// namespace Plato