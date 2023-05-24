/*
 * MaterialElectricalConductivityTwoPhaseAlloy_decl.hpp
 *
 *  Created on: May 23, 2023
 */

#pragma once

/// @include standard cpp includes
#include <vector>
#include <string>
#include <unordered_map>

/// @include analyze includes
#include "MaterialModel.hpp"
#include "elliptic/electrical/SupportedOptionEnums.hpp"

namespace Plato
{

/// @class MaterialElectricalConductivityTwoPhaseAlloy
/// @brief class for electrical conductivity material constitutive models used to model two-phase alloys
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class MaterialElectricalConductivityTwoPhaseAlloy : public MaterialModel<EvaluationType>
/******************************************************************************/
{
private:
    /// @brief topological element type
    using ElementType = typename EvaluationType::ElementType; 
    /// @brief number of spatial dimensions 
    static constexpr int mNumSpatialDims  = ElementType::mNumSpatialDims;
    // number of nodes per cell
    static constexpr int mNumNodesPerCell = ElementType::mNumNodesPerCell;
    /// @brief automatic differentiation types
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    /// @brief exponent for ersatz material penalty model
    Plato::Scalar mPenaltyExponent = 3.0;
    /// @brief minimum value allowed for the ersatz material
    Plato::Scalar mMinErsatzMaterialValue = 0.0;
    /// @brief list of material names 
    std::vector<std::string>   mMaterialNames;
    /// @brief list of material electrical conductivities 
    std::vector<Plato::Scalar> mConductivities;
    /// @brief list of out-of-plane material thicknesses
    std::vector<Plato::Scalar> mOutofPlaneThickness;
    /// @brief map from input property string to supported material property enum
    Plato::electrical::PropEnum mS2E; 
    /// @brief map from material property to list of material property values in string format
    std::unordered_map<Plato::electrical::property,std::vector<std::string>> mProperties;

public:
    /// @brief class constructor
    /// @param [in] aMaterialName name of material parameter list in the input file
    /// @param [in] aParamList    input parameters
    MaterialElectricalConductivityTwoPhaseAlloy(
        const std::string            & aMaterialName, 
              Teuchos::ParameterList & aParamList
    );

    /// @brief destructor 
    ~MaterialElectricalConductivityTwoPhaseAlloy();

    /// @fn computeMaterialTensor
    /// @brief compute material tensor for a two-phase electrical material
    /// @param [in]     aSpatialDomain contains meshed model information
    /// @param [in]     aState         state workset
    /// @param [in]     aControl       control workse
    /// @param [in,out] aResult        material tensor 
    void 
    computeMaterialTensor(
        const Plato::SpatialDomain                         & aSpatialDomain,
        const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray4DT<ResultScalarType>      & aResult
    ) override;

    /// @fn property
    /// @brief return list of material property values in string format
    /// @param [in] aPropertyID input material property name
    /// @return list of string values
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

    /// @fn parseMaterialProperties
    /// @brief parse physical material properties from input file
    /// @param [in] aParamList input problem parameters
    void parseMaterialProperties(
        Teuchos::ParameterList & aParamList
    );
    
    /// @fn parseMaterialNames
    /// @brief parse material names from input file
    /// @param [in] aParamList input problem parameters
    void parseMaterialNames(
        Teuchos::ParameterList & aParamList
    );

    /// @fn parseOutofPlaneThickness
    /// @brief parse out-of-plane material thicknesses from input file
    /// @param [in] aParamList input problem parameters
    void parseOutofPlaneThickness(
        Teuchos::ParameterList &aParamList
    );

    /// @fn parsePenaltyModel
    /// @brief parse parameters associated with the ersatz material penalty model
    /// @param [in] aParamList input problem parameters
    void parsePenaltyModel(
        Teuchos::ParameterList & aParamList
    );

    /// @fn setMaterialTensors
    /// @brief set material tensors for the two-phase material
    void setMaterialTensors();
};

}
// namespace Plato