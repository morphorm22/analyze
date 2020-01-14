/*
 * StructuralDynamics.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef STRUCTURALDYNAMICS_HPP_
#define STRUCTURALDYNAMICS_HPP_

#include <string>
#include <memory>
#include <sstream>

#include <Teuchos_ParameterList.hpp>

#include "Simp.hpp"
#include "Ramp.hpp"
#include "ExpVolume.hpp"
#include "DynamicCompliance.hpp"
#include "AbstractVectorFunction.hpp"
#include "StructuralDynamicsResidual.hpp"
#include "HyperbolicTangentProjection.hpp"
#include "AdjointStructuralDynamicsResidual.hpp"

namespace Plato
{

namespace StructuralDynamicsFactory
{

/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<AbstractVectorFunction<EvaluationType>>
    createVectorFunction(Omega_h::Mesh& aMesh, 
                         Omega_h::MeshSets& aMeshSets, 
                         Plato::DataMap& aDataMap, 
                         Teuchos::ParameterList& aParamList,   
                         const std::string & aFunctionType)
    /******************************************************************************/
    {
        if(aFunctionType == "StructuralDynamics")
        {
            assert(aParamList.isSublist(aFunctionType));
            assert(aParamList.isSublist("Frequency Steps"));
            assert(aParamList.sublist(aFunctionType).isSublist("Penalty Function"));

            auto tPenaltyParams = aParamList.sublist(aFunctionType).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::StructuralDynamicsResidual<EvaluationType, Plato::MSIMP, Plato::HyperbolicTangentProjection>>
                        (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::StructuralDynamicsResidual<EvaluationType, Plato::RAMP, Plato::HyperbolicTangentProjection>>
                        (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            {
                return std::make_shared<Plato::StructuralDynamicsResidual<EvaluationType, Plato::MSIMP, Plato::HyperbolicTangentProjection>>
                        (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
        }
        else if(aFunctionType == "StructuralDynamics Adjoint")
        {
            assert(aParamList.isSublist("Frequency Steps"));
            assert(aParamList.sublist("StructuralDynamics").isSublist("Penalty Function"));

            auto tPenaltyParams = aParamList.sublist("StructuralDynamics").sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::AdjointStructuralDynamicsResidual<EvaluationType, Plato::MSIMP, Plato::HyperbolicTangentProjection>>
                        (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::AdjointStructuralDynamicsResidual<EvaluationType, Plato::RAMP, Plato::HyperbolicTangentProjection>>
                        (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            {
                return std::make_shared<Plato::AdjointStructuralDynamicsResidual<EvaluationType, Plato::MSIMP, Plato::HyperbolicTangentProjection>>
                        (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
        }
        else
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n**************\n" << " ERROR IN FILE: " << __FILE__ << ", FUNCTION: "
                    << __PRETTY_FUNCTION__ << ", LINE: " << __LINE__ << "\n\n" << " MESSAGE: UNKNOWN 'PDE CONSTRAINT = "
                    << aFunctionType.c_str() << "' SPECIFIED IN 'PLATO PROBLEM' BLOCK."
                    << " USER SHOULD MAKE SURE THAT THE PDE CONSTRAINT IS SUPPORTED OR PROPERLY IN THE INPUT FILE.\n"
                    << "**************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
    }
    /******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<AbstractScalarFunction<EvaluationType>>
    createScalarFunction(Omega_h::Mesh& aMesh, 
                         Omega_h::MeshSets& aMeshSets, 
                         Plato::DataMap& aDataMap, 
                         Teuchos::ParameterList & aParamList, 
                         const std::string & aFunctionType,
                         const std::string & aFunctionName)
    /******************************************************************************/
    {
        if(aFunctionType == "Dynamic Compliance")
        {
            assert(aParamList.isSublist(aFunctionName));
            assert(aParamList.isSublist("Frequency Steps"));
            assert(aParamList.sublist(aFunctionName).isSublist("Penalty Function"));

            auto tPenaltyParams = aParamList.sublist(aFunctionName).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::DynamicCompliance<EvaluationType, Plato::MSIMP, Plato::HyperbolicTangentProjection>>
                        (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::DynamicCompliance<EvaluationType, Plato::RAMP, Plato::HyperbolicTangentProjection>>
                        (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else
            {
                return std::make_shared<Plato::DynamicCompliance<EvaluationType, Plato::MSIMP, Plato::HyperbolicTangentProjection>>
                        (aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
        }
        else if(aFunctionType == "Volume")
        {
            assert(aParamList.isSublist(aFunctionName));
            assert(aParamList.sublist(aFunctionName).isSublist("Penalty Function"));

            auto tPenaltyParams = aParamList.sublist(aFunctionName).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<ExpVolume<EvaluationType, Plato::MSIMP, Plato::HyperbolicTangentProjection>>(aMesh, aMeshSets, aDataMap, tPenaltyParams);
            }
            else if(tPenaltyType == "RAMP")
            {
                return std::make_shared<ExpVolume<EvaluationType, Plato::RAMP, Plato::HyperbolicTangentProjection>>(aMesh, aMeshSets, aDataMap, tPenaltyParams);
            }
            else
            {
                return std::make_shared<ExpVolume<EvaluationType, Plato::MSIMP, Plato::HyperbolicTangentProjection>>(aMesh, aMeshSets, aDataMap, tPenaltyParams);
            }
        }
        else
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n**************\n" << " ERROR IN FILE: " << __FILE__ << ", FUNCTION: "
                    << __PRETTY_FUNCTION__ << ", LINE: " << __LINE__ << "\n\n"
                    << " MESSAGE: UNKNOWN SCALAR FUNCTION 'TYPE = " << aFunctionType.c_str()
                    << "' SPECIFIED IN PLATO PROBLEM BLOCK."
                    << " USER SHOULD MAKE SURE THIS SCALAR FUNCTION IS SUPPORTED OR PROPERLY DEFINED IN THE INPUT FILE.\n"
                    << "**************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
    }
}; // struct FunctionFactory

} // namespace StructuralDynamicsFactory

template<Plato::OrdinalType SpaceDimParam, Plato::OrdinalType NumControls = 1>
class StructuralDynamics: public SimplexStructuralDynamics<SpaceDimParam, NumControls>
{
public:
    typedef Plato::StructuralDynamicsFactory::FunctionFactory FunctionFactory;
    using SimplexT = SimplexStructuralDynamics<SpaceDimParam, NumControls>;
    static constexpr int SpaceDim = SpaceDimParam;
}; // class StructuralDynamics

} // namespace Plato

#endif /* STRUCTURALDYNAMICS_HPP_ */
