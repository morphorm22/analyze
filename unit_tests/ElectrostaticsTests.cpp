/*
 * ElectrostaticsTests.cpp
 *
 *  Created on: May 10, 2023
 */

// c++ includes
#include <vector>
#include <unordered_map>

// trilinos includes
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

// plato
#include "Solutions.hpp"
#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "SpatialModel.hpp"
#include "MaterialModel.hpp"
#include "elliptic/AbstractVectorFunction.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief Base class for Linear Electric Conduction models
 */
template<int SpatialDim>
class MaterialElectricConduction : public MaterialModel<SpatialDim>
/******************************************************************************/
{
public:
    MaterialElectricConduction(const Teuchos::ParameterList & aParamList)
    {
        this->parseTensor("Electric Conductivity", aParamList);   
    }
};

/******************************************************************************/
/*!
 \brief Factory for creating material models
 */
template<int SpatialDim>
class FactoryElectricConductionMaterial
/******************************************************************************/
{
public:
    FactoryElectricConductionMaterial(const Teuchos::ParameterList& aParamList) :
            mParamList(aParamList)
    {
    }

    std::shared_ptr<MaterialModel<SpatialDim>> create(std::string aModelName)
    {
        if (!mParamList.isSublist("Material Models"))
        {
            ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
        }
        else
        {
            auto tModelsParamList = mParamList.get < Teuchos::ParameterList > ("Material Models");
            if (!tModelsParamList.isSublist(aModelName))
            {
                auto tMsg = std::string("Requested a material model with name ('") + aModelName 
                            + "') that is not defined in the input deck";
                ANALYZE_THROWERR(tMsg);
            }
    
            auto tModelParamList = tModelsParamList.sublist(aModelName);
            if(tModelParamList.isSublist("Electric Conduction"))
            {
                auto tMaterial = std::make_shared<Plato::MaterialElectricConduction<SpatialDim>>
                                 (tModelParamList.sublist("Electric Conduction"));
                return tMaterial;
            }
            else
            {
                auto tErrMsg = this->getErrorMsg();
                ANALYZE_THROWERR(tErrMsg);
            }
        }
    }

private:
    std::string
    getErrorMsg()
    const
    {
        std::string tMsg = std::string("ERROR: Requested a material constitutive model that is not supported. ")
            + "Supported material constitutive models for an electrostatics analysis are: ";
        for(const auto& tElement : mSupportedMaterials)
        {
            tMsg = tMsg + "'" + tElement + "', ";
        }
        auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
        return tSubMsg;
    }

private:
    const Teuchos::ParameterList& mParamList;
    
    std::vector<std::string> mSupportedMaterials = {"Electric Conduction"};
};

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType>
class ElectrostaticsResidual : public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = Plato::Elliptic::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    std::shared_ptr<Plato::MaterialModel<mNumSpatialDims>> mMaterialModel;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>>   mBodyLoads;
    std::shared_ptr<Plato::NaturalBCs<ElementType, mNumDofsPerNode>> mSurfaceLoads;

    std::vector<std::string> mPlottable;

public:
    ElectrostaticsResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams
    )
    {
        // obligatory: define dof names in order
        mDofNames.push_back("electric_potential");
        // create material constitutive model
        Plato::FactoryElectricConductionMaterial<mNumSpatialDims> tMaterialFactory(aProblemParams);
        mMaterialModel = tMaterialFactory.create(aSpatialDomain.getMaterialName());
        // TODO: create body loads
        // TODO: create surface loads
        auto tResidualParams = aProblemParams.sublist("Elliptic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
            mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
    }
    ~ElectrostaticsResidual(){}

    Plato::Solutions 
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override
    { return aSolutions; }

    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {}

    void
    evaluate_boundary(
        const Plato::SpatialModel                           & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {}
};

}

}

namespace ElectrostaticsTest
{

}