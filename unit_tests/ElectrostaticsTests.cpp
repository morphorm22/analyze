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
#include "ScalarGrad.hpp"
#include "SpatialModel.hpp"
#include "MaterialModel.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"
#include "GeneralFluxDivergence.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/AbstractVectorFunction.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief Base class for linear electric conductivity material model
 */
template<typename EvaluationType>
class MaterialElectricConductivity : public MaterialModel<EvaluationType>
/******************************************************************************/
{
private:
    using ElementType = typename EvaluationType::ElementType; // set local element type
    using ElementType::mNumSpatialDims;  /*!< spatial dimensions */

    std::string mName = "";
    
public:
    MaterialElectricConductivity(
        const std::string            & aMaterialName,
        const Teuchos::ParameterList & aParamList
    ) : 
      mName(aMaterialName)
    {
        this->parseScalar("Electric Conductivity", aParamList);   
        auto tElectricConductivity = this->getScalarConstant("Electric Conductivity");
        this->setTensorConstant("material tensor",Plato::TensorConstant<mNumSpatialDims>(tElectricConductivity));
    }
    ~MaterialElectricConductivity(){}
};

/******************************************************************************/
/*!
 \brief Base class for linear electric conductivity material model
 */
template<typename EvaluationType>
class MaterialElectricConductivityTwoPhaseAlloy : public MaterialModel<EvaluationType>
/******************************************************************************/
{
private:
    using ElementType = typename EvaluationType::ElementType; // set local element type
    using ElementType::mNumSpatialDims;  /*!< spatial dimensions */

    std::string mMaterialName = "";
    Plato::OrdinalType mNumMaterials = 0;
    std::vector<Plato::Scalar> mConductivities;

public:
    MaterialElectricConductivityTwoPhaseAlloy(
        const std::string            & aModelName, 
        const Teuchos::ParameterList & aParamList
    ) : 
      mMaterialName(aModelName)
    {
        this->parseProperties(aParamList);
        this->setTensors();
    }
    ~MaterialElectricConductivityTwoPhaseAlloy(){}

private:
    void parseProperties(
        const Teuchos::ParameterList & aParamList
    )
    {
        bool tIsArray = aParamList.isType<Teuchos::Array<Plato::Scalar>>("Electric Conductivity");
        if (tIsArray)
        {
            // parse inputs
            Teuchos::Array<Plato::Scalar> tConductivities = 
              aParamList.get<Teuchos::Array<Plato::Scalar>>("Electric Conductivity");
            // create mirror 
            mNumMaterials = tConductivities.size();
            for( Plato::OrdinalType tIndex = 0; tIndex < mNumMaterials; tIndex++)
            {
                mConductivities.push_back(tConductivities[tIndex]);
            }
        }
        else
        {
            auto tMsg = std::string("Array of electric conductivities is not defined in material block with name '") 
                + mMaterialName + "'. Material tensor for the two-phase alloy material cannnot be computed.";
            ANALYZE_THROWERR(tMsg)
        }
    }
    void setTensors()
    {
        for(const auto& tConductivity : mConductivities){
            Plato::OrdinalType tIndex = &tConductivity - &mConductivities[0];
            auto tName = std::string("material tensor ") + std::to_string(tIndex);
            this->setTensorConstant(tName,Plato::TensorConstant<mNumSpatialDims>(tConductivity));
        }
    }

    
};

/******************************************************************************/
/*!
 \brief Base class for linear dielectric material model
 */
template<typename EvaluationType>
class MaterialDielectric : public MaterialModel<EvaluationType>
/******************************************************************************/
{
private:
    using ElementType = typename EvaluationType::ElementType; // set local element type
    using ElementType::mNumSpatialDims;  /*!< spatial dimensions */
    
    std::string mName = "";

public:
    MaterialDielectric(
        const std::string            & aModelName, 
        const Teuchos::ParameterList & aParamList
    ) : 
      mName(aModelName)
    {
        this->parseScalar("Electric Constant", aParamList);
        this->parseScalar("Relative Static Permittivity", aParamList);
        auto tElectricConstant = this->getScalarConstant("Electric Constant");
        auto tRelativeStaticPermittivity = this->getScalarConstant("Relative Static Permittivity");
        auto tValue = tElectricConstant * tRelativeStaticPermittivity;
        this->setTensorConstant("material tensor",Plato::TensorConstant<mNumSpatialDims>(tValue));
    }
    ~MaterialDielectric(){}
};

/******************************************************************************/
/*!
 \brief Factory for creating electric material models
 */
template<typename EvaluationType>
class FactoryElectricMaterial
/******************************************************************************/
{
public:
    FactoryElectricMaterial(
        const Teuchos::ParameterList& aParamList
    ) :
      mParamList(aParamList)
    {
    }

    std::shared_ptr<MaterialModel<EvaluationType>> create(std::string aModelName)
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
            if(tModelParamList.isSublist("Electric Conductivity"))
            {
                auto tMaterial = std::make_shared<Plato::MaterialElectricConductivity<EvaluationType>>
                                 (aModelName, tModelParamList.sublist("Electric Conductivity"));
                return tMaterial;
            }
            else 
            if(tModelParamList.isSublist("Dielectric"))
            {
                auto tMaterial = std::make_shared<Plato::MaterialDielectric<EvaluationType>>
                                (aModelName, tModelParamList.sublist("Dielectric"));
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
    
    std::vector<std::string> mSupportedMaterials = {"Electric Conductivity", "Dielectric"};
};

template<typename EvaluationType>
class CurrentDensity
{
private:
    using ElementType = typename EvaluationType::ElementType; /*<! local element type */
    using ElementType::mNumSpatialDims;
    
public:
    CurrentDensity(){}
    ~CurrentDensity(){}

    template<
      typename TScalarType, 
      typename TGradScalarType, 
      typename TMatScalarType,
      typename TOutScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
            Plato::Array<mNumSpatialDims,TOutScalarType>                  & aCurrentDensity,
      const Plato::Array<mNumSpatialDims,TGradScalarType>                 & aStateGradient,
      const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,TMatScalarType> & aMaterialTensor,
      const TScalarType                                                   & aElecPotential
    ) const
    {
        // compute thermal flux - linear case
        for (Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++) 
        {
            aCurrentDensity(tDimI) = 0.0;
            for (Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++) 
            {
                aCurrentDensity(tDimJ) += aMaterialTensor(tDimI,tDimJ) * aStateGradient(tDimJ);
            }
        }
    }
};
// class CurrentDensity

template<typename EvaluationType>
class DarkCurrent
{
private:
    // set local element type
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumSpatialDims;  /*!< spatial dimensions */
    using ElementType::mNumDofsPerNode;  /*!< number of degrees of freedom per node */
    using ElementType::mNumNodesPerCell; /*!< number of nodes per cell/element */

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    std::unordered_map<std::string,Plato::Scalar> mParameters;

public:
    DarkCurrent(Teuchos::ParameterList &aParamList)
    {
        if( !aParamList.isSublist("Dark Current") )
        { ANALYZE_THROWERR("Parameter in Body Loads block is not valid. Expects a Parameter lists only."); }
        Teuchos::ParameterList& tSublist = aParamList.sublist("Dark Current");

        mParameters["a"]  = tSublist.get<Plato::Scalar>("a",0.);
        mParameters["b"]  = tSublist.get<Plato::Scalar>("b",1.27e-6);
        mParameters["c"]  = tSublist.get<Plato::Scalar>("c",25.94253);
        mParameters["m1"] = tSublist.get<Plato::Scalar>("m1",0.38886);
        mParameters["b1"] = tSublist.get<Plato::Scalar>("b1",0.);
        mParameters["m2"] = tSublist.get<Plato::Scalar>("m2",30.);
        mParameters["b2"] = tSublist.get<Plato::Scalar>("b2",6.520373);

        mParameters["limit"] = tSublist.get<Plato::Scalar>("limit",-0.22);
    }
    ~DarkCurrent(){}

    void
    get(
        const Plato::SpatialDomain                         & aSpatialDomain,
        const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
        const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
              Plato::Scalar                                  aScale
    ) const
    {
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        // interpolate nodal values to integration points
        Plato::InterpolateFromNodal<ElementType,mNumDofsPerNode> tInterpolateFromNodal;

        // parameters for dark current model
        Plato::Scalar tCoefA  = mParameters.find("a");
        Plato::Scalar tCoefB  = mParameters.find("b");
        Plato::Scalar tCoefC  = mParameters.find("c");
        Plato::Scalar tCoefM1 = mParameters.find("m1");
        Plato::Scalar tCoefB1 = mParameters.find("b1");
        Plato::Scalar tCoefM2 = mParameters.find("m2");
        Plato::Scalar tCoefB2 = mParameters.find("b2");
        Plato::Scalar tPerformanceLimit = mParameters.find("limit");

        // integrate and assemble
        Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
        Kokkos::parallel_for("compute dark current", 
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
          KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tDetJ = Plato::determinant(ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal));

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            ControlScalarType tErsatzDensity = tInterpolateFromNodal(iCellOrdinal,tBasisValues,aControl);
            StateScalarType tCellElectricPotential = tInterpolateFromNodal(iCellOrdinal,tBasisValues,aState);

            StateScalarType tDarkCurrentDensity = 0.0;
            if( tCellElectricPotential > 0.0 )
              { tDarkCurrentDensity = tCoefA + tCoefB * exp(tCoefC * tCellElectricPotential); }
            else 
            if( (tPerformanceLimit < tCellElectricPotential) && (tCellElectricPotential < 0.0) )
              { tDarkCurrentDensity = tCoefM1 * tCellElectricPotential + tCoefB1; }
            else 
            if( tCellElectricPotential < tPerformanceLimit )
              { tDarkCurrentDensity = tCoefM2 * tCellElectricPotential + tCoefB2; }

            auto tWeight = aScale * tCubWeights(iGpOrdinal) * tDetJ;
            for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < mNumNodesPerCell; tFieldOrdinal++)
            {
                ResultScalarType tCellResult = 
                  tWeight * tDarkCurrentDensity * tBasisValues(tFieldOrdinal) * tErsatzDensity; 
                Kokkos::atomic_add( &aResult(iCellOrdinal,tFieldOrdinal*mNumDofsPerNode),tCellResult );
            }
        });
    }
};

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType>
class ElectrostaticsResidual : public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    // set local element type
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

    std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;

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
        Plato::FactoryElectricMaterial<EvaluationType> tMaterialFactory(aProblemParams);
        mMaterialModel = tMaterialFactory.create(aSpatialDomain.getMaterialName());
        // TODO: create body loads
        // TODO: create surface loads
        auto tResidualParams = aProblemParams.sublist("Elliptic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
            mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
        this->parseBodyLoads();
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
    {
        using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
        
        // inline functors
        Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
        Plato::GeneralFluxDivergence<ElementType> tComputeDivergence;
        Plato::ScalarGrad<ElementType>            tComputeScalarGrad;
        Plato::CurrentDensity<EvaluationType>     tComputeCurrentDensity;  

        // interpolate nodal values to integration points
        Plato::InterpolateFromNodal<ElementType,mNumDofsPerNode> tInterpolateFromNodal;

        // quantity of interests
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVectorT<ConfigScalarType>      
          tVolume("InterpolateFromNodalvolume",tNumCells);
        Plato::ScalarMultiVectorT<GradScalarType>   
          tElectricField("electric field", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultScalarType> 
          tCurrentDensity("current density", tNumCells, mNumSpatialDims);

        // evaluate residual    
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();      
        Kokkos::parallel_for("evaluate electrostatics residual", 
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
          KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigScalarType tCellVolume(0.0);
  
            Plato::Array<mNumSpatialDims,GradScalarType>   tCellElectricField(0.0);
            Plato::Array<mNumSpatialDims,ResultScalarType> tCellCurrentDensity(0.0);
            Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
  
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            tComputeGradient(iCellOrdinal,tCubPoint,aConfig,tGradient,tCellVolume);
            tComputeScalarGrad(iCellOrdinal,tCellElectricField,aState,tGradient);
            StateScalarType tCellElectricPotential = tInterpolateFromNodal(iCellOrdinal,tBasisValues,aState);
            //tComputeCurrentDensity(tCellCurrentDensity,tCellElectricField,tCellElectricPotential);
      
            tCellVolume *= tCubWeights(iGpOrdinal);
            tComputeDivergence(iCellOrdinal,aResult,tCellCurrentDensity,tGradient,tCellVolume,-1.0);
          
            for(Plato::OrdinalType tIndex=0; tIndex<mNumSpatialDims; tIndex++)
            {
                Kokkos::atomic_add(&tElectricField(iCellOrdinal,tIndex),  -1.0*tCellVolume*tCellElectricField(tIndex));
                Kokkos::atomic_add(&tCurrentDensity(iCellOrdinal,tIndex), tCellVolume*tCellCurrentDensity(tIndex));
            }
            Kokkos::atomic_add(&tVolume(iCellOrdinal),tCellVolume);
        });

        Kokkos::parallel_for("compute cell quantities", 
          Kokkos::RangePolicy<>(0, tNumCells),
          KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
        {
            for(Plato::OrdinalType tIndex=0; tIndex<mNumSpatialDims; tIndex++)
            {
                tElectricField(iCellOrdinal,tIndex)  /= tVolume(iCellOrdinal);
                tCurrentDensity(iCellOrdinal,tIndex) /= tVolume(iCellOrdinal);
            }
        });

        if( std::count(mPlottable.begin(),mPlottable.end(),"electric field") ) 
        { toMap(mDataMap, tElectricField, "electric field", mSpatialDomain); }
        if( std::count(mPlottable.begin(),mPlottable.end(),"current density" ) )
        { toMap(mDataMap, tCurrentDensity, "current density" , mSpatialDomain); }
    }

    void
    evaluate_boundary(
        const Plato::SpatialModel                           & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
        // add contributions from natural boundary conditions
        if( mSurfaceLoads != nullptr )
        {
            mSurfaceLoads->get(aSpatialModel,aState,aControl,aConfig,aResult,1.0);
        }
    }

private:
    void parseBodyLoads(Teuchos::ParameterList & aParamList)
    {
        if(aParamList.isSublist("Body Loads"))
        {
            mBodyLoads = 
                std::make_shared<Plato::BodyLoads<EvaluationType, ElementType>>(aParamList.sublist("Body Loads"));
        }
    }
};

}

}

namespace ElectrostaticsTest
{

}