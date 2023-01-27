/*
 * IfemTests.cpp
 *
 *  Created on: Nov 21, 2022
 */

// c++ includes
#include <vector>
#include <unordered_map>

// trilinos includes
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

// immersus includes
#include "Simp.hpp"
#include "BLAS1.hpp"
#include "FadTypes.hpp"
#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "Assembly.hpp"
#include "CellVolume.hpp"
#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "WorksetBase.hpp"
#include "SurfaceArea.hpp"
#include "SmallStrain.hpp"
#include "UtilsTeuchos.hpp"
#include "EssentialBCs.hpp"
#include "SpatialModel.hpp"
#include "LinearStress.hpp"
#include "MaterialModel.hpp"
#include "AnalyzeOutput.hpp"
#include "ScalarProduct.hpp"
#include "PlatoMathExpr.hpp"
#include "PlatoMeshExpr.hpp"
#include "PlatoUtilities.hpp"
#include "GradientMatrix.hpp"
#include "ApplyWeighting.hpp"
#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "ApplyConstraints.hpp"
#include "PlatoStaticsTypes.hpp"
#include "ElasticModelFactory.hpp"
#include "WeightedNormalVector.hpp"
#include "PlatoAbstractProblem.hpp"
#include "InterpolateFromNodal.hpp"
#include "GeneralFluxDivergence.hpp"
#include "GeneralStressDivergence.hpp"

#include "alg/PlatoSolverFactory.hpp"
#include "alg/PlatoAbstractSolver.hpp"
#include "elliptic/EvaluationTypes.hpp"

// unit test includes
#include "util/PlatoTestHelpers.hpp"
#include "Analyze_Diagnostics.hpp"

namespace Plato
{

namespace exp
{

template<Plato::OrdinalType SpatialDim>
class MaterialElastic : public MaterialModel<SpatialDim>
{
public:
    MaterialElastic(const Teuchos::ParameterList& aParamList)
    {
        this->parse(aParamList);
        this->computeLameConstants();
    }

    Plato::Scalar mu() const
    { return this->getScalarConstant("mu"); }

    Plato::Scalar lambda() const
    { return this->getScalarConstant("lambda"); }

private:
    void parse(const Teuchos::ParameterList& aParamList)
    {
        this->parseScalarConstant("Youngs Modulus", aParamList);
        this->parseScalarConstant("Poissons Ratio", aParamList);
    }
    void computeLameConstants()
    {
        auto tYoungsModulus = this->getScalarConstant("youngs modulus");
        if(tYoungsModulus <= std::numeric_limits<Plato::Scalar>::epsilon())
        {
            ANALYZE_THROWERR(std::string("Error: The Young's Modulus is less than the machine epsilon. ")
                + "The input material properties were not parsed properly.");
        }

        auto tPoissonsRatio = this->getScalarConstant("poissons ratio");
        if(tPoissonsRatio <= std::numeric_limits<Plato::Scalar>::epsilon())
        {
            ANALYZE_THROWERR(std::string("Error: The Poisson's Ratio is less than the machine epsilon. ")
                + "The input material properties were not parsed properly.");
        }

        auto tMu = tYoungsModulus / (2.0 * (1.0 + tPoissonsRatio) );
        this->setScalarConstant("mu",tMu);
        auto tLambda = (tYoungsModulus * tPoissonsRatio) / ( (1.0 + tPoissonsRatio) * (1.0 - 2.0 * tPoissonsRatio) );
        this->setScalarConstant("lambda",tLambda);
    }
};

/******************************************************************************//**
 * \brief Factory for creating linear elastic material models.
 *
 * \tparam SpatialDim spatial dimensions: options 1D, 2D, and 3D
 *
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class FactoryElasticMaterial
{
private:
    const Teuchos::ParameterList& mParamList; /*!< Input parameter list */

public:
    /******************************************************************************//**
    * \brief Linear elastic material model factory constructor.
    * \param [in] aParamList input parameter list
    **********************************************************************************/
    FactoryElasticMaterial(const Teuchos::ParameterList& aParamList) :
        mParamList(aParamList){}

    /******************************************************************************//**
    * \brief Create a linear elastic material model.
    * \param [in] aModelName name of the model to be created.
    * \return Teuchos reference counter pointer to linear elastic material model
    **********************************************************************************/
    std::shared_ptr<Plato::MaterialModel<SpatialDim>>
    create(std::string aModelName) const
    {
        if (!mParamList.isSublist("Material Models"))
        {
            ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
        }
        else
        {
            auto tModelsParamList = mParamList.get<Teuchos::ParameterList>("Material Models");
            if (!tModelsParamList.isSublist(aModelName))
            {
                std::stringstream tSS;
                tSS << "Requested a material model ('" << aModelName << "') that isn't defined";
                ANALYZE_THROWERR(tSS.str());
            }

            auto tModelParamList = tModelsParamList.sublist(aModelName);
            if(tModelParamList.isSublist("Isotropic Linear Elastic"))
            {
                return std::make_shared<MaterialElastic<SpatialDim>>(tModelParamList.sublist("Isotropic Linear Elastic"));
            }
            else
            {
                auto tErrMsg = this->getErrorMsg();
                ANALYZE_THROWERR(tErrMsg);
            }
        }
    }

private:
    /*!< map from input force type string to supported enum */
    std::vector<std::string> mSupportedMaterials =
        {"isotropic linear elastic"};

    std::string
    getErrorMsg()
    const
    {
        std::string tMsg = std::string("ERROR: Requested material constitutive model is not supported. ")
            + "Supported material constitutive models for mechanical analysis are: ";
        for(const auto& tElement : mSupportedMaterials)
        {
            tMsg = tMsg + "'" + tElement + "', ";
        }
        auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
        return tSubMsg;
    }
};
// class ElasticModelFactory

/******************************************************************************/
/*! Infinitesimal strain functor.

 Given a gradient matrix and displacement array, compute the strain.
 strain tensor in Voigt notation = {e_xx, e_yy, e_zz, e_yz, e_xz, e_xy}

 */
/******************************************************************************/
template<typename EvaluationType>
class ComputeStrainTensor
{
private:
    // set local element type
    using ElementType = typename EvaluationType::ElementType;

    // set local static parameters
    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell; /*!< number of nodes per element */

public:
    template
    <typename StrainScalarType,
     typename GradientScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()
    (const Plato::OrdinalType                                                  & aCellOrdinal,
     const Plato::Matrix<mNumNodesPerCell,mNumSpatialDims, GradientScalarType> & aGradient,
           Plato::Matrix<mNumSpatialDims,mNumSpatialDims, StrainScalarType>    & aVirtualStrains) const
    {
        constexpr Plato::Scalar tDeltaState = 1.0;
        constexpr Plato::Scalar tOneOverTwo = 0.5;
        for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
        {
            for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
            {
                for(Plato::OrdinalType tNodeI = 0; tNodeI < mNumNodesPerCell; tNodeI++)
                {
                    aVirtualStrains(tDimI,tDimJ) += tOneOverTwo *
                        ( ( tDeltaState * aGradient(tNodeI, tDimI) )
                        + ( tDeltaState * aGradient(tNodeI, tDimJ) ) );
                }
            }
        }
    }

    template
    <typename StrainScalarType,
     typename DispScalarType,
     typename GradientScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()
    (const Plato::OrdinalType                                                    & aCellOrdinal,
      const Plato::ScalarMultiVectorT<DispScalarType>                            & aState,
      const Plato::Matrix<mNumNodesPerCell,mNumSpatialDims, GradientScalarType> & aGradient,
            Plato::Matrix<mNumSpatialDims,mNumSpatialDims, StrainScalarType>    & aStrains) const
    {
        constexpr Plato::Scalar tOneOverTwo = 0.5;
        for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
        {
            for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
            {
                for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
                {
                    auto tLocalOrdinalI = tNodeIndex * mNumSpatialDims + tDimI;
                    auto tLocalOrdinalJ = tNodeIndex * mNumSpatialDims + tDimJ;
                    aStrains(tDimI,tDimJ) += tOneOverTwo *
                            ( aState(aCellOrdinal, tLocalOrdinalJ) * aGradient(tNodeIndex, tDimI)
                            + aState(aCellOrdinal, tLocalOrdinalI) * aGradient(tNodeIndex, tDimJ));
                }
            }
        }
    }
};
// class StrainTensor

template< typename EvaluationType>
class ComputeStressTensor
{
private:
    // set local element type
    using ElementType = typename EvaluationType::ElementType;

    // set local static parameters
    static constexpr auto mNumSpatialDims   = ElementType::mNumSpatialDims;
    static constexpr auto mNumNodesPerCell  = ElementType::mNumNodesPerCell; /*!< number of nodes per element */

    // set local fad types
    using StateScalarType  = typename EvaluationType::StateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigScalarType = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultScalarType = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */
    using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>; /*!< strain variables automatic differentiation type */

    // set local member data
    Plato::Scalar mMu = 1.0;
    Plato::Scalar mLambda = 1.0;
public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model interface
    **********************************************************************************/
    ComputeStressTensor(const Plato::MaterialModel<mNumSpatialDims> & aMaterialModel)
    {
        mMu = aMaterialModel.getScalarConstant("mu");
        if(mMu <= std::numeric_limits<Plato::Scalar>::epsilon())
        {
            ANALYZE_THROWERR(std::string("Error: Lame constant 'mu' is less than the machine epsilon. ")
                + "The input material properties were not parsed properly.");
        }
        mLambda = aMaterialModel.getScalarConstant("lambda");
        if(mLambda <= std::numeric_limits<Plato::Scalar>::epsilon())
        {
            ANALYZE_THROWERR(std::string("Error: Lame constant 'lambda' is less than the machine epsilon. ")
                + "The input material properties were not parsed properly.");
        }
    }

    template<typename ResultScalarType,
             typename StrainScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()
    (const Plato::Matrix<mNumSpatialDims, mNumSpatialDims, StrainScalarType> & aStrainTensor,
           Plato::Matrix<mNumSpatialDims, mNumSpatialDims, ResultScalarType> & aStressTensor) const
    {
        // compute first strain invariant
        StrainScalarType tFirstStrainInvariant(0.0);
        for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
        {
            tFirstStrainInvariant += aStrainTensor(tDim,tDim);
        }

        // add contribution from first stress invariant to the stress tensor
        for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
        {
            aStressTensor(tDim,tDim) += mLambda * tFirstStrainInvariant;
        }

        // add shear stress contribution to the stress tensor
        for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
        {
            for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
            {
                aStressTensor(tDimI,tDimJ) += 2.0 * mMu * aStrainTensor(tDimI,tDimJ);
            }
        }
    }
};
// class StressTensor

template< typename EvaluationType>
class ComputeStressDivergence
{
private:
    // set local element type
    using ElementType = typename EvaluationType::ElementType;

    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
    static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;

public:
    template<typename ResultScalarType,
             typename StressScalarType,
           typename GradientScalarType,
             typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        const Plato::OrdinalType & aCellOrdinal,
        const Plato::ScalarMultiVectorT<ResultScalarType> & aResult,
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims, StressScalarType> & aStressTensor,
        const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType> & aGradient,
        const VolumeScalarType & aCellVolume,
        const Plato::Scalar aMultiplier = 1.0) const
    {

        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumSpatialDims + tDimI;
                for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
                {
                    ResultScalarType tVal =
                        aMultiplier * aCellVolume * aStressTensor(tDimI,tDimJ) * aGradient(tNodeIndex, tDimJ);
                    Kokkos::atomic_add(&aResult(aCellOrdinal, tLocalOrdinal),tVal);
                }
            }
        }
    }
};

template< typename EvaluationType>
class ComputeSideStressTensors
{
private:
    // set local element type
    using ElementType = typename EvaluationType::ElementType;

    // set local static parameters
    static constexpr auto mNumSpatialDims   = ElementType::mNumSpatialDims;
    static constexpr auto mNumNodesPerCell  = ElementType::mNumNodesPerCell; /*!< number of nodes per element */

    // set local fad types
    using StateScalarType  = typename EvaluationType::StateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigScalarType = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultScalarType = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */
    using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>; /*!< strain variables automatic differentiation type */

    // set local member data
    Plato::Scalar mMu = 1.0;
    Plato::Scalar mLambda = 1.0;

    const Plato::MaterialModel<mNumSpatialDims>& mMaterialModel;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model interface
    **********************************************************************************/
    ComputeSideStressTensors(const Plato::MaterialModel<mNumSpatialDims> & aMaterialModel) :
        mMaterialModel(aMaterialModel)
    {
    }

    void operator()
    (const Plato::WorkSets                                 & aWorkSets,
     const Plato::OrdinalVectorT<const Plato::OrdinalType> & aSideCellOrdinals,
     const Plato::Scalar                                   & aCycle,
           Plato::ScalarArray3DT<ResultScalarType>         & aStressTensors,
           Plato::ScalarArray3DT<ResultScalarType>         & aVirtualStressTensors)
    {
        // create local worksets
        Plato::OrdinalType tNumSideCells = aSideCellOrdinals.size();
        Plato::ScalarVectorT<ConfigScalarType> tCellVolume("volume",tNumSideCells);

        // create local functors
        ComputeStressTensor<EvaluationType>     tComputeStressTensor(mMaterialModel);
        ComputeStrainTensor<EvaluationType>     tComputeStrainTensor;
        Plato::ComputeGradientMatrix<ElementType> tComputeGradient;

        // get input worksets (i.e., domain for function evaluate)
        auto tStateWS   = Plato::metadata<Plato::ScalarMultiVectorT<StateScalarType>>( aWorkSets.get("states"));
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigScalarType>>( aWorkSets.get("configuration"));

        // get element integration points and weights
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for("compute elastostatic residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumSideCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal, const Plato::OrdinalType aGpOrdinal)
        {
            // get integration point
            auto tCubPoint = tCubPoints(aGpOrdinal);

            // get cell ordinal
            auto tCellOrdinal = aSideCellOrdinals(aSideOrdinal);

            // compute cell gradients and volume at this integration point
            ConfigScalarType tVolume(0.0);
            Plato::Matrix<mNumNodesPerCell,mNumSpatialDims, ConfigScalarType> tGradient;
            tComputeGradient(tCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);

            // compute strains and stresses at this integration point
            Plato::Matrix<mNumSpatialDims,mNumSpatialDims, StrainScalarType>  tStrainTensor(0.0);
            Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ResultScalarType>  tStressTensor(0.0);
            tComputeStrainTensor(tCellOrdinal,tStateWS,tGradient,tStrainTensor);
            tComputeStressTensor(tStrainTensor,tStressTensor);

            // compute virtual strains and stresses at this integration point
            Plato::Matrix<mNumSpatialDims,mNumSpatialDims, StrainScalarType>  tVirtualStrainTensor(0.0);
            Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ResultScalarType>  tVirtualStressTensor(0.0);
            tComputeStrainTensor(tCellOrdinal,tGradient,tVirtualStrainTensor);
            tComputeStressTensor(tVirtualStrainTensor,tVirtualStressTensor);

            // add contribution to volume from this integration point
            tVolume *= tCubWeights(aGpOrdinal);

            // compute cell stress and strain: aggregate stress and strain contribution from each integration point
            for(Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
            {
                for(Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++)
                {
                    Kokkos::atomic_add(&aStressTensors(aSideOrdinal,tDimI,tDimJ), tVolume*tStressTensor(tDimI,tDimJ));
                    Kokkos::atomic_add(&aVirtualStressTensors(aSideOrdinal,tDimI,tDimJ), tVolume*tVirtualStressTensor(tDimI,tDimJ));
                }
            }
            // compute cell volume: aggregate volume contribution from each integration
            Kokkos::atomic_add(&tCellVolume(aSideOrdinal), tVolume);
        });

        // compute cell stress and strain tensors by multiplying by 1/volume factor
        Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumSideCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal)
        {
            for(Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
            {
                for(Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++)
                {
                    aStressTensors(aSideOrdinal,tDimI,tDimJ)        /= tCellVolume(aSideOrdinal);
                    aVirtualStressTensors(aSideOrdinal,tDimI,tDimJ) /= tCellVolume(aSideOrdinal);
                }
            }
        });
    }
};

/******************************************************************************/
/*! Stress functor.

 given a strain, compute the stress.
 stress tensor in Voigt notation = {s_xx, s_yy, s_zz, s_yz, s_xz, s_xy}
 */
/******************************************************************************/
template< typename EvaluationType>
class CauchyStress
{
private:
    // set local element type
    using ElementType = typename EvaluationType::ElementType;

    // set local static parameters
    static constexpr auto mNumVoigtTerms  = ElementType::mNumVoigtTerms; /*!< number of stress/strain terms */
    static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;

    // set local array types
    Plato::Array<mNumVoigtTerms> mReferenceStrain; /*!< reference strain tensor */
    const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness; /*!< material stiffness matrix */

    // set local fad types
    using StateFadType  = typename EvaluationType::StateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigFadType = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultFadType = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */
    using StrainFadType = typename Plato::fad_type_t<ElementType, StateFadType, ConfigFadType>; /*!< strain variables automatic differentiation type */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model interface
    **********************************************************************************/
    CauchyStress(const Plato::LinearElasticMaterial<mNumSpatialDims> & aMaterial) :
        mCellStiffness  (aMaterial.getStiffnessMatrix()),
        mReferenceStrain(aMaterial.getReferenceStrain())
    {}

    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
    **********************************************************************************/
    KOKKOS_INLINE_FUNCTION void
    operator()
    (      Plato::Array<mNumVoigtTerms, ResultFadType> & aCauchyStress,
     const Plato::Array<mNumVoigtTerms, StrainFadType> & aSmallStrain) const
    {
        // Method used to compute the stress and called from within a
        // Kokkos parallel_for.
        for(Plato::OrdinalType tVoigtIndex_I = 0; tVoigtIndex_I < mNumVoigtTerms; tVoigtIndex_I++)
        {
            aCauchyStress(tVoigtIndex_I) = 0.0;

            for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
            {
                aCauchyStress(tVoigtIndex_I) +=
                  (aSmallStrain(tVoigtIndex_J) -
                   this->mReferenceStrain(tVoigtIndex_J)) *
                    this->mCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
            }
        }
    }
};
// class LinearStress


enum class volume_force_t
{
    BODYLOAD,
    UNDEFINED
};

template<typename EvaluationType>
class VolumeForceBase
{
public:
    virtual ~VolumeForceBase(){}
    virtual volume_force_t type() const = 0;
    virtual void evaluate
    (const Plato::SpatialDomain & aSpatialDomain,
     const Plato::WorkSets      & aWorkSets,
     const Plato::Scalar        & aScale,
     const Plato::Scalar        & aCycle) = 0;

    std::map<volume_force_t,std::string> SupportedForces =
        { {volume_force_t::BODYLOAD,"Body Load"} };
};

/******************************************************************************/
/*!
 \brief Class for essential boundary conditions.
 */
template<typename EvaluationType>
class BodyForce : public VolumeForceBase<EvaluationType>
/******************************************************************************/
{
private:
    // set local element type definition
    using ElementType = typename EvaluationType::ElementType;

    // set local fad type definitions
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;

    // set local static types
    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims; /*!< spatial dimensions */
    static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell; /*!< number of nodes per cell/element */

    const std::string mName;
    const std::string mFunction;
    const Plato::OrdinalType mDof;

public:
    BodyForce<EvaluationType>(const std::string &aName, Teuchos::ParameterList &aProbParam) :
            mName(aName),
            mDof(aProbParam.get<Plato::OrdinalType>("Index", 0)),
            mFunction(aProbParam.get<std::string>("Function"))
    {
    }

    volume_force_t type() const { return volume_force_t::BODYLOAD; }

    void evaluate
    (const Plato::SpatialDomain & aSpatialDomain,
     const Plato::WorkSets      & aWorkSets,
     const Plato::Scalar        & aScale,
     const Plato::Scalar        & aCycle)
    {
        // get input worksets (i.e., domain for function evaluate)
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
        auto tResultWS  = Plato::metadata<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
        auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlScalarType>>(aWorkSets.get("controls"));

        // get integration points and weights
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        // map points to physical space
        Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
        Plato::ScalarArray3DT<ConfigScalarType> tPhysicalPoints("cub points physical space", tNumCells, tNumPoints, mNumSpatialDims);
        Plato::mapPoints<ElementType>(tConfigWS, tPhysicalPoints);

        // get integrand values at quadrature points
        Plato::ScalarMultiVectorT<ConfigScalarType> tFxnValues("function values", tNumCells*tNumPoints, 1);
        Plato::getFunctionValues<mNumSpatialDims>(tPhysicalPoints, mFunction, tFxnValues);

        // integrate and assemble
        //
        auto tDof = mDof;
        Plato::VectorEntryOrdinal<mNumSpatialDims, mNumSpatialDims> tVectorEntryOrdinal(aSpatialDomain.Mesh);
        Kokkos::parallel_for("compute body load", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tDetJ = Plato::determinant(ElementType::jacobian(tCubPoint, tConfigWS, iCellOrdinal));

            ControlScalarType tErsatzMaterial(0.0);
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < mNumNodesPerCell; tFieldOrdinal++)
            {
                tErsatzMaterial += tBasisValues(tFieldOrdinal)*tControlWS(iCellOrdinal, tFieldOrdinal);
            }

            auto tEntryOffset = iCellOrdinal * tNumPoints;

            auto tFxnValue = tFxnValues(tEntryOffset + iGpOrdinal, 0);
            auto tWeight = aScale * tCubWeights(iGpOrdinal) * tDetJ;
            for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < mNumNodesPerCell; tFieldOrdinal++)
            {
                ResultScalarType tValue = tWeight * tFxnValue * tBasisValues(tFieldOrdinal) * tErsatzMaterial;
                Kokkos::atomic_add(&tResultWS(iCellOrdinal,tFieldOrdinal*mNumDofsPerNode+tDof), tValue);
            }
        });
    }
};
// end class BodyForce

/******************************************************************************/
/*!
 \brief Contains list of BodyForce objects.
 */
template<typename EvaluationType>
class VolumeForces
/******************************************************************************/
{
private:
    std::unordered_map<volume_force_t,std::vector< std::shared_ptr< VolumeForceBase<EvaluationType> > > > mVolumeForces;

public:

    /******************************************************************************//**
     * \brief Constructor that parses and creates a vector of BodyForce objects based on
     *   the ParameterList.
     * \param aParams Body Loads sublist with input parameters
    **********************************************************************************/
    VolumeForces(Teuchos::ParameterList& aParams) :
        mVolumeForces()
    {
        this->initialize(aParams);
    }

    /**************************************************************************/
    /*!
     \brief Add the body load to the result workset
     */
    void evaluate
    (const Plato::SpatialDomain & aSpatialDomain,
     const Plato::WorkSets      & aWorkSets,
     const Plato::Scalar        & aScale,
     const Plato::Scalar        & aCycle)
    {
        for(const auto & tPair : mVolumeForces)
        {
            for(const auto & tForce : tPair.second)
            {
                tForce->evaluate(aSpatialDomain, aWorkSets, aScale, aCycle);
            }
        }
    }

private:
    void initialize(Teuchos::ParameterList& aParams)
    {
        for(Teuchos::ParameterList::ConstIterator tIndex = aParams.begin(); tIndex != aParams.end(); ++tIndex)
        {
            const Teuchos::ParameterEntry & tEntry = aParams.entry(tIndex);
            const std::string &tName = aParams.name(tIndex);

            if(!tEntry.isList())
            {
                ANALYZE_THROWERR("Parameter in Body Loads block not valid.  Expect lists only.");
            }

            Teuchos::ParameterList& tSublist = aParams.sublist(tName);
            auto tVolumeForce = std::make_shared<BodyForce<EvaluationType>>(tName, tSublist);
            volume_force_t tType = tVolumeForce->type();
            mVolumeForces[tType].push_back(tVolumeForce);
        }
    }
};

enum class boundary_condition_t
{
    FLUX,
    PRESSURE,
    UNDEFINED
};

template<typename EvaluationType>
class NaturalBoundaryConditionBase
{
protected:
    using ElementType = typename EvaluationType::ElementType;

    static constexpr auto mNumDofsPerNode = ElementType::mNumDofsPerNode;

    // allocate local member data
    const std::string mName;        /*!< boundary condition sublist name */
    const std::string mSideSetName; /*!< entity set name */

    // allocate local member instances
    Plato::Array<mNumDofsPerNode> mCoefficients; /*!< natural boundary condition coefficients */
    std::shared_ptr<Plato::MathExpr> mCoefficientsExpr[mNumDofsPerNode];

public:
    NaturalBoundaryConditionBase
    (const std::string            & aLoadName,
           Teuchos::ParameterList & aSubList) :
        mName(aLoadName),
        mSideSetName(aSubList.get<std::string>("Sides")),
        mCoefficientsExpr{nullptr}
    {
        this->setCoefficients(aSubList);
    }
    virtual ~NaturalBoundaryConditionBase(){}

    std::string name() const { return mName; }
    std::string sideset() const { return mSideSetName; }
    Plato::Array<mNumDofsPerNode> coefficients() const { return mCoefficients; }

    virtual boundary_condition_t type() const = 0;

    virtual void evaluate
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets     & aWorkSets,
     const Plato::Scalar       & aScale,
     const Plato::Scalar       & aCycle) = 0;

protected:
    void evaluateExpression(const Plato::Scalar & aCycle)
    {
        for(int tDof=0; tDof<mNumDofsPerNode; tDof++)
        {
            if(mCoefficientsExpr[tDof])
            {
                mCoefficients(tDof) = mCoefficientsExpr[tDof]->value(aCycle);
            }
        }
    }

private:
    void setCoefficients(Teuchos::ParameterList & aSubList)
    {
        auto tIsValue = aSubList.isType<Teuchos::Array<Plato::Scalar>>("Vector");
        auto tIsExpr  = aSubList.isType<Teuchos::Array<std::string>>("Vector");
        if (tIsValue)
        {
            auto tForceVal = aSubList.get<Teuchos::Array<Plato::Scalar>>("Vector");
            for(Plato::OrdinalType tDof=0; tDof<mNumDofsPerNode; tDof++)
            {
                mCoefficients(tDof) = tForceVal[tDof];
            }
        }
        else
        if (tIsExpr)
        {
            auto tExpression = aSubList.get<Teuchos::Array<std::string>>("Vector");
            for(Plato::OrdinalType tDof=0; tDof<mNumDofsPerNode; tDof++)
            {
                mCoefficientsExpr[tDof] = std::make_shared<Plato::MathExpr>(tExpression[tDof]);
                mCoefficients(tDof) = mCoefficientsExpr[tDof]->value(0.0);
            }
        }
    }
};


template<typename EvaluationType>
class NaturalBoundaryConditionPressure : public NaturalBoundaryConditionBase<EvaluationType>
{
private:
    // set local element type definition
    using ElementType = typename EvaluationType::ElementType;

    // set local base class type
    using BaseClassType = NaturalBoundaryConditionBase<EvaluationType>;

    // set natural boundary condition base class member data
    using BaseClassType::mCoefficients;
    using BaseClassType::mSideSetName;

    // set local fad type definitions
    using ResultScalarType = typename EvaluationType::ResultScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;

public:
    NaturalBoundaryConditionPressure
    (const std::string            & aLoadName,
           Teuchos::ParameterList & aSubList) :
        BaseClassType(aLoadName,aSubList)
    {}
    ~NaturalBoundaryConditionPressure()
    {}

    boundary_condition_t type() const
    {
        return boundary_condition_t::PRESSURE;
    }

    void evaluate
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets     & aWorkSets,
     const Plato::Scalar       & aScale,
     const Plato::Scalar       & aCycle)
    {
        // get side set connectivity information
        auto tElementOrds = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
        auto tNodeOrds    = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);

        // local functor - calculate normal vector
        Plato::WeightedNormalVector<ElementType> tWeightedNormalVector;

        // get integration point and weights
        auto tCoefficients = mCoefficients;
        auto tCubatureWeights = ElementType::Face::getCubWeights();
        auto tCubaturePoints  = ElementType::Face::getCubPoints();
        auto tNumPoints = tCubatureWeights.size();

        // get input worksets (i.e., domain for function evaluate)
        auto tResultWS = Plato::metadata<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
        auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));

        // pressure forces should act towards the surface; thus, -1.0 is used to invert the outward facing normal inwards.
        Plato::Scalar tNormalMultiplier(-1.0);
        Plato::OrdinalType tNumFaces = tElementOrds.size();
        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
        {
            auto tElementOrdinal = tElementOrds(aSideOrdinal);

            Plato::Array<ElementType::mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
            for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<ElementType::mNumNodesPerFace; tNodeOrd++)
            {
                tLocalNodeOrds(tNodeOrd) = tNodeOrds(aSideOrdinal*ElementType::mNumNodesPerFace+tNodeOrd);
            }

            auto tCubatureWeight = tCubatureWeights(aPointOrdinal);
            auto tCubaturePoint = tCubaturePoints(aPointOrdinal);
            auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);
            auto tBasisGrads  = ElementType::Face::basisGrads(tCubaturePoint);

            // compute area weighted normal vector
            Plato::Array<ElementType::mNumSpatialDims, ConfigScalarType> tWeightedNormalVec;
            tWeightedNormalVector(tElementOrdinal, tLocalNodeOrds, tBasisGrads, tConfigWS, tWeightedNormalVec);

            // project into aResult workset
            for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
            {
                for( Plato::OrdinalType tDof=0; tDof<ElementType::mNumDofsPerNode; tDof++)
                {
                    auto tElementDofOrdinal = (tLocalNodeOrds[tNode] * ElementType::mNumDofsPerNode) + tDof;
                    ResultScalarType tVal =
                      tWeightedNormalVec(tDof) * tCoefficients(tDof) * aScale * tCubatureWeight * tNormalMultiplier * tBasisValues(tNode);
                    Kokkos::atomic_add(&tResultWS(tElementOrdinal, tElementDofOrdinal), tVal);
                }
            }
        }, "pressure force");
    }
};

template<typename EvaluationType>
class NaturalBoundaryConditionFlux : public NaturalBoundaryConditionBase<EvaluationType>
{
private:
    // set local element type definition
    using ElementType = typename EvaluationType::ElementType;

    // set local parent class type
    using BaseClassType = NaturalBoundaryConditionBase<EvaluationType>;

    // set local fad type definitions
    using ResultScalarType = typename EvaluationType::ResultScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;

public:
    NaturalBoundaryConditionFlux
    (const std::string            & aLoadName,
           Teuchos::ParameterList & aSubList) :
        BaseClassType(aLoadName,aSubList)
    {}
    ~NaturalBoundaryConditionFlux(){}

    boundary_condition_t type() const
    {
        return boundary_condition_t::FLUX;
    }

    void evaluate
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets     & aWorkSets,
     const Plato::Scalar       & aScale,
     const Plato::Scalar       & aCycle)
    {
        // evaluate expression if defined
        this->evaluateExpression(aCycle);

        // get input worksets (i.e., domain for function evaluate)
        auto tResultWS = Plato::metadata<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
        auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));

        // get side set connectivity
        auto tElementOrds = aSpatialModel.Mesh->GetSideSetElements(BaseClassType::mSideSetName);
        auto tNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(BaseClassType::mSideSetName);
        Plato::OrdinalType tNumFaces = tElementOrds.size();

        // create surface area functor
        Plato::SurfaceArea<ElementType> tComputeSurfaceArea;

        // get integration points and weights
        auto tCoefficients = BaseClassType::mCoefficients;
        auto tCubatureWeights = ElementType::Face::getCubWeights();
        auto tCubaturePoints  = ElementType::Face::getCubPoints();
        auto tNumPoints = tCubatureWeights.size();

        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
        {
          auto tElementOrdinal = tElementOrds(aSideOrdinal);

          Plato::Array<ElementType::mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
          for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<ElementType::mNumNodesPerFace; tNodeOrd++)
          {
              tLocalNodeOrds(tNodeOrd) = tNodeOrds(aSideOrdinal*ElementType::mNumNodesPerFace+tNodeOrd);
          }

          auto tCubatureWeight = tCubatureWeights(aPointOrdinal);
          auto tCubaturePoint = tCubaturePoints(aPointOrdinal);
          auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);
          auto tBasisGrads  = ElementType::Face::basisGrads(tCubaturePoint);

          ResultScalarType tSurfaceArea(0.0);
          tComputeSurfaceArea(tElementOrdinal, tLocalNodeOrds, tBasisGrads, tConfigWS, tSurfaceArea);
          tSurfaceArea *= aScale;
          tSurfaceArea *= tCubatureWeight;

          for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
          {
              for( Plato::OrdinalType tDof=0; tDof<ElementType::mNumDofsPerNode; tDof++)
              {
                  auto tElementDofOrdinal = ( tLocalNodeOrds[tNode] * ElementType::mNumDofsPerNode ) + tDof;
                  ResultScalarType tResult = tBasisValues(tNode)*tCoefficients[tDof]*tSurfaceArea;
                  Kokkos::atomic_add(&tResultWS(tElementOrdinal,tElementDofOrdinal), tResult);
              }
          }
        }, "flux force");
    }
};


enum class nitsche_t
{
    DISPLACEMENTS,
    TEMPERATURE,
    UNDEFINED
};

template<typename EvaluationType>
class NitscheBase
{
protected:
    using ElementType = typename EvaluationType::ElementType;

    static constexpr auto mNumDofsPerNode = ElementType::mNumDofsPerNode;

    // allocate member data
    Plato::Scalar mNitschePenalty = 1.0; /*!< penalty parameter for the Nitsche method */

    const std::string mName;        /*!< user defined essential boundary condition sublist name */
    const std::string mSideSetName; /*!< entity set name */
    const std::string mMaterialModelName;  /*!< name assigned to the material model used on this boundary */

public:
    NitscheBase
    (const std::string            & aName,
           Teuchos::ParameterList & aSubList) :
        mName(aName),
        mSideSetName(aSubList.get<std::string>("Sides")),
        mMaterialModelName(aSubList.get<std::string>("Material Model"))
    {
        // parse penalty parameter
        if(aSubList.isType<Plato::Scalar>("Penalty")){
            mNitschePenalty = aSubList.get<Plato::Scalar>("Penalty");
        }
    }
    virtual ~NitscheBase(){}

    std::string name() const { return mName; }
    std::string sideset() const { return mSideSetName; }

    virtual nitsche_t type() const = 0;

    virtual void initialize(Teuchos::ParameterList & aProbParams) = 0;

    virtual void evaluate
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets     & aWorkSets,
     const Plato::Scalar       & aScale,
     const Plato::Scalar       & aCycle) = 0;
};

template<typename EvaluationType>
class ComputeSideCellVolumes
{
private:
    // set local element types
    using ElementType = typename EvaluationType::ElementType;

    // set local fad type definitions
    using ResultScalarType = typename EvaluationType::ResultScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;

    // set local member data
    std::string mSideSetName;

public:
    ComputeSideCellVolumes(const std::string& aEntitySetName) :
        mSideSetName(aEntitySetName)
    {}

    void operator()
    (const Plato::SpatialModel                  & aSpatialModel,
     const Plato::WorkSets                      & aWorkSets,
         Plato::ScalarVectorT<ConfigScalarType> & aSideCellVolumes)
    {
        // get side set connectivity information
        auto tSideCellOrdinals = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
        Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();

        // get input workset
        auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));

        // get body integration points and weights
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        // compute volume of each cell in the entity set
        Kokkos::parallel_for("compute volume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumSideCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal, const Plato::OrdinalType aGpOrdinal)
        {
            auto tCubPoint  = tCubPoints(aGpOrdinal);
            auto tCubWeight = tCubWeights(aGpOrdinal);

            auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
            auto tJacobian = ElementType::jacobian(tCubPoint, tConfigWS, tCellOrdinal);
            ConfigScalarType tVolume = Plato::determinant(tJacobian);
            tVolume *= tCubWeight;

            Kokkos::atomic_add(&aSideCellVolumes(aSideOrdinal), tVolume);
        });
    }
};

template<typename EvaluationType>
class ComputeSideCellFaceAreas
{
private:
    // set local element types
    using BodyElementType = typename EvaluationType::ElementType;
    using FaceElementType = typename BodyElementType::Face;

    // set local constexpr members
    static constexpr auto mNumNodesPerFace = BodyElementType::mNumNodesPerFace;

    // set local fad type definitions
    using ResultScalarType = typename EvaluationType::ResultScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;

    // set local member data
    std::string mSideSetName;

public:
    ComputeSideCellFaceAreas(const std::string& aEntitySetName) :
        mSideSetName(aEntitySetName)
    {}

    void operator()
    (const Plato::SpatialModel                  & aSpatialModel,
     const Plato::WorkSets                      & aWorkSets,
         Plato::ScalarVectorT<ConfigScalarType> & aSideCellFaceAreas)
    {
        // get side set connectivity information
        auto tSideCellOrdinals = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
        auto tLocalNodeOrds    = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
        Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();

        // get input workset
        auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));

        // create surface area functor
        Plato::SurfaceArea<BodyElementType> tComputeFaceArea;

        // get face integration points and weights
        auto tFaceCubPoints    = FaceElementType::getCubPoints();
        auto tFaceCubWeights   = FaceElementType::getCubWeights();
        auto tFaceNumCubPoints = tFaceCubWeights.size();

        // compute characteristic length of each cell in the entity set
        Kokkos::parallel_for("compute volume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumSideCells, tFaceNumCubPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal, const Plato::OrdinalType aGpOrdinal)
        {
            // get integration point and weight
            auto tFaceCubPoint   = tFaceCubPoints(aGpOrdinal);
            auto tFaceCubWeight  = tFaceCubWeights(aGpOrdinal);
            auto tFaceBasisGrads = FaceElementType::basisGrads(tFaceCubPoint);

            // get local node ordinal on boundary entity
            Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrdinals;
            for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++)
            {
                tFaceLocalNodeOrdinals(tIndex) = tLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
            }

            // compute entity area
            ConfigScalarType tFaceArea(0.0);
            auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
            tComputeFaceArea(tCellOrdinal,tFaceLocalNodeOrdinals,tFaceBasisGrads,tConfigWS,tFaceArea);
            tFaceArea *= tFaceCubWeight;

            // add characteristic length contribution from this integration point, i.e. Gauss point
            Kokkos::atomic_add(&aSideCellFaceAreas(aSideOrdinal), tFaceArea);
        });
    }
};

template<typename EvaluationType>
class ComputeCharacteristicLength
{
private:
    // set local fad type definitions
    using ResultScalarType = typename EvaluationType::ResultScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;

    // set local member data
    std::string mSideSetName;

public:
    ComputeCharacteristicLength(const std::string& aEntitySetName) :
        mSideSetName(aEntitySetName)
    {}

    void operator()
    (const Plato::SpatialModel                    & aSpatialModel,
     const Plato::WorkSets                        & aWorkSets,
           Plato::ScalarVectorT<ConfigScalarType> & aCharLength)
    {
        // get side set connectivity information
        auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
        auto tLocalNodeOrds     = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
        Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();

        // compute volumes of cells in side
        Plato::ScalarVectorT<ConfigScalarType> tSideCellVolumes("volume",tNumSideCells);
        ComputeSideCellVolumes<EvaluationType> tComputeSideCellVolumes(mSideSetName);
        tComputeSideCellVolumes(aSpatialModel,aWorkSets,tSideCellVolumes);

        // compute face areas of cells in side
        Plato::ScalarVectorT<ConfigScalarType> tSideCellFaceAreas("area",tNumSideCells);
        ComputeSideCellFaceAreas<EvaluationType> tComputeSideCellFaceAreas(mSideSetName);
        tComputeSideCellFaceAreas(aSpatialModel,aWorkSets,tSideCellFaceAreas);

        // compute characteristic length of each cell in the entity set
        Kokkos::parallel_for("compute characteristic length", Kokkos::RangePolicy<>(0, tNumSideCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal)
        {
            aCharLength(aSideOrdinal) = tSideCellVolumes(aSideOrdinal) / tSideCellFaceAreas(aSideOrdinal);
        });
    }
};

/***************************************************************************//**
 *
 * \tparam EvaluationType scalar types are set based on the evaluation type
 *
 * \class NitscheLinearFormMechanics
 *
 * \brief This class is responsible for evaluating the linear form derived by
 *          enforcing the Dirichlet boundary conditions in linear mechanics
 *          problem via Nitsche's method. The resulting linear form is given by:
 *
 *          \f$
 *              - \int_{\Gamma_D}\delta\mathbf{u}\cdot\left(\sigma\cdot\mathbf{n}_{\Gamma}\right)d\Gamma
 *              + \int_{\Gamma_D}\delta\left(\sigma\mathbf{n}_{\Gamma}\right)\cdot\left(\mathbf{u}-\mathbf{u}_{D}\right)d\Gamma
 *              + \int_{\Gamma_D}\gamma_{N}^{\mathbf{u}}\delta\mathbf{u}\cdot\left(\mathbf{u}-\mathbf{u}_{D}\right)d\Gamma
 *          \f$
 *
 *          where a non-symmetric Nitsche formulation is considered, see for example
 *          Burman (2012) and Schillinger et al. (2016a), and \f$\mathbf{u}_D\f$ is the
 *          displacement imposed on the Dirichlet boundary \f$\Gamma_D\f$. The
 *          parameter \f$\gamma_{N}^{\mathbf{u}}\f$ is chosen to achieve a desired
 *          accuracy in satisfying the boundary conditions.
 *
*******************************************************************************/
template<typename EvaluationType>
class NitscheLinearMechanics : public NitscheBase<EvaluationType>
{
private:
    // set local element type definition
    using BodyElementBase = typename EvaluationType::ElementType;
    using FaceElementBase = typename BodyElementBase::Face;

    // set local constexpr members
    static constexpr auto mNumSpatialDims  = BodyElementBase::mNumSpatialDims;
    static constexpr auto mNumDofsPerNode  = BodyElementBase::mNumDofsPerNode;
    static constexpr auto mNumNodesPerCell  = BodyElementBase::mNumNodesPerCell;
    static constexpr auto mNumNodesPerFace = BodyElementBase::mNumNodesPerFace;
    static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;

    // set local base class type
    using BaseClassType = NitscheBase<EvaluationType>;

    // set class type names for functors
    using ProjectFromNodes =
        Plato::InterpolateFromNodal<FaceElementBase,mNumDofsPerNode,/*offset=*/0,mNumDofsPerNode>;

    // set natural boundary condition base class member data
    using BaseClassType::mSideSetName;
    using BaseClassType::mNitschePenalty;
    using BaseClassType::mMaterialModelName;

    // set local fad type definitions
    using StateScalarType  = typename EvaluationType::StateScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;

    /*!< strain variables automatic differentiation type */
    using StrainScalarType = typename Plato::fad_type_t<BodyElementBase,StateScalarType,ConfigScalarType>;

    // set member data
    std::shared_ptr<Plato::MaterialModel<mNumSpatialDims>> mMaterial;

public:
    NitscheLinearMechanics
    (const std::string            & aName,
           Teuchos::ParameterList & aSubList) :
        BaseClassType(aName,aSubList)
    {}
    ~NitscheLinearMechanics()
    {}

    nitsche_t type() const
    {
        return nitsche_t::DISPLACEMENTS;
    }

    void initialize(Teuchos::ParameterList & aProbParams)
    {
        // create material model and get stiffness
        FactoryElasticMaterial<mNumSpatialDims> tMaterialFactory(aProbParams);
        mMaterial = tMaterialFactory.create(mMaterialModelName);
    }

    void evaluate
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets     & aWorkSets,
     const Plato::Scalar       & aMultiplier,
     const Plato::Scalar       & aCycle)
    {
        // get input & output worksets (i.e., domain & range for function evaluate)
        auto tStateWS     = Plato::metadata<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
        auto tResultWS    = Plato::metadata<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
        auto tConfigWS    = Plato::metadata<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
        auto tDirichletWS = Plato::metadata<Plato::ScalarMultiVector>(aWorkSets.get("dirichlet"));

        // get side set connectivity information
        auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
        auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
        auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
        Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();

        // compute characteristic length
        ComputeCharacteristicLength<EvaluationType> tComputeCharacteristicLength(mSideSetName);
        Plato::ScalarVectorT<ConfigScalarType> tCharacteristicLength("characteristic length",tNumCellsOnSideSet);
        tComputeCharacteristicLength(aSpatialModel, aWorkSets, tCharacteristicLength);

        // compute trial and test stresses
        ComputeStrainTensor<EvaluationType> tComputeStrainTensor;
        ComputeStressTensor<EvaluationType> tComputeStressTensor(mMaterial.operator*());
        Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;

        // create surface area functor
        Plato::SurfaceArea<BodyElementBase> tComputeFaceArea;
        // create calculate weighted (by the area) normal vector functor
        Plato::WeightedNormalVector<BodyElementBase> tComputeNormalVector;
        // create interpolate from nodal functor
        ProjectFromNodes tProjectFromNodes;

        // get integration points and weights
        auto tCubPointsInFaceParentElem = FaceElementBase::getCubPoints();
        auto tCubPointsOnBodyParentElemSurfaces = BodyElementBase::getFaceCubPoints();

        auto tCubWeightsOnBodyParentElemSurface = BodyElementBase::getFaceCubWeights();

        auto tYoungsModulus  = mMaterial->getScalarConstant("youngs modulus");
        auto tNitschePenaltyTimesModulus = mNitschePenalty * tYoungsModulus;
        Kokkos::parallel_for("nitsche bcs", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
          {tNumCellsOnSideSet, mNumGaussPointsPerFace}),
          KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
         {
            auto tCubPointInFaceParentElem = tCubPointsInFaceParentElem(aPointOrdinal);
            auto tBasisGradsInFaceParentElem = FaceElementBase::basisGrads(tCubPointInFaceParentElem);

            // quadrature data to evaluate integral on the body surface of interest
            Plato::Array<mNumSpatialDims> tCubPointOnBodyParentElemSurface;
            Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
            auto tCubPointsOnBodyParentElemSurface = tCubPointsOnBodyParentElemSurfaces(tLocalFaceOrdinal);
            for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
                Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
                tCubPointOnBodyParentElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
            }
            auto tCubWeightOnBodyParentElemSurface = tCubWeightsOnBodyParentElemSurface(aPointOrdinal);
            auto tBasisGradsOnBodyParentElemSurface  = BodyElementBase::basisGrads(tCubPointOnBodyParentElemSurface);
            auto tBasisValuesOnBodyParentElemSurface = BodyElementBase::basisValues(tCubPointOnBodyParentElemSurface);

            Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
            for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++)
            {
                tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
            }

            // TERM 1
            //

            // compute normal vector weighted by the entity area
            auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
            Plato::Array<mNumSpatialDims, ConfigScalarType> tNormalVector;
            tComputeNormalVector(tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsInFaceParentElem,tConfigWS,tNormalVector);

            // compute entity area
            ConfigScalarType tFaceArea(0.0);
            tComputeFaceArea(tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsInFaceParentElem,tConfigWS,tFaceArea);

            // compute strains and stresses for this integration point
            ConfigScalarType tVolume(0.0);
            Plato::Matrix<mNumNodesPerCell,mNumSpatialDims, ConfigScalarType> tGradient;
            tComputeGradient(tCellOrdinal,tCubPointOnBodyParentElemSurface,tConfigWS,tGradient,tVolume);
            Plato::Matrix<mNumSpatialDims,mNumSpatialDims, StrainScalarType>  tStrainTensor(0.0);
            tComputeStrainTensor(tCellOrdinal,tStateWS, tGradient, tStrainTensor);
            Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ResultScalarType>  tStressTensor(0.0);
            tComputeStressTensor(tStrainTensor,tStressTensor);

            // term 1: int_{\Gamma_D} \delta{u}\cdot(\sigma\cdot{n}) d\Gamma_D
            for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++)
            {
                for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
                {
                    auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
                    for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++)
                    {
                        ResultScalarType tValue = -aMultiplier * tBasisValuesOnBodyParentElemSurface(tNode)
                            * ( tStressTensor(tDimI,tDimJ) * tNormalVector[tDimJ] )
                            * tCubWeightOnBodyParentElemSurface * tFaceArea;
                        Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
                    }
                }
            }

            // TERM 2
            //

            // interpolate state from nodes
            Plato::Array<mNumSpatialDims,StateScalarType> tProjectedStates;
            for(Plato::OrdinalType tDof = 0; tDof < mNumDofsPerNode; tDof++)
            {
                tProjectedStates(tDof) = 0.0;
                for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
                {
                    Plato::OrdinalType tCellDofIndex = (mNumDofsPerNode * tNodeIndex) + tDof;
                    tProjectedStates(tDof) += tStateWS(tCellOrdinal, tCellDofIndex)
                            * tBasisValuesOnBodyParentElemSurface(tNodeIndex);
                }
            }

            Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ConfigScalarType>  tVirtualStrainTensor(0.0);
            tComputeStrainTensor(tCellOrdinal,tGradient,tVirtualStrainTensor);
            Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ConfigScalarType>  tVirtualStressTensor(0.0);
            tComputeStressTensor(tVirtualStrainTensor,tVirtualStressTensor);

            // term 2: int_{\Gamma_D} \delta(\sigma\cdot{n})\cdot(u - u_D) d\Gamma_D
            for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++)
            {
                for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
                {
                    auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
                    for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++)
                    {
                        ResultScalarType tValue = aMultiplier * tCubWeightOnBodyParentElemSurface
                            * tFaceArea * ( tVirtualStressTensor(tDimI,tDimJ) * tNormalVector[tDimJ] )
                            * ( tProjectedStates[tDimI] - tDirichletWS(tCellOrdinal,tLocalDofOrdinal) );
                        Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
                    }
                }
            }

            // TERM 3
            //

            // term 3: int_{\Gamma_D}\gamma_N^u \delta{u}\cdot(u - u_D) d\Gamma_D
            ConfigScalarType tGamma = tNitschePenaltyTimesModulus / tCharacteristicLength(aSideOrdinal);
            for(Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++)
            {
                for(Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
                {
                    auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
                    ResultScalarType tValue = aMultiplier * tGamma * tBasisValuesOnBodyParentElemSurface(tNode)
                        * ( tProjectedStates[tDimI] - tDirichletWS(tCellOrdinal,tLocalDofOrdinal) ) * tFaceArea
                        * tCubWeightOnBodyParentElemSurface;
                    Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
                }
            }
         });
    }
};

template<typename EvaluationType>
class FactoryNitscheBC
{
private:
    /*!< map from input Nitsche boundary condition type string to supported enum */
    std::map<std::string,nitsche_t> mSupportedNitscheBCs =
        {
            {"displacements",nitsche_t::DISPLACEMENTS},
            {"temperature"  ,nitsche_t::TEMPERATURE}
        };

public:
    FactoryNitscheBC(){}
    ~FactoryNitscheBC(){}

    std::shared_ptr<NitscheBase<EvaluationType>>
    create
    (const std::string            & aName,
           Teuchos::ParameterList & aParams)
    {
        auto tStringVariableType = aParams.get<std::string>("Variable");
        auto tVariableType = this->type(tStringVariableType);
        switch(tVariableType)
        {
            case nitsche_t::DISPLACEMENTS:
            { return std::make_shared<NitscheLinearMechanics<EvaluationType>>(aName, aParams); }
            case nitsche_t::TEMPERATURE:
            default:
            {
                return {nullptr};
            }
        }
    }

private:
    nitsche_t type(const std::string& aVariable) const
    {
        auto tVariable = Plato::tolower(aVariable);
        auto tItr = mSupportedNitscheBCs.find(tVariable);
        if( tItr == mSupportedNitscheBCs.end() ){
            std::string tMsg = std::string("Nitsche's method cannot be applied to variable '")
                + tVariable + "'. " + "Supported variables are: ";
            for(const auto& tPair : mSupportedNitscheBCs)
            {
                tMsg = tMsg + tPair.first + ", ";
            }
            auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
            ANALYZE_THROWERR(tSubMsg)
        }
        return (tItr->second);
    }
};

template<typename EvaluationType>
class NitscheBCs
{
// private member data
private:
    // set class type names for functors
    using NitscheBoundaryConditions = std::vector< std::shared_ptr< NitscheBase< EvaluationType> > >;

    /*!< list of essential boundary conditions (EBCs) enforced using Nitsche's method */
    std::unordered_map<nitsche_t,NitscheBoundaryConditions> mNitscheBCs;

// public member functions
public:
    NitscheBCs
    (const std::string&            aNameSublist,
           Teuchos::ParameterList& aProbParams) :
        mNitscheBCs()
    {
        auto tNitscheSubLists = aProbParams.sublist(aNameSublist);
        this->parse(tNitscheSubLists);
        this->initialize(aProbParams);
    }

    NitscheBoundaryConditions&
    get(const nitsche_t& aType) const
    {
        auto tItr = mNitscheBCs.find(aType);
        if( tItr == mNitscheBCs.end() ){
            ANALYZE_THROWERR("ERROR: Did not find requested Nitsche boundary condition")
        }
        return tItr->second;
    }

    void evaluate
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets     & aWorkSets,
     const Plato::Scalar       & aMultiplier,
     const Plato::Scalar       & aCycle)
    {
        for (const auto &tPair : mNitscheBCs)
        {
            for(const auto &tNitscheBC : tPair.second)
            {
                tNitscheBC->evaluate(aSpatialModel, aWorkSets, aMultiplier, aCycle);
            }
        }
    }

// private member functions
private:
    void parse
    (Teuchos::ParameterList& aNitscheSublists)
    {
        FactoryNitscheBC<EvaluationType> tFactory;
        auto tNameNitscheSublists = aNitscheSublists.name();
        for (Teuchos::ParameterList::ConstIterator tItr = aNitscheSublists.begin(); tItr != aNitscheSublists.end(); ++tItr)
        {
            const Teuchos::ParameterEntry &tEntry = aNitscheSublists.entry(tItr);
            if (!tEntry.isList())
            {
                ANALYZE_THROWERR(std::string("ERROR: ") + tNameNitscheSublists + " block is not valid. "
                                 + "Constructor expects Parameter Lists only")
            }

            const std::string &tName = aNitscheSublists.name(tItr);
            if(aNitscheSublists.isSublist(tName) == false)
            {
                ANALYZE_THROWERR(std::string("ERROR: Parameter sublist: '") + tName.c_str() + "' is NOT defined")
            }

            Teuchos::ParameterList &tMyNitscheSublist = aNitscheSublists.sublist(tName);
            std::shared_ptr<NitscheBase<EvaluationType>> tBC = tFactory.create(tName, tMyNitscheSublist);
             auto tVariableType = tBC->type();
            mNitscheBCs[tVariableType].push_back(tBC);
        }
    }

    void initialize(Teuchos::ParameterList & aProbParams)
    {
        for (const auto &tPair : mNitscheBCs)
        {
            for(const auto &tNitscheBC : tPair.second)
            {
                tNitscheBC->initialize(aProbParams);
            }
        }
    }
};


template<typename EvaluationType>
class FactoryNaturalBC
{
private:
    // set local element type definition
    using ElementType = typename EvaluationType::ElementType;

    // set local static types
    static constexpr auto mNumDofsPerNode = ElementType::mNumDofsPerNode;

    /*!< map from input force type string to supported enum */
    std::map<std::string,boundary_condition_t> mSupportedBCs =
        {
            {"uniform",boundary_condition_t::FLUX},
            {"pressure",boundary_condition_t::PRESSURE}
        };

public:
    FactoryNaturalBC(){}
    ~FactoryNaturalBC(){}

    std::shared_ptr<NaturalBoundaryConditionBase<EvaluationType>>
    create(const std::string & aName, Teuchos::ParameterList &aSubList)
    {
        this->parseCoefficients(aName,aSubList);
        auto tType = this->type(aSubList);
        switch(tType)
        {
            case boundary_condition_t::FLUX:
            {
                return std::make_shared<NaturalBoundaryConditionFlux<EvaluationType>>(aName, aSubList);
            }
            case boundary_condition_t::PRESSURE:
            {
                return std::make_shared<NaturalBoundaryConditionPressure<EvaluationType>>(aName, aSubList);
            }
            default:
            {
                return {nullptr};
            }
        }
    }

private:
    boundary_condition_t type(Teuchos::ParameterList & aSubList)
    {
        std::string tType = aSubList.get<std::string>("Type");
        tType = Plato::tolower(tType);
        auto tItr = mSupportedBCs.find(tType);
        if( tItr == mSupportedBCs.end() ){
            std::string tMsg = std::string("Natural Boundary Condition of type '")
                + tType + "' is not supported. " + "Supported options are: ";
            for(const auto& tPair : mSupportedBCs)
            {
                tMsg = tMsg + tPair.first + ", ";
            }
            auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
            ANALYZE_THROWERR(tSubMsg)
        }
        return (tItr->second);
    }

    void parseCoefficients(const std::string & aName, Teuchos::ParameterList &aSubList)
    {
        bool tValueNBC  = ( aSubList.isType<Plato::Scalar>("Value") ||
                            aSubList.isType<std::string>("Value") );
        bool tValuesNBC = ( aSubList.isType<Teuchos::Array<Plato::Scalar>>("Values") ||
                            aSubList.isType<Teuchos::Array<std::string>>("Values") );

        if (tValuesNBC && tValueNBC)
        {
            auto tErrMsg = std::string("ERROR: The 'Values' and 'Value' keyword cannot be defined simultaneously ")
                + "in parameter list '" + aName.c_str() + "'. Only one of the two options must be selected.";
            ANALYZE_THROWERR(tErrMsg)
        }

        if(tValueNBC)
        {
            this->parseValue(aSubList);
        } else
        if(tValuesNBC)
        {
            this->parseValues(aSubList);
        }
        else
        {
            auto tErrorMsg = std::string("ERROR: A natural boundary condition was requested but no coefficient ")
                + "values were defined. Check input parameter list '" + aName.c_str() + "' definition.";
            ANALYZE_THROWERR(tErrorMsg)
        }
    }

    void parseValue(Teuchos::ParameterList &aSubList)
    {
        if(aSubList.isType<Plato::Scalar>("Value"))
        {
            auto tValue = aSubList.get<Plato::Scalar>("Value");
            Teuchos::Array<Plato::Scalar> tForceVector(mNumDofsPerNode, tValue);
            aSubList.set("Vector", tForceVector);
        } else
        if(aSubList.isType<std::string>("Value"))
        {
            auto tValue = aSubList.get<std::string>("Value");
            Teuchos::Array<std::string> tForceVector(mNumDofsPerNode, tValue);
            aSubList.set("Vector", tForceVector);
        } else
        {
            std::string tErrMsg = std::string("ERROR: Unexpected type encountered for 'Value' Parameter Keyword. ")
                 + "Specify 'type' of 'double' or 'string'.";
            ANALYZE_THROWERR(tErrMsg)
        }
    }

    void parseValues(Teuchos::ParameterList &aSubList)
    {
        if(aSubList.isType<Teuchos::Array<Plato::Scalar>>("Values"))
        {
            auto tValues = aSubList.get<Teuchos::Array<Plato::Scalar>>("Values");
            aSubList.set("Vector", tValues);
        } else
        if(aSubList.isType<Teuchos::Array<std::string>>("Values"))
        {
            auto tValues = aSubList.get<Teuchos::Array<std::string>>("Values");
            aSubList.set("Vector", tValues);
        } else
        {
            auto tErrMsg = std::string("ERROR: Unexpected type encountered for 'Values' parameter keyword. ")
                 + "Specify 'type' of 'Array(double)' or 'Array(string)'.";
            ANALYZE_THROWERR(tErrMsg)
        }
    }
};

template<typename EvaluationType>
class NaturalBCs
{
// private member data
private:
    FactoryNaturalBC<EvaluationType> mFactory;
    /*!< list of natural boundary condition */
    std::unordered_map<boundary_condition_t,std::vector<std::shared_ptr<NaturalBoundaryConditionBase<EvaluationType> > > > mNaturalBCs;

// public member functions
public:
    NaturalBCs(Teuchos::ParameterList &aParams) :
        mNaturalBCs()
    {
        this->parse(aParams);
    }

    void evaluate
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets     & aWorkSets,
     const Plato::Scalar       & aScale,
     const Plato::Scalar       & aCycle)
    {
        for (const auto &tPair : mNaturalBCs)
        {
            for(const auto &tNaturalBC : tPair.second)
            {
                tNaturalBC->evaluate(aSpatialModel, aWorkSets, aScale, aCycle);
            }
        }
    }

// private member functions
private:
    void parse(Teuchos::ParameterList &aParams)
    {
        for (Teuchos::ParameterList::ConstIterator tItr = aParams.begin(); tItr != aParams.end(); ++tItr)
        {
            const Teuchos::ParameterEntry &tEntry = aParams.entry(tItr);
            if (!tEntry.isList())
            {
                ANALYZE_THROWERR("Natural Boundary Condition block is not valid. Constructor expects Parameter Lists only")
            }

            const std::string &tName = aParams.name(tItr);
            if(aParams.isSublist(tName) == false)
            {
                std::stringstream tMsg;
                tMsg << "Natural Boundary Condition Sublist: '" << tName.c_str() << "' is NOT defined";
                ANALYZE_THROWERR(tMsg.str().c_str())
            }
            Teuchos::ParameterList &tSubList = aParams.sublist(tName);

            if(tSubList.isParameter("Type") == false)
            {
                std::stringstream tMsg;
                tMsg << "Natural Boundary Condition 'Type' keyword is not defined in "
                     << "Natural Boundary Condition Parameter Sublist with name '"
                     << tName.c_str() << "'";
                ANALYZE_THROWERR(tMsg.str().c_str())
            }
            std::shared_ptr<NaturalBoundaryConditionBase<EvaluationType>> tBC = mFactory.create(tName, tSubList);
            boundary_condition_t tType = tBC->type();
            mNaturalBCs[tType].push_back(tBC);
        }
    }
};


class ResidualBase
{
protected:
    const Plato::SpatialDomain     & mSpatialDomain;  /*!< Plato spatial model containing mesh, meshsets, etc */
          Plato::DataMap           & mDataMap;        /*!< Plato Analyze database */
          std::vector<std::string>   mDofNames;       /*!< state dof names */

public:
    explicit ResidualBase
    (const Plato::SpatialDomain & aSpatialDomain,
           Plato::DataMap       & aDataMap) :
        mSpatialDomain(aSpatialDomain),
        mDataMap(aDataMap)
    {}
    virtual ~ResidualBase()
    {}

    decltype(mSpatialDomain.Mesh) getMesh() const
    {
        return (mSpatialDomain.Mesh);
    }

    const decltype(mDofNames)& getDofNames() const
    {
        return mDofNames;
    }

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate residual within the domain, exclude boundary terms.
     * \param [in] aWorkSets holds state and control worksets
     ******************************************************************************/
    virtual void evaluate
    (const Plato::WorkSets & aWorkSets,
     const Plato::Scalar   & aCycle) = 0;

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate residual on domain boundaries.
     * \param [in]  aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in]  aWorkSets     holds input worksets (e.g. states, control, etc)
     ******************************************************************************/
    virtual void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets     & aWorkSets,
     const Plato::Scalar       & aCycle) = 0;
};
// class abstract residual


template<typename EvaluationType, typename PenaltyFunction>
class ApplyWeighting
{
private:
    // set local element type
    using ElementType = typename EvaluationType::ElementType;

    // set local static types
    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;

    // set local fad types
    using ControlScalarType = typename EvaluationType::ControlScalarType;

    PenaltyFunction mPenaltyFunction; /*!< penalty model used for topology optimization - density discretization */

public:
    /******************************************************************************//**
     * \brief Default Constructor
     * \param [in] aPenaltyFunction penalty function interface
    **********************************************************************************/
    ApplyWeighting(PenaltyFunction aPenaltyFunction) :
        mPenaltyFunction(aPenaltyFunction)
    {
    }

    template<typename InputScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        const Plato::OrdinalType                                               & aCellOrdinal,
        const Plato::ScalarMultiVectorT<ControlScalarType>                     & aControl,
        const Plato::Array<mNumNodesPerCell>                                   & aBasisValues,
              Plato::Matrix<mNumSpatialDims, mNumSpatialDims, InputScalarType> & aInputOutput
    ) const
    {
        // apply weighting
        //
        ControlScalarType tCellDensity = 0.0;
        for (Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
        {
            tCellDensity += aControl(aCellOrdinal, tNode)*aBasisValues(tNode);
        }
        for (Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
        {
            for (Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
            {
                aInputOutput(tDimI,tDimJ) *= mPenaltyFunction(tCellDensity);
            }
        }
    }
};

template<typename EvaluationType>
class ResidualElastostatics : public ResidualBase
{
private:
    // set local element type
    using ElementType = typename EvaluationType::ElementType;

    // set local static types
    static constexpr auto mNumVoigtTerms   = ElementType::mNumVoigtTerms;
    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
    static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;

    // set local fad types
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;

    // set strain fad type
    using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

    std::shared_ptr<NaturalBCs<EvaluationType>> mNaturalBCs;
    std::shared_ptr<NitscheBCs<EvaluationType>> mNitscheBCs;
    std::shared_ptr<VolumeForces<EvaluationType>> mVolumeForces;
    std::shared_ptr<Plato::MaterialModel<mNumSpatialDims>> mMaterial;

    Plato::MSIMP mPenaltyFunction;
    ApplyWeighting<EvaluationType, Plato::MSIMP> mApplyWeighting;

public:
    ResidualElastostatics
    (const std::string          & aTypePDE,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aProbParams) :
         ResidualBase(aDomain, aDataMap),
         mPenaltyFunction(aProbParams.sublist(aTypePDE).sublist("Penalty Function")),
         mApplyWeighting(mPenaltyFunction)
    {
        // obligatory: define dof names in order
        mDofNames.push_back("displacement X");
        if(mNumSpatialDims > 1) mDofNames.push_back("displacement Y");
        if(mNumSpatialDims > 2) mDofNames.push_back("displacement Z");

        // create material model
        FactoryElasticMaterial<mNumSpatialDims> tMaterialFactory(aProbParams);
        mMaterial = tMaterialFactory.create(aDomain.getMaterialName());

        // initialize boundary condition and load functors
        this->initialize(aProbParams);
    }

    void evaluate
    (const Plato::WorkSets & aWorkSets,
     const Plato::Scalar   & aCycle)
    {
        // create local worksets
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVectorT<ConfigScalarType> tCellVolume("volume",tNumCells);
        Plato::ScalarArray3DT<StrainScalarType> tCellStrain("strain",tNumCells,mNumSpatialDims,mNumSpatialDims);
        Plato::ScalarArray3DT<ResultScalarType> tCellStress("stress",tNumCells,mNumSpatialDims,mNumSpatialDims);

        // create local functors
        ComputeStressTensor<EvaluationType>     tComputeStressTensor(mMaterial.operator*());
        ComputeStrainTensor<EvaluationType>     tComputeStrainTensor;
        ComputeStressDivergence<EvaluationType> tComputeStressDivergence;
        Plato::ComputeGradientMatrix<ElementType> tComputeGradient;

        // get input worksets (i.e., domain for function evaluate)
        auto tStateWS   = Plato::metadata<Plato::ScalarMultiVectorT<StateScalarType>>( aWorkSets.get("states"));
        auto tResultWS  = Plato::metadata<Plato::ScalarMultiVectorT<ResultScalarType>>( aWorkSets.get("result"));
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigScalarType>>( aWorkSets.get("configuration"));
        auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlScalarType>>( aWorkSets.get("controls"));

        // get element integration points and weights
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        auto& tApplyWeighting = mApplyWeighting;
        Kokkos::parallel_for("compute elastostatic residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigScalarType tVolume(0.0);

            // create local containers for stress, strains, and gradients
            Plato::Matrix<mNumNodesPerCell,mNumSpatialDims, ConfigScalarType> tGradient;
            Plato::Matrix<mNumSpatialDims,mNumSpatialDims, StrainScalarType>  tStrainTensor(0.0);
            Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ResultScalarType>  tStressTensor(0.0);

            // get integration point
            auto tCubPoint = tCubPoints(iGpOrdinal);

            // compute strains and stresses for this integration point
            tComputeGradient(iCellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
            tComputeStrainTensor(iCellOrdinal,tStateWS, tGradient, tStrainTensor);
            tComputeStressTensor(tStrainTensor,tStressTensor);

            // add contribution to volume from this integration point
            tVolume *= tCubWeights(iGpOrdinal);

            // apply ersatz penalization
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            tApplyWeighting(iCellOrdinal, tControlWS, tBasisValues, tStressTensor);

            // apply divergence to stress
            tComputeStressDivergence(iCellOrdinal, tResultWS, tStressTensor, tGradient, tVolume);

            // compute cell stress and strain: aggregate stress and strain contribution from each integration point
            for(Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
            {
                for(Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++)
                {
                    Kokkos::atomic_add(&tCellStrain(iCellOrdinal,tDimI,tDimJ), tVolume*tStrainTensor(tDimI,tDimJ));
                    Kokkos::atomic_add(&tCellStress(iCellOrdinal,tDimI,tDimJ), tVolume*tStressTensor(tDimI,tDimJ));
                }
            }
            // compute cell volume: aggregate volume contribution from each integration
            Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
        });

        // compute cell stress and strain tensors by multiplying by 1/volume factor
        Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
        {
            for(Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
            {
                for(Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++)
                {
                    tCellStrain(iCellOrdinal,tDimI,tDimJ) /= tCellVolume(iCellOrdinal);
                    tCellStress(iCellOrdinal,tDimI,tDimJ) /= tCellVolume(iCellOrdinal);
                }
            }
        });

        // add contributions from external volume forces
        if( mVolumeForces != nullptr )
        {
            mVolumeForces->evaluate(mSpatialDomain,aWorkSets,/*scale=*/-1.0,aCycle);
        }
    }

    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets     & aWorkSets,
     const Plato::Scalar       & aCycle)
    {
        // add contributions from natural boundary conditions
        if( mNaturalBCs != nullptr )
        {
            mNaturalBCs->evaluate(aSpatialModel,aWorkSets,/*scale=*/-1.0,aCycle);
        }

        // add contributions from nitsche boundary conditions
        if( mNitscheBCs != nullptr )
        {
            mNitscheBCs->evaluate(aSpatialModel,aWorkSets,/*scale=*/1.0,aCycle);
        }
    }

private:
    void initialize(Teuchos::ParameterList & aProbParams)
    {
        // parse body loads
        if(aProbParams.isSublist("Body Loads"))
        {
            mVolumeForces = std::make_shared<VolumeForces<EvaluationType>>
                    (aProbParams.sublist("Body Loads"));
        }

        // parse natural boundary conditions
        if(aProbParams.isSublist("Natural Boundary Conditions"))
        {
            mNaturalBCs = std::make_shared<NaturalBCs<EvaluationType>>
                    (aProbParams.sublist("Natural Boundary Conditions"));
        }

        // if essential boundary conditions are enforced weakly, allocate nitsche residual
        if(aProbParams.isSublist("Nitsche Boundary Conditions"))
        {
            mNitscheBCs = std::make_shared<NitscheBCs<EvaluationType>>
                    ("Nitsche Boundary Conditions", aProbParams);
        }
    }
};



template<int SpatialDim>
class MaterialThermalConduction : public MaterialModel<SpatialDim>
{
  public:
    MaterialThermalConduction(const Teuchos::ParameterList& paramList)
    {
        this->parseTensor("Thermal Conductivity", paramList);
    }
};

template<Plato::OrdinalType SpatialDim>
class FactoryThermalMaterial
{
private:
    const Teuchos::ParameterList& mParamList;

public:
    FactoryThermalMaterial(const Teuchos::ParameterList& aParamList) :
            mParamList(aParamList)
    {
    }

    std::shared_ptr<Plato::MaterialModel<SpatialDim>>
    create
    (std::string aModelName) const
    {
        if(!mParamList.isSublist("Material Models"))
        {
            ANALYZE_THROWERR("ERROR: 'Material Models' parameter list was not found! Returning 'nullptr'");
        }
        else
        {
            auto tModelsParamList = mParamList.get < Teuchos::ParameterList > ("Material Models");

            if(!tModelsParamList.isSublist(aModelName))
            {
                auto tErrMsg = std::string("Requested a material model ('") + aModelName + "') that isn't defined";
                ANALYZE_THROWERR(tErrMsg);
            }

            auto tModelParamList = tModelsParamList.sublist(aModelName);
            if(tModelParamList.isSublist("Thermal Conduction"))
            {
                return std::make_shared<MaterialThermalConduction<SpatialDim>>(tModelParamList.sublist("Thermal Conduction"));
            }
            else
            {
                auto tErrMsg = this->getErrorMsg();
                ANALYZE_THROWERR(tErrMsg);
            }
        }
    }
    std::vector<std::string> mSupportedMaterials =
        {"thermal conduction"};

    std::string
    getErrorMsg()
    const
    {
        std::string tMsg = std::string("ERROR: Requested material constitutive model is not supported. ")
            + "Supported material constitutive models for thermal analysis are: ";
        for(const auto& tElement : mSupportedMaterials)
        {
            tMsg = tMsg + "'" + tElement + "', ";
        }
        auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
        return tSubMsg;
    }
};

template<typename EvaluationType>
class ResidualThermostatics : public ResidualBase
{
private:
    // set local element type
    using ElementType = typename EvaluationType::ElementType;

    // set local static types
    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
    static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;

    // set local fad types
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;

    using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

    std::shared_ptr<VolumeForces<EvaluationType>> mBodyForces;
    std::shared_ptr<NaturalBCs<EvaluationType>>   mNaturalForces;

    std::shared_ptr<Plato::MaterialModel<mNumSpatialDims>> mMaterial;

    Plato::MSIMP mPenaltyFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, Plato::MSIMP> mApplyWeighting;

public:
    ResidualThermostatics
    (const std::string          & aTypePDE,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aProbParams) :
         ResidualBase(aDomain, aDataMap),
         mPenaltyFunction(aProbParams.sublist(aTypePDE).sublist("Penalty Function")),
         mApplyWeighting(mPenaltyFunction)
    {
        // obligatory: define dof names in order
        mDofNames.push_back("temperature");

        // create material model and get stiffness
        FactoryThermalMaterial<mNumSpatialDims> tMaterialFactory(aProbParams);
        mMaterial = tMaterialFactory.create(aDomain.getMaterialName());

        // parse body loads
        if(aProbParams.isSublist("Body Loads"))
        {
            mBodyForces = std::make_shared<VolumeForces<EvaluationType>>(aProbParams.sublist("Body Loads"));
        }

        // parse natural boundary conditions
        if(aProbParams.isSublist("Natural Boundary Conditions"))
        {
            mNaturalForces = std::make_shared<NaturalBCs<EvaluationType>>(aProbParams.sublist("Natural Boundary Conditions"));
        }
    }

    void evaluate
    (const Plato::WorkSets & aWorkSets,
     const Plato::Scalar   & aCycle)
    {
        // create local worksets
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVectorT<ConfigScalarType>      tCellVolume("volume", tNumCells);
        Plato::ScalarMultiVectorT<GradScalarType>   tCellGrad  ("temperature gradient", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultScalarType> tCellFlux  ("thermal flux", tNumCells, mNumSpatialDims);

        // create local functors
        Plato::ScalarGrad<ElementType>            tScalarGrad;
        Plato::ThermalFlux<ElementType>           tThermalFlux(mMaterial.operator*());
        Plato::GeneralFluxDivergence<ElementType> tFluxDivergence;
        Plato::ComputeGradientMatrix<ElementType> tComputeGradient;

        Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode> tInterpolateFromNodal;

        // get input worksets (i.e., domain for function evaluate)
        auto tStateWS   = Plato::metadata<Plato::ScalarMultiVectorT<StateScalarType>>( aWorkSets.get("states"));
        auto tResultWS  = Plato::metadata<Plato::ScalarMultiVectorT<ResultScalarType>>( aWorkSets.get("result"));
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigScalarType>>( aWorkSets.get("configuration"));
        auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlScalarType>>( aWorkSets.get("controls"));

        // get cubature weights and points
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        auto& tApplyWeighting = mApplyWeighting;
        Kokkos::parallel_for("compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigScalarType tVolume(0.0);

            Plato::Array <mNumSpatialDims,GradScalarType> tGrad(0.0);
            Plato::Array <mNumSpatialDims,ResultScalarType> tFlux(0.0);
            Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;

            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);

            tComputeGradient(iCellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
            tScalarGrad(iCellOrdinal, tGrad, tStateWS, tGradient);
            StateScalarType tTemperature = tInterpolateFromNodal(iCellOrdinal, tBasisValues, tStateWS);
            tThermalFlux(tFlux, tGrad, tTemperature);

            tVolume *= tCubWeights(iGpOrdinal);
            tApplyWeighting(iCellOrdinal, tControlWS, tBasisValues, tFlux);
            tFluxDivergence(iCellOrdinal, tResultWS, tFlux, tGradient, tVolume, -1.0);

            for(Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++)
            {
                Kokkos::atomic_add(&tCellGrad(iCellOrdinal,tDim), tVolume*tGrad(tDim));
                Kokkos::atomic_add(&tCellFlux(iCellOrdinal,tDim), tVolume*tFlux(tDim));
            }
            Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
        });

        Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
        {
            for(Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++)
            {
                tCellGrad(iCellOrdinal,tDim) /= tCellVolume(iCellOrdinal);
                tCellFlux(iCellOrdinal,tDim) /= tCellVolume(iCellOrdinal);
            }
        });

        if( mBodyForces != nullptr )
        {
            mBodyForces->evaluate( mSpatialDomain,aWorkSets,/*multiplier=*/-1.0,aCycle );
        }
    }


    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets     & aWorkSets,
     const Plato::Scalar       & aCycle)
    {
        if( mNaturalForces != nullptr )
        {
            mNaturalForces->evaluate(aSpatialModel,aWorkSets,/*multiplier=*/-1.0,aCycle);
        }
    }
};


class CriterionBase
{
protected:
    const Plato::SpatialDomain & mSpatialDomain; /*!< spatial domain */
          Plato::DataMap       & mDataMap;       /*!< analyze data map */
    const std::string            mName;          /*!< criterion name */
          bool                   mCompute;       /*!< if true, include in evaluation */

public:
    /******************************************************************************//**
     * \brief CriterionBase constructor
     * \param [in] aSpatialDomain spatial domain
     * \param [in] aDataMap       data map
     * \param [in] aInputs        problem input, used to set up active domains
     * \param [in] aName          criterion name
    **********************************************************************************/
    CriterionBase
    (const std::string            & aName,
     const Plato::SpatialDomain   & aDomain,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aProbParams) :
        mSpatialDomain (aDomain),
        mDataMap       (aDataMap),
        mName          (aName),
        mCompute       (true)
    {
        this->initialize(aProbParams);
    }

    /******************************************************************************//**
     * \brief CriterionBase destructor
    **********************************************************************************/
    virtual ~CriterionBase(){}

    virtual bool isLinear() const = 0;

    virtual void
    evaluateConditional(const Plato::WorkSets & aWorksets,
                        const Plato::Scalar   & aCycle) = 0;

    virtual void
    evaluate(const Plato::WorkSets & aWorksets,
             const Plato::Scalar   & aCycle)
    { if(mCompute) this->evaluateConditional(aWorksets, aCycle); }


    /******************************************************************************//**
     * \brief Return criterion name
     * \return name
    **********************************************************************************/
    const decltype(mName)& getName()
    { return mName; }

    /******************************************************************************//**
     * \brief Get abstract scalar function evaluation and total gradient
    **********************************************************************************/
    virtual void postEvaluate(Plato::ScalarVector, Plato::Scalar)
    { return; }

    /******************************************************************************//**
     * \brief Get abstract scalar function evaluation
     * \param [out] aOutput scalar function evaluation
    **********************************************************************************/
    virtual void postEvaluate(Plato::Scalar& aOutput)
    { return; }

    /******************************************************************************//**
     * \brief Set spatial weight function
     * \param [in] aInput math expression
    **********************************************************************************/
    virtual void setSpatialWeightFunction(std::string aWeightFunctionString)
    { return; }

private:
    void
    initialize
    (Teuchos::ParameterList & aProbParams)
    {
        std::string tCurrentDomainName = mSpatialDomain.getDomainName();

        auto tMyCriteria = aProbParams.sublist("Criteria").sublist(mName);
        std::vector<std::string> tDomains = Plato::teuchos::parse_array<std::string>("Domains", tMyCriteria);
        if(tDomains.size() != 0)
        {
            mCompute = (std::find(tDomains.begin(), tDomains.end(), tCurrentDomainName) != tDomains.end());
            if(!mCompute)
            {
                std::stringstream ss;
                ss << "Block '" << tCurrentDomainName << "' will not be included in the calculation of '" << mName << "'.";
                REPORT(ss.str());
            }
        }
    }
};

template<typename EvaluationType>
class CriterionVolume : public CriterionBase
{
private:
    using ElementType = typename EvaluationType::ElementType;

    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;

    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;

    Plato::MSIMP mPenaltyFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, /*num_weighted_terms=*/ 1, Plato::MSIMP> mApplyWeighting;

public:
    CriterionVolume
    (const std::string            & aFuncName,
     const Plato::SpatialDomain   & aDomain,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aProbParams) :
        CriterionBase(aFuncName, aDomain, aDataMap, aProbParams),
        mPenaltyFunction(aProbParams.sublist("Criteria").sublist(aFuncName).sublist("Penalty Function")),
        mApplyWeighting(mPenaltyFunction)
    { return; }

    bool isLinear() const
    { return true; }

    void
    evaluateConditional
    (const Plato::WorkSets & aWorkSets,
     const Plato::Scalar   & aCycle)
    {
        auto tNumCells = mSpatialDomain.numCells();
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        // get input worksets (i.e., domain for function evaluate)
        auto tResultWS  = Plato::metadata<Plato::ScalarVectorT<ResultScalarType>>( aWorkSets.get("result") );
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigScalarType>>( aWorkSets.get("configuration") );
        auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlScalarType>>( aWorkSets.get("controls") );

        auto& tApplyWeighting  = mApplyWeighting;
        Kokkos::parallel_for("compute volume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint  = tCubPoints(iGpOrdinal);
            auto tCubWeight = tCubWeights(iGpOrdinal);

            auto tJacobian = ElementType::jacobian(tCubPoint, tConfigWS, iCellOrdinal);
            ResultScalarType tCellVolume = Plato::determinant(tJacobian);
            tCellVolume *= tCubWeight;

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            tApplyWeighting(iCellOrdinal, tControlWS, tBasisValues, tCellVolume);

            Kokkos::atomic_add(&tResultWS(iCellOrdinal), tCellVolume);
        });
    }
};


template<Plato::OrdinalType DimI,Plato::OrdinalType DimJ>
class Contraction
{
public:
    template<typename ProductScalarType,
             typename Matrix1ScalarType,
             typename Matrix2ScalarType,
             typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()
    (const Plato::OrdinalType                         & aCellOrdinal,
      const Plato::ScalarVectorT<ProductScalarType>   & aTensorProduct,
      const Plato::Matrix<DimI,DimJ,Matrix1ScalarType> & aMatrix1,
      const Plato::Matrix<DimI,DimJ,Matrix2ScalarType> & aMatrix2,
      const VolumeScalarType                          & aCellVolume,
                Plato::Scalar                           aScale = 1.0
    ) const
    {
      // compute tensor product
      ProductScalarType tInc(0.0);
      for( Plato::OrdinalType tDimI=0; tDimI<DimI; tDimI++)
      {
          for( Plato::OrdinalType tDimJ=0; tDimJ<DimI; tDimJ++)
          {
              tInc += aMatrix1(tDimI,tDimJ)*aMatrix2(tDimI,tDimJ);
          }
      }
      ProductScalarType tProduct = aScale*tInc*aCellVolume;
      Kokkos::atomic_add(&aTensorProduct(aCellOrdinal), tProduct);
    }
};


template<typename EvaluationType>
class CriterionInternalElasticEnergy : public CriterionBase
{
private:
    using ElementType = typename EvaluationType::ElementType;

    static constexpr auto mNumVoigtTerms   = ElementType::mNumVoigtTerms;
    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
    static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;

    // set local fad scalar type
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;

    // set local fad strain scalar type
    using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

    std::shared_ptr<Plato::MaterialModel<mNumSpatialDims>> mMaterial;

    Plato::MSIMP mPenaltyFunction;
    ApplyWeighting<EvaluationType, Plato::MSIMP> mApplyWeighting;

public:
    CriterionInternalElasticEnergy
    (const std::string            & aFuncName,
     const Plato::SpatialDomain   & aDomain,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aProbParams) :
        CriterionBase(aFuncName, aDomain, aDataMap, aProbParams),
        mPenaltyFunction(aProbParams.sublist("Criteria").sublist(aFuncName).sublist("Penalty Function")),
        mApplyWeighting(mPenaltyFunction)
    {
        // create material model and get stiffness
        FactoryElasticMaterial<mNumSpatialDims> tMaterialFactory(aProbParams);
        mMaterial = tMaterialFactory.create(aDomain.getMaterialName());
    }

    bool isLinear() const
    { return false; }

    void
    evaluateConditional
    (const Plato::WorkSets & aWorkSets,
     const Plato::Scalar   & aCycle)
    {
        // get input worksets (i.e., domain for function evaluate)
        auto tStateWS   = Plato::metadata<Plato::ScalarMultiVectorT<StateScalarType>>( aWorkSets.get("states") );
        auto tResultWS  = Plato::metadata<Plato::ScalarVectorT<ResultScalarType>>( aWorkSets.get("result") );
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigScalarType>>( aWorkSets.get("configuration") );
        auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlScalarType>>( aWorkSets.get("controls") );

        ComputeStressTensor<EvaluationType>                 tComputeStress(mMaterial.operator*());
        ComputeStrainTensor<EvaluationType>                 tComputeStrain;
        Plato::ComputeGradientMatrix<ElementType>    tComputeGradient;
        Contraction<mNumSpatialDims,mNumSpatialDims> tTensorContraction;

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        auto& tApplyWeighting = mApplyWeighting;
        auto tNumCells = mSpatialDomain.numCells();
        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigScalarType tVolume(0.0);

            Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tGradient;
            Plato::Matrix<mNumSpatialDims, mNumSpatialDims, StrainScalarType>  tStrainTensor(0.0);
            Plato::Matrix<mNumSpatialDims, mNumSpatialDims, ResultScalarType>  tStressTensor(0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);

            tComputeGradient(iCellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);

            tComputeStrain(iCellOrdinal, tStateWS, tGradient, tStrainTensor);

            tComputeStress(tStrainTensor, tStressTensor);

            tVolume *= tCubWeights(iGpOrdinal);

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            tApplyWeighting(iCellOrdinal, tControlWS, tBasisValues, tStressTensor);

            tTensorContraction(iCellOrdinal, tResultWS, tStressTensor, tStrainTensor, tVolume, 0.5);
        });
    }
};

template<typename EvaluationType>
class CriterionInternalThermalEnergy : public CriterionBase
{
private:
    using ElementType = typename EvaluationType::ElementType;

    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
    static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;

    using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

    Plato::MSIMP mPenaltyFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, Plato::MSIMP> mApplyWeighting;

    std::shared_ptr<Plato::MaterialModel<mNumSpatialDims>> mMaterial;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param aSpatialDomain Plato Analyze spatial domain
     * \param aProblemParams input database for overall problem
     * \param aPenaltyParams input database for penalty function
     **********************************************************************************/
    CriterionInternalThermalEnergy
    (const std::string            & aFuncName,
     const Plato::SpatialDomain   & aDomain,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aProbParams) :
        CriterionBase(aFuncName, aDomain, aDataMap, aProbParams),
        mPenaltyFunction(aProbParams.sublist("Criteria").sublist(aFuncName).sublist("Penalty Function")),
        mApplyWeighting(mPenaltyFunction)
    {
        // create material model and get stiffness
        FactoryThermalMaterial<mNumSpatialDims> tMaterialFactory(aProbParams);
        mMaterial = tMaterialFactory.create(aDomain.getMaterialName());
    }

    bool isLinear() const
    { return false; }

    void
    evaluateConditional
    (const Plato::WorkSets & aWorkSets,
     const Plato::Scalar   & aCycle)
    {
        // get input worksets (i.e., domain for function evaluate)
        auto tStateWS   = Plato::metadata<Plato::ScalarMultiVectorT<StateScalarType>>( aWorkSets.get("states") );
        auto tResultWS  = Plato::metadata<Plato::ScalarVectorT<ResultScalarType>>( aWorkSets.get("result") );
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigScalarType>>( aWorkSets.get("configuration") );
        auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlScalarType>>( aWorkSets.get("controls") );

        Plato::ScalarGrad<ElementType>            tComputeScalarGrad;
        Plato::ThermalFlux<ElementType>           tComputeThermalFlux(mMaterial.operator*());
        Plato::ScalarProduct<mNumSpatialDims>     tComputeScalarProduct;
        Plato::ComputeGradientMatrix<ElementType> tComputeGradient;

        Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode> tInterpolateFromNodal;

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        auto tApplyWeighting  = mApplyWeighting;
        auto tNumCells = mSpatialDomain.numCells();
        Kokkos::parallel_for("thermal energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigScalarType tVolume(0.0);

            Plato::Array <mNumSpatialDims, GradScalarType> tGrad(0.0);
            Plato::Array <mNumSpatialDims, ResultScalarType> tFlux(0.0);
            Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tGradient;

            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);

            tComputeGradient(iCellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);

            tVolume *= tCubWeights(iGpOrdinal);

            // compute temperature gradient
            //
            tComputeScalarGrad(iCellOrdinal, tGrad, tStateWS, tGradient);

            // compute flux
            //
            StateScalarType tTemperature = tInterpolateFromNodal(iCellOrdinal, tBasisValues, tStateWS);
            tComputeThermalFlux(tFlux, tGrad, tTemperature);

            // apply weighting
            //
            tApplyWeighting(iCellOrdinal, tControlWS, tBasisValues, tFlux);

            // compute element internal energy (inner product of tgrad and weighted tflux)
            //
            tComputeScalarProduct(iCellOrdinal, tResultWS, tFlux, tGrad, tVolume, -1.0);
        });
    }
};

/******************************************************************************//**
 * \brief Factory for linear mechanics problem
**********************************************************************************/
struct FactoryMechanicsResidual
{
    /******************************************************************************//**
     * \brief Create a PLATO vector function (i.e. residual equation)
     * \return reference count pointer to a residual base instance
     * \param [in] aPDE           PDE type
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze physics-based database
     * \param [in] aProblemParams input parameters
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<ResidualBase>
    createResidual(
            const std::string          & aTypePDE,
            const Plato::SpatialDomain & aDomain,
              Plato::DataMap           & aDataMap,
              Teuchos::ParameterList   & aProbParams)
    {
        return std::make_shared<ResidualElastostatics<EvaluationType>>(aTypePDE, aDomain, aDataMap, aProbParams);
    }

};

/******************************************************************************//**
 * \brief Factory for mechanics problem
**********************************************************************************/
struct FactoryMechanicsCriterion
{
private:
    /*!< map from input force type string to supported enum */
    std::vector<std::string> mSupportedCriterion =
        {"internal elastic energy","volume"};

    std::string
    getErrorMsg
    (const std::string & aLowerType)
    const
    {
        std::string tMsg = std::string("ERROR: Mechanics criterion of type '")
            + aLowerType + "' is not supported. " + "Supported criteria are: ";
        for(const auto& tElement : mSupportedCriterion)
        {
            tMsg = tMsg + "'" + tElement + "', ";
        }
        auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
        return tSubMsg;
    }

public:
    /******************************************************************************//**
     * \brief Create a criterion function.
     * \return reference count pointer to a criterion base instance
     * \param [in] aPDE           criterion type
     * \param [in] aName          criterion user-defined name
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze physics-based database
     * \param [in] aProblemParams input parameters
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<CriterionBase>
    createCriterion(
        const std::string            & aType,
        const std::string            & aName,
        const Plato::SpatialDomain   & aDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProbParams)
    {
        auto tLowerType = Plato::tolower(aType);
        if( tLowerType == "internal elastic energy" )
        {
            return std::make_shared<CriterionInternalElasticEnergy<EvaluationType>>(aName, aDomain, aDataMap, aProbParams);
        }
        else
        if ( tLowerType == "volume" )
        {
            return std::make_shared<CriterionVolume<EvaluationType>>(aName, aDomain, aDataMap, aProbParams);
        }
        else
        {
            auto tErrMsg = this->getErrorMsg(tLowerType);
            ANALYZE_THROWERR(tErrMsg)
        }
    }

};

template<typename ElementTopoType>
class PhysicsMechanics
{
public:
    typedef FactoryMechanicsResidual  FactoryResidual;
    typedef FactoryMechanicsCriterion FactoryCriterion;

    using ElementType = Plato::MechanicsElement<ElementTopoType>;
};

/******************************************************************************//**
 * \brief Residual factory for thermal problems.
**********************************************************************************/
struct FactoryThermalResidual
{
    /******************************************************************************//**
     * \brief Create a thermal residual function
     * \return reference count pointer to a residual base instance
     * \param [in] aPDE           PDE type
     * \param [in] aSpatialDomain spatial domain database
     * \param [in] aDataMap       output database
     * \param [in] aProblemParams input parameters
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<ResidualBase>
    createResidual
    (const std::string            & aTypePDE,
     const Plato::SpatialDomain   & aDomain,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aProbParams)
    {
        return std::make_shared<ResidualThermostatics<EvaluationType>>(aTypePDE, aDomain, aDataMap, aProbParams);
    }

};

/******************************************************************************//**
 * \brief Criterion factory for thermal problems
**********************************************************************************/
struct FactoryThermalCriterion
{
private:
    /*!< map from input force type string to supported enum */
    std::vector<std::string> mSupportedCriterion =
        {"internal thermal energy","volume"};

    std::string
    getErrorMsg
    (const std::string & aLowerType)
    const
    {
        std::string tMsg = std::string("ERROR: Thermal criterion of type '")
            + aLowerType + "' is not supported. " + "Supported criteria are: ";
        for(const auto& tElement : mSupportedCriterion)
        {
            tMsg = tMsg + "'" + tElement + "', ";
        }
        auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
        return tSubMsg;
    }

public:
    /******************************************************************************//**
     * \brief Create a criterion function.
     * \return reference count pointer to a criterion base instance
     * \param [in] aPDE           criterion type
     * \param [in] aName          criterion user-defined name
     * \param [in] aSpatialDomain spatial domain database
     * \param [in] aDataMap       output database
     * \param [in] aProblemParams input parameters
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<CriterionBase>
    createCriterion
    (const std::string            & aType,
     const std::string            & aName,
     const Plato::SpatialDomain   & aDomain,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aProbParams)
    {
        auto tLowerType = Plato::tolower(aType);
        if( tLowerType == "internal thermal energy" )
        {
            return std::make_shared<CriterionInternalThermalEnergy<EvaluationType>>(aName, aDomain, aDataMap, aProbParams);
        }
        else
        if ( tLowerType == "volume" )
        {
            return std::make_shared<CriterionVolume<EvaluationType>>(aName, aDomain, aDataMap, aProbParams);
        }
        else
        {
            auto tErrMsg = this->getErrorMsg(tLowerType);
            ANALYZE_THROWERR(tErrMsg)
        }
    }

};

template<typename ElementTopoType>
class PhysicsThermal
{
public:
    typedef FactoryThermalResidual  FactoryResidual;
    typedef FactoryThermalCriterion FactoryCriterion;

    using ElementType = Plato::ThermalElement<ElementTopoType>;
};


template<typename EvaluationType>
class WorksetBuilder
{
private:
    // set local types
    using ElementType = typename EvaluationType::ElementType;
    using WorksetFunctionality = Plato::WorksetBase<ElementType>;

    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;

    const Plato::WorksetBase<ElementType> & mWorksetFuncs;

public:
    WorksetBuilder(const Plato::WorksetBase<ElementType>& aWorksetFuncs) :
        mWorksetFuncs(aWorksetFuncs)
    {}

    void build
    (const Plato::SpatialDomain & aDomain,
     const Plato::Database      & aDatabase,
           Plato::WorkSets      & aWorkSets) const
    {
        // number of cells in the spatial domain
        auto tNumCells = aDomain.numCells();

        // build state workset
        using StateScalarType = typename EvaluationType::StateScalarType;
        auto tStateWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<StateScalarType> > >
            ( Plato::ScalarMultiVectorT<StateScalarType>("State Workset", tNumCells, mNumDofsPerCell) );
        mWorksetFuncs.worksetState(aDatabase.vector("states"), tStateWS->mData, aDomain);
        aWorkSets.set("states", tStateWS);

        // build control workset
        using ControlScalarType = typename EvaluationType::ControlScalarType;
        auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlScalarType> > >
            ( Plato::ScalarMultiVectorT<ControlScalarType>("Control Workset", tNumCells, mNumNodesPerCell) );
        mWorksetFuncs.worksetControl(aDatabase.vector("controls"), tControlWS->mData, aDomain);
        aWorkSets.set("controls", tControlWS);

        // build configuration workset
        using ConfigScalarType = typename EvaluationType::ConfigScalarType;
        auto tConfigWS = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigScalarType> > >
            ( Plato::ScalarArray3DT<ConfigScalarType>("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims) );
        mWorksetFuncs.worksetConfig(tConfigWS->mData, aDomain);
        aWorkSets.set("configuration", tConfigWS);
    }

    void build
    (const Plato::OrdinalType   & tNumCells,
     const Plato::Database      & aDatabase,
           Plato::WorkSets      & aWorkSets) const
    {
        // build state workset
        using StateScalarType = typename EvaluationType::StateScalarType;
        auto tStateWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<StateScalarType> > >
            ( Plato::ScalarMultiVectorT<StateScalarType>("State Workset", tNumCells, mNumDofsPerCell) );
        mWorksetFuncs.worksetState(aDatabase.vector("states"), tStateWS->mData);
        aWorkSets.set("states", tStateWS);

        // build control workset
        using ControlScalarType = typename EvaluationType::ControlScalarType;
        auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlScalarType> > >
            ( Plato::ScalarMultiVectorT<ControlScalarType>("Control Workset", tNumCells, mNumNodesPerCell) );
        mWorksetFuncs.worksetControl(aDatabase.vector("controls"), tControlWS->mData);
        aWorkSets.set("controls", tControlWS);

        // build configuration workset
        using ConfigScalarType = typename EvaluationType::ConfigScalarType;
        auto tConfigWS = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigScalarType> > >
            ( Plato::ScalarArray3DT<ConfigScalarType>("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims) );
        mWorksetFuncs.worksetConfig(tConfigWS->mData);
        aWorkSets.set("configuration", tConfigWS);

        // if essential boundary conditions are enforced weakly, set essential states workset
        if( aDatabase.isScalarVectorDefined("dirichlet") )
        {
            auto tEssentialStateWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
                ( Plato::ScalarMultiVector("Dirichlet Workset", tNumCells, mNumDofsPerCell) );
            mWorksetFuncs.worksetState(aDatabase.vector("dirichlet"), tEssentialStateWS->mData);
            aWorkSets.set("dirichlet", tEssentialStateWS);
        }
    }
};

class VectorFunctionBase
{
public:
    virtual ~VectorFunctionBase(){}

    virtual
    Plato::ScalarVector
    value(const Plato::Database & aDatabase,
          const Plato::Scalar   & aCycle) = 0;

    virtual
    Teuchos::RCP<Plato::CrsMatrixType>
    jacobianState(const Plato::Database & aDatabase,
                  const Plato::Scalar   & aCycle,
                        bool              aTranspose) = 0;

    virtual
    Teuchos::RCP<Plato::CrsMatrixType>
    jacobianControl(const Plato::Database & aDatabase,
                    const Plato::Scalar   & aCycle,
                          bool              aTranspose) = 0;

    virtual
    Teuchos::RCP<Plato::CrsMatrixType>
    jacobianConfig(const Plato::Database & aDatabase,
                   const Plato::Scalar   & aCycle,
                         bool              aTranspose) = 0;
};

template<typename PhysicsType>
class VectorFunction : public VectorFunctionBase
{
private:
    using ElementType = typename PhysicsType::ElementType;

    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
    static constexpr auto mNumNodesPerFace = ElementType::mNumNodesPerFace;
    static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
    static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr auto mNumControl      = ElementType::mNumControl;

    static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims*mNumNodesPerCell;

    using ResidualEvalType  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using JacobianUEvalType = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    using JacobianXEvalType = typename Plato::Elliptic::Evaluation<ElementType>::GradientX;
    using JacobianZEvalType = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;

    std::unordered_map<std::string,std::shared_ptr<ResidualBase>> mResiduals;
    std::unordered_map<std::string,std::shared_ptr<ResidualBase>> mJacobiansU;
    std::unordered_map<std::string,std::shared_ptr<ResidualBase>> mJacobiansX;
    std::unordered_map<std::string,std::shared_ptr<ResidualBase>> mJacobiansZ;

    Plato::DataMap & mDataMap;
    const Plato::SpatialModel & mSpatialModel;
    Plato::WorksetBase<ElementType> mWorksetFuncs;

public:
    VectorFunction
    (const std::string            & aType,
     const Plato::SpatialModel    & aSpatialModel,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aProbParams) :
        mSpatialModel (aSpatialModel),
        mWorksetFuncs (aSpatialModel.Mesh),
        mDataMap      (aDataMap)
    {
        typename PhysicsType::FactoryResidual tFactoryResidual;
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();
            mResiduals [tName] = tFactoryResidual.template createResidual<ResidualEvalType> (aType, tDomain, aDataMap, aProbParams);
            mJacobiansU[tName] = tFactoryResidual.template createResidual<JacobianUEvalType>(aType, tDomain, aDataMap, aProbParams);
            mJacobiansZ[tName] = tFactoryResidual.template createResidual<JacobianZEvalType>(aType, tDomain, aDataMap, aProbParams);
            mJacobiansX[tName] = tFactoryResidual.template createResidual<JacobianXEvalType>(aType, tDomain, aDataMap, aProbParams);
        }
    }

    Plato::OrdinalType numDofs() const
    {
        auto tNumNodes = mSpatialModel.Mesh->NumNodes();
        return (tNumNodes*mNumDofsPerNode);
    }

    std::vector<std::string> getDofNames() const
    {
        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
        return mResiduals.at(tFirstBlockName)->getDofNames();
    }

    Plato::ScalarVector
    value
    (const Plato::Database & aDatabase,
     const Plato::Scalar   & aCycle)
    {
        // set local result workset scalar type
        using ResultScalarType  = typename ResidualEvalType::ResultScalarType;

        auto tNumNodes = mSpatialModel.Mesh->NumNodes();
        WorksetBuilder<ResidualEvalType> tWorksetBuilder(mWorksetFuncs);
        Plato::ScalarVector tResidual("Assembled Residual",mNumDofsPerNode*tNumNodes);

        // internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            // build residual domain worksets
            Plato::WorkSets tWorksets;
            tWorksetBuilder.build(tDomain, aDatabase, tWorksets);

            // build residual range workset
            auto tNumCells = tDomain.numCells();
            auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
                ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
            tWorksets.set("result", tResultWS);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            mResiduals.at(tName)->evaluate( tWorksets, aCycle );

            // assemble to return view
            mWorksetFuncs.assembleResidual(tResultWS->mData, tResidual, tDomain );
        }

        // prescribed boundary conditions
        {
            // build residual domain worksets
            Plato::WorkSets tWorksets;
            auto tNumCells = mSpatialModel.Mesh->NumElements();
            tWorksetBuilder.build(tNumCells, aDatabase, tWorksets);

            // build residual range workset
            auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
                ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
            tWorksets.set("result", tResultWS);

            // evaluate prescribed forces
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mResiduals.at(tFirstBlockName)->evaluateBoundary(mSpatialModel, tWorksets, aCycle );

            // create and assemble to return view
            mWorksetFuncs.assembleResidual(tResultWS->mData, tResidual);
        }

        return tResidual;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    jacobianState
    (const Plato::Database & aDatabase,
     const Plato::Scalar   & aCycle,
           bool              aTranspose = true)
    {
        // set local result workset scalar type
        using ResultScalarType = typename JacobianUEvalType::ResultScalarType;

        // create return Jacobian
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianU =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( tMesh );

        WorksetBuilder<JacobianUEvalType> tWorksetBuilder(mWorksetFuncs);
        // internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            // build jacobian domain worksets
            Plato::WorkSets tWorksets;
            tWorksetBuilder.build(tDomain, aDatabase, tWorksets);

            // build jacobian range workset
            auto tNumCells = tDomain.numCells();
            auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
                ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
            tWorksets.set("result", tResultWS);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            mJacobiansU.at(tName)->evaluate(tWorksets, aCycle);

            // assembly to return Jacobian
            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumDofsPerNode, mNumDofsPerNode>
                tJacEntryOrdinal( tJacobianU, tMesh );

            auto tJacEntries = tJacobianU->entries();
            mWorksetFuncs.assembleJacobianFad(mNumDofsPerCell,mNumDofsPerCell,tJacEntryOrdinal,tResultWS->mData,tJacEntries,tDomain);
        }

        // prescribed forces
        {
            // build jacobian domain worksets
            Plato::WorkSets tWorksets;
            auto tNumCells = mSpatialModel.Mesh->NumElements();
            tWorksetBuilder.build(tNumCells, aDatabase, tWorksets);

            // build jacobian range workset
            auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
                ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
            tWorksets.set("result", tResultWS);

            // evaluate prescribed forces
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mJacobiansU.at(tFirstBlockName)->evaluateBoundary(mSpatialModel, tWorksets, aCycle );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal( tJacobianU, tMesh );

            auto tJacEntries = tJacobianU->entries();
            mWorksetFuncs.assembleJacobianFad(mNumDofsPerCell, mNumDofsPerCell,tJacEntryOrdinal,tResultWS->mData,tJacEntries);
        }
        return tJacobianU;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    jacobianConfig
    (const Plato::Database & aDatabase,
     const Plato::Scalar   & aCycle,
           bool              aTranspose = true)
    {
        // set local result workset scalar type
        using ResultScalarType = typename JacobianXEvalType::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianX;
        if(aTranspose)
        { tJacobianX = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumDofsPerNode>(tMesh); }
        else
        { tJacobianX = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumSpatialDims>(tMesh); }

        WorksetBuilder<JacobianXEvalType> tWorksetBuilder(mWorksetFuncs);
        // internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            // build jacobian domain worksets
            Plato::WorkSets tWorksets;
            tWorksetBuilder.build(tDomain, aDatabase, tWorksets);

            // build jacobian range workset
            auto tNumCells = tDomain.numCells();
            auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
                ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
            tWorksets.set("result", tResultWS);

            // evaluate internal forces
            auto tName     = tDomain.getDomainName();
            mJacobiansX.at(tName)->evaluate(tWorksets, aCycle);

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumSpatialDims, mNumDofsPerNode>
                tJacEntryOrdinal(tJacobianX, tMesh);

            auto tJacEntries = tJacobianX->entries();
            if(aTranspose)
            { mWorksetFuncs.assembleTransposeJacobian(mNumDofsPerCell, mNumConfigDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries, tDomain); }
            else
            { mWorksetFuncs.assembleJacobianFad(mNumDofsPerCell, mNumConfigDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries, tDomain); }
        }

        // prescribed forces
        {
            // build jacobian domain worksets
            Plato::WorkSets tWorksets;
            auto tNumCells = mSpatialModel.Mesh->NumElements();
            tWorksetBuilder.build(tNumCells, aDatabase, tWorksets);

            // build jacobian range workset
            auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
                ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
            tWorksets.set("result", tResultWS);

            // evaluate prescribed forces
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mJacobiansX.at(tFirstBlockName)->evaluateBoundary(mSpatialModel, tWorksets, aCycle );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumSpatialDims, mNumDofsPerNode>
                tJacEntryOrdinal(tJacobianX, tMesh);

            auto tJacEntries = tJacobianX->entries();
            if(aTranspose)
            { mWorksetFuncs.assembleTransposeJacobian(mNumDofsPerCell, mNumConfigDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries); }
            else
            { mWorksetFuncs.assembleJacobianFad(mNumDofsPerCell, mNumConfigDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries); }
        }

        return tJacobianX;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    jacobianControl
    (const Plato::Database & aDatabase,
     const Plato::Scalar   & aCycle,
           bool              aTranspose = true)
    {
        // set local result workset scalar type
        using ResultScalarType = typename JacobianZEvalType::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianZ;
        if(aTranspose)
        { tJacobianZ = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumDofsPerNode>( tMesh ); }
        else
        { tJacobianZ = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumControl>( tMesh ); }

        WorksetBuilder<JacobianZEvalType> tWorksetBuilder(mWorksetFuncs);
        // internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            // build jacobian domain worksets
            Plato::WorkSets tWorksets;
            tWorksetBuilder.build(tDomain, aDatabase, tWorksets);

            // build jacobian range workset
            auto tNumCells = tDomain.numCells();
            auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
                ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
            tWorksets.set("result", tResultWS);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            mJacobiansZ.at(tName)->evaluate(tWorksets, aCycle);

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumControl, mNumDofsPerNode> tJacEntryOrdinal( tJacobianZ, tMesh );

            auto tJacEntries = tJacobianZ->entries();
            if(aTranspose)
            { mWorksetFuncs.assembleTransposeJacobian(mNumDofsPerCell, mNumNodesPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries, tDomain); }
            else
            { mWorksetFuncs.assembleJacobianFad(mNumDofsPerCell, mNumNodesPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries, tDomain); }
        }

        // prescribed forces
        {
            // build jacobian domain worksets
            Plato::WorkSets tWorksets;
            auto tNumCells = mSpatialModel.Mesh->NumElements();
            tWorksetBuilder.build(tNumCells, aDatabase, tWorksets);

            // build jacobian range workset
            auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
                ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
            tWorksets.set("result", tResultWS);

            // evaluate prescribed forces
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mJacobiansZ.at(tFirstBlockName)->evaluateBoundary(mSpatialModel, tWorksets, aCycle );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumControl, mNumDofsPerNode> tJacEntryOrdinal( tJacobianZ, tMesh );

            auto tJacEntries = tJacobianZ->entries();
            if(aTranspose)
            { mWorksetFuncs.assembleTransposeJacobian(mNumDofsPerCell, mNumNodesPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries); }
            else
            { mWorksetFuncs.assembleJacobianFad(mNumDofsPerCell, mNumNodesPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries); }
        }

        return tJacobianZ;
    }
};

enum class evaluator_t
{
    VALUE, GRAD_U, GRAD_Z, GRAD_X
};


class ScalarFunctionBase
{
public:
    virtual ~ScalarFunctionBase(){}

    virtual std::string name() const = 0;

    virtual bool isLinear() const = 0;

    virtual Plato::Scalar
    value(const Plato::Database & aDatabase,
          const Plato::Scalar   & aCycle) = 0;

    virtual Plato::ScalarVector
    gradientControl(const Plato::Database & aDatabase,
                    const Plato::Scalar   & aCycle) = 0;

    virtual Plato::ScalarVector
    gradientState(const Plato::Database & aDatabase,
                  const Plato::Scalar   & aCycle) = 0;

    virtual Plato::ScalarVector
    gradientConfig(const Plato::Database & aDatabase,
                   const Plato::Scalar   & aCycle) = 0;
};

template<typename PhysicsType>
class ScalarFunction : public ScalarFunctionBase
{
private:
    using ElementType = typename PhysicsType::ElementType;

    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
    static constexpr auto mNumNodesPerFace = ElementType::mNumNodesPerFace;
    static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
    static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr auto mNumControl      = ElementType::mNumControl;

    using ValueEvalType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using GradUEvalType = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    using GradXEvalType = typename Plato::Elliptic::Evaluation<ElementType>::GradientX;
    using GradZEvalType = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;

    using CriterionType = std::shared_ptr<CriterionBase>;

    std::unordered_map<std::string, std::shared_ptr<CriterionBase>> mValueFunctions;     /*!< map from domain name to criterion value */
    std::unordered_map<std::string, std::shared_ptr<CriterionBase>> mGradientUFunctions; /*!< map from domain name to criterion gradient wrt state */
    std::unordered_map<std::string, std::shared_ptr<CriterionBase>> mGradientXFunctions; /*!< map from domain name to criterion gradient wrt configuration */
    std::unordered_map<std::string, std::shared_ptr<CriterionBase>> mGradientZFunctions; /*!< map from domain name to criterion gradient wrt control */

    Plato::DataMap & mDataMap;
    const Plato::SpatialModel & mSpatialModel;
    Plato::WorksetBase<ElementType> mWorksetFuncs;

    std::string mName;

public:
    ScalarFunction
    (const std::string            & aFuncName,
     const Plato::SpatialModel    & aSpatialModel,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aProbParams) :
        mDataMap      (aDataMap),
        mSpatialModel (aSpatialModel),
        mWorksetFuncs (aSpatialModel.Mesh),
        mName         (aFuncName)
    {
        this->initialize(aProbParams);
    }

    void
    setEvaluator
    (const evaluator_t   & aEvalType,
     const CriterionType & aCriterion,
     const std::string   & aDomainName)
    {
        switch(aEvalType)
        {
            case evaluator_t::VALUE:
            {
                mValueFunctions[aDomainName] = nullptr; // ensures shared_ptr is decremented
                mValueFunctions[aDomainName] = aCriterion;
                break;
            }
            case evaluator_t::GRAD_U:
            {
                mGradientUFunctions[aDomainName] = nullptr; // ensures shared_ptr is decremented
                mGradientUFunctions[aDomainName] = aCriterion;
                break;
            }
            case evaluator_t::GRAD_Z:
            {
                mGradientZFunctions[aDomainName] = nullptr; // ensures shared_ptr is decremented
                mGradientZFunctions[aDomainName] = aCriterion;
                break;
            }
            case evaluator_t::GRAD_X:
            {
                mGradientXFunctions[aDomainName] = nullptr; // ensures shared_ptr is decremented
                mGradientXFunctions[aDomainName] = aCriterion;
                break;
            }
        }
    }

    std::string name() const { return mName; }

    bool isLinear() const
    {
        auto tDomainName = mSpatialModel.Domains.front().getDomainName();
        return ( mValueFunctions.at(tDomainName)->isLinear() );
    }

    Plato::Scalar
    value
    (const Plato::Database & aDatabase,
     const Plato::Scalar   & aCycle)
    {
        // set local result scalar type
        using ResultScalarType = typename ValueEvalType::ResultScalarType;

        Plato::Scalar tReturnVal(0.0);
        WorksetBuilder<ValueEvalType> tWorksetBuilder(mWorksetFuncs);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            // build criterion value domain worksets
            Plato::WorkSets tWorksets;
            tWorksetBuilder.build(tDomain, aDatabase, tWorksets);

            // build criterion value range workset
            auto tNumCells = tDomain.numCells();
            auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarVectorT<ResultScalarType> > >
                ( Plato::ScalarVectorT<ResultScalarType>("Result Workset", tNumCells) );
            Kokkos::deep_copy(tResultWS->mData, 0.0);
            tWorksets.set("result", tResultWS);

            // save result workset to database
            auto tDomainName = tDomain.getDomainName();
            mDataMap.scalarVectors[mValueFunctions.at(tDomainName)->getName()] = tResultWS->mData;

            // evaluate criterion
            mValueFunctions.at(tDomainName)->evaluate(tWorksets, aCycle);

            // sum across elements
            tReturnVal += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS->mData);
        }

        // apply post operation to return value, if defined
        auto tDomainName = mSpatialModel.Domains.front().getDomainName();
        mValueFunctions.at(tDomainName)->postEvaluate(tReturnVal);

        return tReturnVal;
    }

    Plato::ScalarVector
    gradientConfig
    (const Plato::Database & aDatabase,
     const Plato::Scalar   & aCycle)
    {
        // set local result type
        using ResultScalarType = typename GradXEvalType::ResultScalarType;

        // create output
        auto tNumNodes = mWorksetFuncs.numNodes();
        Plato::ScalarVector tGradientX("criterion gradient configuration", mNumSpatialDims * tNumNodes);

        // evaluate gradient
        Plato::Scalar tValue(0.0);
        WorksetBuilder<GradXEvalType> tWorksetBuilder(mWorksetFuncs);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            // build gradient domain worksets
            Plato::WorkSets tWorksets;
            tWorksetBuilder.build(tDomain, aDatabase, tWorksets);

            // build gradient range workset
            auto tNumCells = tDomain.numCells();
            auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarVectorT<ResultScalarType> > >
                ( Plato::ScalarVectorT<ResultScalarType>("Result Workset", tNumCells) );
            Kokkos::deep_copy(tResultWS->mData, 0.0);
            tWorksets.set("result", tResultWS);

            // evaluate gradient
            auto tName = tDomain.getDomainName();
            mGradientXFunctions.at(tName)->evaluate(tWorksets, aCycle);

            // assemble gradient
            mWorksetFuncs.assembleVectorGradientFadX(tResultWS->mData, tGradientX);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS->mData);
        }

        // apply post operation to return values, if defined
        auto tDomainName = mSpatialModel.Domains.front().getDomainName();
        mGradientXFunctions.at(tDomainName)->postEvaluate(tGradientX, tValue);

        return tGradientX;
    }

    Plato::ScalarVector
    gradientState
    (const Plato::Database & aDatabase,
     const Plato::Scalar   & aCycle)
    {
        // set local result type
        using ResultScalarType = typename GradUEvalType::ResultScalarType;

        // create output
        auto tNumNodes = mWorksetFuncs.numNodes();
        Plato::ScalarVector tGradientU("criterion gradient state", mNumDofsPerNode * tNumNodes);

        // evaluate gradient
        Plato::Scalar tValue(0.0);
        WorksetBuilder<GradUEvalType> tWorksetBuilder(mWorksetFuncs);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            // build gradient domain worksets
            Plato::WorkSets tWorksets;
            tWorksetBuilder.build(tDomain, aDatabase, tWorksets);

            // build gradient range workset
            auto tNumCells = tDomain.numCells();
            auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarVectorT<ResultScalarType> > >
                ( Plato::ScalarVectorT<ResultScalarType>("Result Workset", tNumCells) );
            Kokkos::deep_copy(tResultWS->mData, 0.0);
            tWorksets.set("result", tResultWS);

            // evaluate function
            auto tName = tDomain.getDomainName();
            mGradientUFunctions.at(tName)->evaluate(tWorksets, aCycle);

            // assemble gradient
            mWorksetFuncs.assembleVectorGradientFadU(tResultWS->mData, tGradientU);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS->mData);
        }

        // apply post operation to return values, if defined
        auto tDomainName = mSpatialModel.Domains.front().getDomainName();
        mGradientUFunctions.at(tDomainName)->postEvaluate(tGradientU, tValue);

        return tGradientU;
    }

    Plato::ScalarVector
    gradientControl
    (const Plato::Database & aDatabase,
     const Plato::Scalar   & aCycle)
    {
        // set local result type
        using ResultScalarType = typename GradZEvalType::ResultScalarType;

        // create output
        auto tNumNodes = mWorksetFuncs.numNodes();
        Plato::ScalarVector tGradientZ("criterion gradient control", tNumNodes);

        // evaluate gradient
        Plato::Scalar tValue(0.0);
        WorksetBuilder<GradZEvalType> tWorksetBuilder(mWorksetFuncs);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            // build gradient domain worksets
            Plato::WorkSets tWorksets;
            tWorksetBuilder.build(tDomain, aDatabase, tWorksets);

            // build gradient range workset
            auto tNumCells = tDomain.numCells();
            auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarVectorT<ResultScalarType> > >
                ( Plato::ScalarVectorT<ResultScalarType>("Result Workset", tNumCells) );
            Kokkos::deep_copy(tResultWS->mData, 0.0);
            tWorksets.set("result", tResultWS);

            // evaluate gradient
            auto tName = tDomain.getDomainName();
            mGradientZFunctions.at(tName)->evaluate(tWorksets, aCycle);

            // assemble gradient
            mWorksetFuncs.assembleScalarGradientFadZ(tDomain, tResultWS->mData, tGradientZ);

            // assemble value
            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS->mData);
        }

        // apply post operation to return values, if defined
        auto tName = mSpatialModel.Domains.front().getDomainName();
        mGradientZFunctions.at(tName)->postEvaluate(tGradientZ, tValue);

        return tGradientZ;
    }

private:
    void initialize(Teuchos::ParameterList & aProbParams)
    {
        typename PhysicsType::FactoryCriterion tFactory;

        auto tProblemDefaults = aProbParams.sublist("Criteria").sublist(mName);
        auto tFunType = tProblemDefaults.get<std::string>("Scalar Function Type", "");

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tDomainName = tDomain.getDomainName();
            mValueFunctions    [tDomainName] = tFactory.template createCriterion<ValueEvalType>(tFunType, mName, tDomain, mDataMap, aProbParams);
            mGradientUFunctions[tDomainName] = tFactory.template createCriterion<GradUEvalType>(tFunType, mName, tDomain, mDataMap, aProbParams);
            mGradientXFunctions[tDomainName] = tFactory.template createCriterion<GradXEvalType>(tFunType, mName, tDomain, mDataMap, aProbParams);
            mGradientZFunctions[tDomainName] = tFactory.template createCriterion<GradZEvalType>(tFunType, mName, tDomain, mDataMap, aProbParams);
        }
    }
};

template<typename PhysicsType>
class FactoryScalarFunction
{
public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    FactoryScalarFunction () {}

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    ~FactoryScalarFunction() {}

    /******************************************************************************//**
     * \brief Create method
     * \param [in] aMesh mesh database
     * \param [in] aDataMap Plato Engine and Analyze data map
     * \param [in] aInputParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    std::shared_ptr<ScalarFunctionBase>
    create
    (const std::string            & aFuncName,
     const Plato::SpatialModel    & aSpatialModel,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aProbParams)
    {
        auto tFuncParams = aProbParams.sublist("Criteria").sublist(aFuncName);
        auto tFuncType = tFuncParams.get<std::string>("Type", "Not Defined");

        if(tFuncType == "Scalar Function")
        {
            return std::make_shared<ScalarFunction<PhysicsType>>(aFuncName, aSpatialModel, aDataMap, aProbParams);
        }
        else
        {
            return nullptr;
        }
        return nullptr;
    }
};


void set_essential_state_values
(const Plato::OrdinalVector & aBcDofs,
 const Plato::ScalarVector  & aBcValues,
       Plato::ScalarVector  & aEssentialStates)
{
    const Plato::OrdinalType tNumEssentialDofs = aBcDofs.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumEssentialDofs),
                        KOKKOS_LAMBDA(const Plato::OrdinalType & aDofOrdinal)
    {
        aEssentialStates( aBcDofs(aDofOrdinal) ) = aBcValues(aDofOrdinal);
    }, "Set Essential State Values");
}


template<typename PhysicsType>
class Problem : public Plato::AbstractProblem
{
private:
    // define local types
    using ElementType = typename PhysicsType::ElementType;
    using Criterion = std::shared_ptr<ScalarFunctionBase>;

    bool mSaveState;
    bool mWeakEssentialBoundaryConditions = false;

    std::string mPDE; /*!< Partial Differential Equation (PDE) type */
    std::string mPhysics; /*!< problem physics type */
    std::string mEssentialBC; /*!< essential boundary condition type: 'strong' or 'weak' */

    Plato::ScalarVector mResidual;
    Plato::ScalarVector mEssentialStates;
    Plato::ScalarVector mEssentialAdjoints;

    Plato::ScalarMultiVector mStates;
    Plato::ScalarMultiVector mAdjoints;

    Teuchos::RCP<Plato::CrsMatrixType> mJacobianState; /*!< Jacobian with respect to state variables */

    Plato::OrdinalVector mBcDofs;  /*!< list of essential boundary condition degrees of freedom */
    Plato::ScalarVector mBcValues; /*!< list of values associated with the essential boundary condition degrees of freedom */

    std::shared_ptr<Plato::AbstractSolver> mSolver;
    std::shared_ptr<VectorFunction<PhysicsType>> mResidualEvaluator;

    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance, contains the mesh, mesh sets, domains, etc. */

    std::unordered_map<std::string, Criterion> mCriterionEvaluator;

public:
    Problem
    (Plato::Mesh            &aMesh,
     Teuchos::ParameterList &aProbParams,
     Comm::Machine          &aMachine) :
            Plato::AbstractProblem(aMesh, aProbParams),
            mSpatialModel(aMesh, aProbParams, mDataMap),
            mSaveState(aProbParams.sublist("Elliptic").isType < Teuchos::Array < std::string >> ("Plottable")),
            mResidual(),
            mStates(),
            mJacobianState(Teuchos::null),
            mPDE(aProbParams.get < std::string > ("PDE Constraint")),
            mPhysics(aProbParams.get < std::string > ("Physics"))
    {
        this->initializeSolver(aMesh,aProbParams,aMachine);
        this->initializeEvaluators(aProbParams);
        this->readEssentialBoundaryConditions(aProbParams);
    }

    Plato::Solutions getSolution() const
    {
        Plato::Solutions tSolution(mPhysics, mPDE);
        tSolution.set("state", mStates, mResidualEvaluator->getDofNames());
        return tSolution;
    }

    void
    output
    (const std::string & aFilepath)
    {
        auto tDataMap = this->getDataMap();
        auto tSolution = this->getSolution();
        Plato::universal_solution_output(aFilepath, tSolution, tDataMap, mSpatialModel.Mesh);
    }

    void
    updateProblem
    (const Plato::ScalarVector & aControl,
     const Plato::Solutions    & aSolution)
    { return; }

    Plato::Solutions
    solution
    (const Plato::ScalarVector & aControl)
    {
        // build database
        Plato::Database tDatabase;
        this->buildDatabase(aControl,tDatabase);

        // initializa state vector
        Plato::ScalarVector tStateVector = tDatabase.vector("states");
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tStateVector);

        mDataMap.clearStates();
        mDataMap.scalarNodeFields["Topology"] = tDatabase.vector("controls");

        constexpr Plato::Scalar tCYCLE = 0.0;
        mResidual = mResidualEvaluator->value(tDatabase,tCYCLE);
        Plato::blas1::scale(-1.0, mResidual);
        mJacobianState = mResidualEvaluator->jacobianState(tDatabase, tCYCLE, /*transpose=*/ false);

        // solve linear system of equations
        if( !mWeakEssentialBoundaryConditions )
        { this->enforceStrongEssentialBoundaryConditions(mJacobianState,mResidual,1.0); }
        Plato::ScalarVector tDeltaState("increment", tStateVector.extent(0));
        Plato::blas1::fill(0.0, tDeltaState);
        mSolver->solve(*mJacobianState, tDeltaState, mResidual);
        Plato::blas1::axpy(1.0, tDeltaState, tStateVector);

        if ( mSaveState )
        {
            // evaluate at new state
            mResidual = mResidualEvaluator->value(tDatabase,tCYCLE);
            mDataMap.saveState();
        }

        auto tSolution = this->getSolution();
        return tSolution;
    }

    Plato::Scalar
    criterionValue
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        Plato::Database tDatabase;
        this->buildDatabase(aControl,tDatabase);
        if( mCriterionEvaluator.count(aName) )
        {
            constexpr Plato::Scalar tCYCLE = 0.0;
            auto tValue = mCriterionEvaluator[aName]->value(tDatabase, tCYCLE);
            return tValue;
        }
        else
        {
            auto tErrMsg = this->getErrorMsg(aName);
            ANALYZE_THROWERR(tErrMsg)
        }
    }

    Plato::Scalar
    criterionValue
    (const Plato::ScalarVector & aControl,
     const Plato::Solutions    & aSolution,
     const std::string         & aName)
    {
        return (this->criterionValue(aControl,aName));
    }

    Plato::ScalarVector
    criterionGradient
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        if( mCriterionEvaluator.find(aName) == mCriterionEvaluator.end() )
        {
            auto tErrMsg = this->getErrorMsg(aName);
            ANALYZE_THROWERR(tErrMsg)
        }

        // build database
        Plato::Database tDatabase;
        this->buildDatabase(aControl,tDatabase);

        // compute gradient
        if(mCriterionEvaluator.at(aName)->isLinear() )
        {
            return ( mCriterionEvaluator.at(aName)->gradientControl(tDatabase,/*cycle=*/ 0.0) );
        }
        else
        {
            return ( this->computeCriterionGradientControl(tDatabase, mCriterionEvaluator[aName]) );
        }
    }

    Plato::ScalarVector
    criterionGradient
    (const Plato::ScalarVector & aControl,
     const Plato::Solutions    & aSolution,
     const std::string         & aName)
    {
        return (this->criterionGradient(aControl,aName));
    }

    Plato::ScalarVector
    criterionGradientX
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        if( mCriterionEvaluator.find(aName) == mCriterionEvaluator.end() )
        {
            auto tErrMsg = this->getErrorMsg(aName);
            ANALYZE_THROWERR(tErrMsg)
        }

        // build database
        Plato::Database tDatabase;
        this->buildDatabase(aControl,tDatabase);

        // compute gradient
        if(mCriterionEvaluator.at(aName)->isLinear() )
        {
            return ( mCriterionEvaluator.at(aName)->gradientConfig(tDatabase,/*cycle=*/ 0.0) );
        }
        else
        {
            return ( this->computeCriterionGradientConfig(tDatabase, mCriterionEvaluator[aName]) );
        }
    }

    Plato::ScalarVector
    criterionGradientX
    (const Plato::ScalarVector & aControl,
     const Plato::Solutions    & aSolution,
     const std::string         & aName)
    {
        return (this->criterionGradientX(aControl,aName));
    }

private:
    std::string
    getErrorMsg
    (const std::string & aName)
    const
    {
        std::string tMsg = std::string("ERROR: Criterion parameter list with name '")
            + aName + "' is not defined. " + "Parsed criterion parameter list names are: ";
        for(const auto& tPair : mCriterionEvaluator)
        {
            tMsg = tMsg + "'" + tPair.first + "', ";
        }
        auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
        tSubMsg += ". The value provided for the criterion 'Type' keyword and the parameter list name must match.";
        return tSubMsg;
    }

    void
    initializeSolver
    (Plato::Mesh            &aMesh,
     Teuchos::ParameterList &aProbParams,
     Comm::Machine          &aMachine)
    {
        mPhysics = Plato::tolower(mPhysics);
        auto tSystemType = LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE;
        if(mPhysics == "electromechanical" || mPhysics == "thermomechanical")
        {
            tSystemType = LinearSystemType::SYMMETRIC_INDEFINITE;
        }

        Plato::SolverFactory tSolverFactory(aProbParams.sublist("Linear Solver"), tSystemType);
        mSolver = tSolverFactory.create(aMesh->NumNodes(), aMachine, ElementType::mNumDofsPerNode);
    }

    void
    initializeEvaluators
    (Teuchos::ParameterList& aProbParams)
    {
        mResidualEvaluator = std::make_shared<VectorFunction<PhysicsType>>(mPhysics,mSpatialModel,mDataMap,aProbParams);
        mStates = Plato::ScalarMultiVector("States", 1, mResidualEvaluator->numDofs());

        if(aProbParams.isSublist("Criteria"))
        {
            FactoryScalarFunction<PhysicsType> tFactoryScalarFunction;

            auto tFuncParams = aProbParams.sublist("Criteria");
            for(Teuchos::ParameterList::ConstIterator tIndex = tFuncParams.begin(); tIndex != tFuncParams.end(); ++tIndex)
            {
                const Teuchos::ParameterEntry &tEntry = tFuncParams.entry(tIndex);
                std::string tScalarFuncName = tFuncParams.name(tIndex);

                std::string tErrMsg("Parameter in Criteria block not valid.  Expect lists only.");
                TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error, tErrMsg);

                auto tCriterion = tFactoryScalarFunction.create(tScalarFuncName, mSpatialModel, mDataMap, aProbParams);
                if(tCriterion != nullptr)
                {
                    mCriterionEvaluator[tScalarFuncName] = tCriterion;
                }
            }
            if( mCriterionEvaluator.size() )
            {
                auto tNumDofs = mResidualEvaluator->numDofs();
                mAdjoints = Plato::ScalarMultiVector("Adjoint Vector", 1, tNumDofs);
            }
        }
    }

    void
    readEssentialBoundaryConditions
    (Teuchos::ParameterList& aProbParams)
    {
        if(aProbParams.isSublist("Essential Boundary Conditions") == false)
        {
            ANALYZE_THROWERR("ERROR: Essential boundary conditions parameter list is not defined")
        }

        Plato::EssentialBCs<ElementType>
        tEssentialBoundaryConditions(aProbParams.sublist("Essential Boundary Conditions", false), mSpatialModel.Mesh);
        tEssentialBoundaryConditions.get(mBcDofs, mBcValues);

        if(aProbParams.isSublist("Nitsche Boundary Conditions") == true)
        { mWeakEssentialBoundaryConditions = true; }
    }

    void
    buildDatabase
    (const Plato::ScalarVector & aControl,
           Plato::Database     & aDatabase)
    {
        constexpr size_t tCYCLE_INDEX = 0;
        auto tStateVector = Kokkos::subview(mStates, tCYCLE_INDEX, Kokkos::ALL());
        aDatabase.vector("states"  , tStateVector);
        aDatabase.vector("controls", aControl);

        if(mWeakEssentialBoundaryConditions)
        { this->enforceWeakEssentialBoundaryConditions(aDatabase); }
    }

    void enforceWeakEssentialBoundaryConditions
    (Plato::Database & aDatabase)
    {
        // Essential Boundary Conditions (EBCs)
        mEssentialStates = Plato::ScalarVector("State EBCs", mResidualEvaluator->numDofs());
        set_essential_state_values(mBcDofs, mBcValues, mEssentialStates);
        aDatabase.vector("dirichlet", mEssentialStates);
    }

    void enforceStrongEssentialBoundaryConditions
    (const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
     const Plato::ScalarVector                & aVector,
     const Plato::Scalar                      & aMultiplier)
    {
        if(aMatrix->isBlockMatrix())
        {
            Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>
            (aMatrix, aVector, mBcDofs, mBcValues, aMultiplier);
        }
        else
        {
            Plato::applyConstraints<ElementType::mNumDofsPerNode>
            (aMatrix, aVector, mBcDofs, mBcValues, aMultiplier);
        }
    }

    void enforceWeakEssentialAdjointBoundaryConditions
    (Plato::Database & aDatabase)
    {
        // Essential Boundary Conditions (EBCs)
        mEssentialAdjoints = Plato::ScalarVector("Adjoint EBCs", mResidualEvaluator->numDofs());
        Kokkos::deep_copy(mEssentialAdjoints, 0.0);
        aDatabase.vector("dirichlet", mEssentialAdjoints);
    }

    void enforceStrongEssentialAdjointBoundaryConditions
    (const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
     const Plato::ScalarVector                & aVector)
    {
        // Essential Boundary Conditions (EBCs)
        Plato::ScalarVector tDirichletValues("Adjoint EBCs", mBcValues.size());
        Plato::blas1::scale(static_cast<Plato::Scalar>(0.0), tDirichletValues);
        if(aMatrix->isBlockMatrix())
        {
            Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, tDirichletValues);
        }
        else
        {
            Plato::applyConstraints<ElementType::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, tDirichletValues);
        }
    }

    Plato::ScalarVector
    computeCriterionGradientControl
    (Plato::Database & aDatabase,
           Criterion       & aCriterion)
    {
        if(aCriterion == nullptr)
        {
            ANALYZE_THROWERR("ERROR: Requested criterion is null");
        }

        if(static_cast<Plato::OrdinalType>(mAdjoints.size()) <= static_cast<Plato::OrdinalType>(0))
        {
            const auto tNumDofs = mResidualEvaluator->numDofs();
            mAdjoints = Plato::ScalarMultiVector("Adjoint Variables", 1, tNumDofs);
        }

        // compute criterion contribution to the gradient
        constexpr Plato::Scalar tCYCLE = 0.0;
        auto tGradientControl = aCriterion->gradientControl(aDatabase, tCYCLE);

        // add residual contribution to the gradient
        {
            // compute gradient with respect to state variables
            auto tGradientState = aCriterion->gradientState(aDatabase, tCYCLE);
            Plato::blas1::scale(-1.0, tGradientState);

            // compute jacobian with respect to state variables
            mJacobianState = mResidualEvaluator->jacobianState(aDatabase, tCYCLE, /*transpose=*/ true);
            if( mWeakEssentialBoundaryConditions )
            { this->enforceWeakEssentialAdjointBoundaryConditions(aDatabase); }
            else
            { this->enforceStrongEssentialAdjointBoundaryConditions(mJacobianState, tGradientState); }

            // solve adjoint system of equations
            constexpr size_t tCYCLE_INDEX = 0;
            Plato::ScalarVector tAdjointVector = Kokkos::subview(mAdjoints, tCYCLE_INDEX, Kokkos::ALL());
            mSolver->solve(*mJacobianState, tAdjointVector, tGradientState, /*isAdjointSolve=*/ true);

            // compute jacobian with respect to control variables
            auto tJacobianControl = mResidualEvaluator->jacobianControl(aDatabase, tCYCLE, /*transpose=*/ true);

            // compute gradient with respect to design variables
            Plato::MatrixTimesVectorPlusVector(tJacobianControl, tAdjointVector, tGradientControl);
        }

        return tGradientControl;
    }

    Plato::ScalarVector
    computeCriterionGradientConfig
    (Plato::Database & aDatabase,
     Criterion       & aCriterion)
    {
        if(aCriterion == nullptr)
        {
            ANALYZE_THROWERR("ERROR: Requested criterion is null");
        }

        if(static_cast<Plato::OrdinalType>(mAdjoints.size()) <= static_cast<Plato::OrdinalType>(0))
        {
            const auto tNumDofs = mResidualEvaluator->numDofs();
            mAdjoints = Plato::ScalarMultiVector("Adjoint Variables", 1, tNumDofs);
        }

        // compute criterion contribution to the gradient
        constexpr Plato::Scalar tCYCLE = 0.0;
        auto tGradientConfig  = aCriterion->gradientConfig(aDatabase, tCYCLE);

        // add residual contribution to the gradient
        {
            // compute gradient with respect to state variables
            auto tGradientState = aCriterion->gradientState(aDatabase, tCYCLE);
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tGradientState);

            // compute jacobian with respect to state variables
            mJacobianState = mResidualEvaluator->jacobianState(aDatabase, tCYCLE, /*transpose=*/true);
            if( mWeakEssentialBoundaryConditions )
            { this->enforceWeakEssentialAdjointBoundaryConditions(aDatabase); }
            else
            { this->enforceStrongEssentialAdjointBoundaryConditions(mJacobianState, tGradientState); }

            // solve adjoint system of equations
            constexpr size_t tCYCLE_INDEX = 0;
            Plato::ScalarVector tAdjointVector = Kokkos::subview(mAdjoints, tCYCLE_INDEX, Kokkos::ALL());
            mSolver->solve(*mJacobianState, tAdjointVector, tGradientState, /*isAdjointSolve=*/ true);

            // compute jacobian with respect to configuration variables
            auto tJacobianConfig = mResidualEvaluator->jacobianConfig(aDatabase, tCYCLE, /*transpose=*/ true);

            // compute gradient with respect to design variables: dgdx * adjoint + dfdx
            Plato::MatrixTimesVectorPlusVector(tJacobianConfig, tAdjointVector, tGradientConfig);
        }

        return tGradientConfig;
    }
};

}
// namespace exp

}
// namespace Plato

namespace IfemTests
{

TEUCHOS_UNIT_TEST(Morphorm, Elastostatics)
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", tMeshWidth);

    // create input
    //
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                             \n"
      "  <ParameterList name='Spatial Model'>                                           \n"
      "    <ParameterList name='Domains'>                                               \n"
      "      <ParameterList name='Design Volume'>                                       \n"
      "        <Parameter name='Element Block' type='string' value='body'/>             \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
      "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
      "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                      \n"
      "  <ParameterList name='Elliptic'>                                                \n"
      "    <ParameterList name='Penalty Function'>                                      \n"
      "      <Parameter name='Exponent' type='double' value='1.0'/>                     \n"
      "      <Parameter name='Minimum Value' type='double' value='0.0'/>                \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                        \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList name='Material Models'>                                         \n"
      "    <ParameterList name='Unobtainium'>                                           \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
      "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
      "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>           \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                            \n"
      "    <ParameterList  name='Traction Vector Boundary Condition'>                   \n"
      "      <Parameter  name='Type'     type='string'        value='Uniform'/>         \n"
      "      <Parameter  name='Values'   type='Array(double)' value='{1.0, 0.0, 0.0}'/> \n"
      "      <Parameter  name='Sides'    type='string'        value='x+'/>              \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList  name='Essential Boundary Conditions'>                          \n"
      "    <ParameterList  name='X Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='0'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='1'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "    <ParameterList  name='Z Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='2'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "</ParameterList>                                                                 \n"
    );

    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    Plato::exp::Problem<Plato::exp::PhysicsMechanics<Plato::Tet10>>
        tElasticityProblem(tMesh, *tParamList, tMachine);

    // SOLVE ELASTOSTATICS EQUATIONS
    auto tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    auto tElasticitySolution = tElasticityProblem.solution(tControl);
    //tElasticityProblem.output("output_strong");

    // TEST RESULTS
    constexpr Plato::OrdinalType tTimeStep = 0;
    auto tState = tElasticitySolution.get("State");
    auto tSolution = Kokkos::subview(tState, tTimeStep, Kokkos::ALL());
    auto tHostSolution = Kokkos::create_mirror_view(tSolution);
    Kokkos::deep_copy(tHostSolution, tSolution);

    std::vector<Plato::Scalar> tGold =
        {8.44215e-8, 9.58193e-7, -7.30424e-8, 4.50125e-9,
         9.61752e-7, -7.46016e-8, -7.46016e-8,
         9.68308e-7, -7.43541e-8, -1.50715e-7,
         9.67836e-7, -1.47979e-7, 1.60339e-7,
         9.65735e-7, -1.47873e-7, 8.41994e-8,
         9.6498e-7, -1.49664e-7, 4.12353e-9,
         9.68308e-7, -1.50715e-7, -7.43541e-8,
         9.79216e-7, -1.52588e-7, -1.52588e-7};

    constexpr Plato::Scalar tTolerance = 1e-4;
    constexpr Plato::OrdinalType tDofOffset = 350; // comparing only the last 25 dofs
    for(Plato::OrdinalType tDofIndex=0; tDofIndex < tGold.size(); tDofIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostSolution(tDofOffset+tDofIndex), tGold[tDofIndex], tTolerance);
    }
}

/******************************************************************************/
/*!
  \brief Compute value and both gradients (wrt state, config, and control) of
         InternalElasticEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Morphorm, InternalElasticEnergy3D )
{
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Internal Elastic Energy'>                            \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create test mesh
  //
  constexpr int tMeshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", tMeshWidth);

  // create mesh based density from host data
  //
  auto tNumNodes = tMesh->NumNodes();
  Plato::ScalarVector tControl("density", tNumNodes);
  Kokkos::deep_copy(tControl, 1.0);

  // create mesh based displacement from host data
  //
  auto tNumDofs = tMesh->NumDimensions()*tMesh->NumNodes();
  Plato::ScalarVector tState("states", tNumDofs);
  auto tHostState = Kokkos::create_mirror_view( tState );
  Plato::Scalar tDisp = 0.0, tDval = 0.0001;
  for(decltype(tNumDofs) i=0; i<tNumDofs; i++)
  {
      tHostState(i) = (tDisp += tDval);
  }
  Kokkos::deep_copy(tState, tHostState);

  // create objective
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  std::string tMyFunctionName("Internal Elastic Energy");
  Plato::exp::ScalarFunction<Plato::exp::PhysicsMechanics<Plato::Tet10>>
    tScalarFunction(tMyFunctionName, tSpatialModel, tDataMap, *tParamList);

  // set database
  //
  Plato::Database tDatabase;
  tDatabase.vector("states"  , tState);
  tDatabase.vector("controls", tControl);

  // compute and test criterion value
  //
  constexpr Plato::Scalar tCYCLE = 0.0;
  auto tValue = tScalarFunction.value(tDatabase,tCYCLE);

  Plato::Scalar tValueGold = 1206.13846153846043;
  TEST_FLOATING_EQUALITY(tValue, tValueGold, 1e-13);

  // compute and test criterion gradient wrt state
  //
  auto tGradU = tScalarFunction.gradientState(tDatabase,tCYCLE);

  auto tHostGradU = Kokkos::create_mirror_view( tGradU );
  Kokkos::deep_copy( tHostGradU, tGradU );

  std::vector<Plato::Scalar> tGoldGradU = {
   0., 0., 0., -2432.692307692301, -1663.461538461530,
   -615.3846153846263, 0., 0., 0., -2432.692307692299,
   -1663.461538461529, -615.3846153846263, 0., 0., 0.,
   -2355.769230769225, -692.3076923077053, -1432.692307692301,
   -3711.538461538447, -1153.846153846157, -1000.000000000002,
   -3711.538461538460, -1153.846153846169, -999.9999999999920,
   -3711.538461538446, -1153.846153846156, -1000.000000000002,
   -1355.769230769233 };

  for(int iNode=0; iNode<int(tGoldGradU.size()); iNode++){
      if(tGoldGradU[iNode] == 0.0)
      {
          TEST_ASSERT(fabs(tHostGradU[iNode]) < 1e-10);
      }
      else
      {
          TEST_FLOATING_EQUALITY(tHostGradU[iNode], tGoldGradU[iNode], 1e-13);
      }
  }

  // compute and test criterion gradient wrt control, control
  //
  auto tGradZ = tScalarFunction.gradientControl(tDatabase,tCYCLE);

  auto tHostGradZ = Kokkos::create_mirror_view( tGradZ );
  Kokkos::deep_copy( tHostGradZ, tGradZ );

  std::vector<Plato::Scalar> tGoldGradZ = {
    -7.53836538461538375, 10.0511538461537988, -10.0511538461538468,
    10.0511538461537935, -2.51278846153846125, 10.0511538461537988,
    10.0511538461537988, 15.0767307692307462, 10.0511538461537935,
    5.02557692307694648, -10.0511538461538485, 15.0767307692307444,
    -15.0767307692307710, 15.0767307692307391, -5.02557692307692072,
    10.0511538461537953, 10.0511538461537917, 15.0767307692307426};

  for(int iNode=0; iNode<int(tGoldGradZ.size()); iNode++){
    TEST_FLOATING_EQUALITY(tHostGradZ[iNode], tGoldGradZ[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto tGradX = tScalarFunction.gradientConfig(tDatabase,tCYCLE);

  auto tHostGradX = Kokkos::create_mirror_view( tGradX );
  Kokkos::deep_copy(tHostGradX, tGradX);

  std::vector<Plato::Scalar> tGoldGradX = {
    0., 0., 0., 91.0903846153847354, -21.9865384615381778,
    5.65384615384538058, 0., 0., 0., 91.0903846153847496,
   -21.9865384615381956, 5.65384615384537881, 0., 0., 0.,
    84.1673076923079861, 26.8846153846146265, -44.8788461538458705,
    75.4500000000003013, 35.1923076923072244, 7.03846153846112799,
    75.4500000000003723, 35.1923076923068976, 7.03846153846193090,
    75.4500000000002871, 35.1923076923071747, 7.03846153846111378
  };

  for(int iNode=0; iNode<int(tGoldGradX.size()); iNode++){
      if(tGoldGradX[iNode] == 0.0)
      {
          TEST_ASSERT(fabs(tHostGradX[iNode]) < 1e-10);
      }
      else
      {
          TEST_FLOATING_EQUALITY(tHostGradX[iNode], tGoldGradX[iNode], 1e-12);
      }
  }
}

TEUCHOS_UNIT_TEST( Morphorm, TestInternalElasticEnergyGradZ_3D_TET10 )
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", tMeshWidth);

    // create input
    //
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                             \n"
      "  <ParameterList name='Spatial Model'>                                           \n"
      "    <ParameterList name='Domains'>                                               \n"
      "      <ParameterList name='Design Volume'>                                       \n"
      "        <Parameter name='Element Block' type='string' value='body'/>             \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
      "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
      "  <ParameterList name='Criteria'>                                                \n"
      "    <ParameterList name='Internal Elastic Energy'>                               \n"
      "      <Parameter name='Type' type='string' value='Scalar Function'/>             \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
      "      <ParameterList name='Penalty Function'>                                    \n"
      "        <Parameter name='Exponent' type='double' value='1.0'/>                   \n"
      "        <Parameter name='Minimum Value' type='double' value='0.0'/>              \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>                      \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList name='Elliptic'>                                                \n"
      "    <ParameterList name='Penalty Function'>                                      \n"
      "      <Parameter name='Exponent' type='double' value='1.0'/>                     \n"
      "      <Parameter name='Minimum Value' type='double' value='0.0'/>                \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                        \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList name='Material Models'>                                         \n"
      "    <ParameterList name='Unobtainium'>                                           \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
      "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
      "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>           \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                            \n"
      "    <ParameterList  name='Traction Vector Boundary Condition'>                   \n"
      "      <Parameter  name='Type'     type='string'        value='Uniform'/>         \n"
      "      <Parameter  name='Values'   type='Array(double)' value='{1.0, 0.0, 0.0}'/> \n"
      "      <Parameter  name='Sides'    type='string'        value='x+'/>              \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList  name='Essential Boundary Conditions'>                          \n"
      "    <ParameterList  name='X Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='0'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='1'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "    <ParameterList  name='Z Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='2'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "</ParameterList>                                                                 \n"
    );

    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    Plato::exp::Problem<Plato::exp::PhysicsMechanics<Plato::Tet10>>
        tElasticityProblem(tMesh, *tParamList, tMachine);

    auto tError = Plato::test_criterion_grad_wrt_control(tElasticityProblem, tMesh, "Internal Elastic Energy");
    TEST_ASSERT(tError < 1e-4);
}

TEUCHOS_UNIT_TEST( Morphorm, TestVolumeGradZ_3D_TET10 )
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", tMeshWidth);

    // create input
    //
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                             \n"
      "  <ParameterList name='Spatial Model'>                                           \n"
      "    <ParameterList name='Domains'>                                               \n"
      "      <ParameterList name='Design Volume'>                                       \n"
      "        <Parameter name='Element Block' type='string' value='body'/>             \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
      "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
      "  <ParameterList name='Criteria'>                                                \n"
      "    <ParameterList name='Volume'>                                                \n"
      "      <Parameter name='Type' type='string' value='Scalar Function'/>             \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Volume'/>      \n"
      "      <ParameterList name='Penalty Function'>                                    \n"
      "        <Parameter name='Exponent' type='double' value='1.0'/>                   \n"
      "        <Parameter name='Minimum Value' type='double' value='0.0'/>              \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>                      \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList name='Elliptic'>                                                \n"
      "    <ParameterList name='Penalty Function'>                                      \n"
      "      <Parameter name='Exponent' type='double' value='1.0'/>                     \n"
      "      <Parameter name='Minimum Value' type='double' value='0.0'/>                \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                        \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList name='Material Models'>                                         \n"
      "    <ParameterList name='Unobtainium'>                                           \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
      "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
      "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>           \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                            \n"
      "    <ParameterList  name='Traction Vector Boundary Condition'>                   \n"
      "      <Parameter  name='Type'     type='string'        value='Uniform'/>         \n"
      "      <Parameter  name='Values'   type='Array(double)' value='{1.0, 0.0, 0.0}'/> \n"
      "      <Parameter  name='Sides'    type='string'        value='x+'/>              \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList  name='Essential Boundary Conditions'>                          \n"
      "    <ParameterList  name='X Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='0'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='1'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "    <ParameterList  name='Z Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='2'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "</ParameterList>                                                                 \n"
    );

    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    Plato::exp::Problem<Plato::exp::PhysicsMechanics<Plato::Tet10>>
        tElasticityProblem(tMesh, *tParamList, tMachine);

    auto tError = Plato::test_criterion_grad_wrt_control(tElasticityProblem, tMesh, "Volume");
    TEST_ASSERT(tError < 1e-4);
}

TEUCHOS_UNIT_TEST( Morphorm, TestInternalElasticEnergyPlusBodyForcesGradZ_3D_TET10 )
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", tMeshWidth);

    // create input
    //
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                             \n"
      "  <ParameterList name='Spatial Model'>                                           \n"
      "    <ParameterList name='Domains'>                                               \n"
      "      <ParameterList name='Design Volume'>                                       \n"
      "        <Parameter name='Element Block' type='string' value='body'/>             \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
      "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
      "  <ParameterList name='Criteria'>                                                \n"
      "    <ParameterList name='Internal Elastic Energy'>                               \n"
      "      <Parameter name='Type' type='string' value='Scalar Function'/>             \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
      "      <ParameterList name='Penalty Function'>                                    \n"
      "        <Parameter name='Exponent' type='double' value='1.0'/>                   \n"
      "        <Parameter name='Minimum Value' type='double' value='0.0'/>              \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>                      \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList name='Elliptic'>                                                \n"
      "    <ParameterList name='Penalty Function'>                                      \n"
      "      <Parameter name='Exponent' type='double' value='1.0'/>                     \n"
      "      <Parameter name='Minimum Value' type='double' value='0.0'/>                \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                        \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList name='Material Models'>                                         \n"
      "    <ParameterList name='Unobtainium'>                                           \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
      "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
      "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>           \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList  name='Body Loads'>                                             \n"
      "    <ParameterList name='y Component'>                                           \n"
      "      <Parameter  name='Function' type='string' value='0.0-2700.0*9.81*0.5'/>    \n"
      "      <Parameter  name='Index' type='int' value='1'/>                            \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                            \n"
      "    <ParameterList  name='Traction Vector Boundary Condition'>                   \n"
      "      <Parameter  name='Type'     type='string'        value='Uniform'/>         \n"
      "      <Parameter  name='Values'   type='Array(double)' value='{1.0, 0.0, 0.0}'/> \n"
      "      <Parameter  name='Sides'    type='string'        value='x+'/>              \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList  name='Essential Boundary Conditions'>                          \n"
      "    <ParameterList  name='X Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='0'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='1'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "    <ParameterList  name='Z Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='2'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "</ParameterList>                                                                 \n"
    );

    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    Plato::exp::Problem<Plato::exp::PhysicsMechanics<Plato::Tet10>>
        tElasticityProblem(tMesh, *tParamList, tMachine);

    auto tError = Plato::test_criterion_grad_wrt_control(tElasticityProblem, tMesh, "Internal Elastic Energy");
    TEST_ASSERT(tError < 1e-4);
}

TEUCHOS_UNIT_TEST( Morphorm, ThermostaticResidual3D )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Physics' type='string' value='Thermostatics'/>             \n"
    "  <ParameterList name='Elliptic'>                                             \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Thermal Conduction'>                               \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='100.0'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create test mesh
  //
  constexpr int tMeshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> tHostControl( tMesh->NumNodes(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    tHostControlView(tHostControl.data(),tHostControl.size());
  auto tControl = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), tHostControlView);


  // create mesh based temperature from host data
  //
  std::vector<Plato::Scalar> tHostState( tMesh->NumNodes() );
  Plato::Scalar tTemp = 0.0, tDval = 0.1;
  for( auto& tVal : tHostState ) tVal = (tTemp += tDval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    tHostView(tHostState.data(),tHostState.size());
  auto tState = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), tHostView);

  // create database
  //
  Plato::Database tDatabase;
  tDatabase.vector("states"  , tState);
  tDatabase.vector("controls", tControl);

  // create residual evaluator
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  using PhysicsType = typename Plato::exp::PhysicsThermal<Plato::Tet4>;
  Plato::exp::VectorFunction<PhysicsType>
    tResidualFunction(tParamList->get<std::string>("PDE Constraint"), tSpatialModel, tDataMap, *tParamList);


  // compute and test residual
  //
  auto tResidual = tResidualFunction.value(tDatabase,/*cycle=*/0.0);

  auto tHostResidual = Kokkos::create_mirror_view( tResidual );
  Kokkos::deep_copy( tHostResidual, tResidual );

  std::vector<Plato::Scalar> tGoldResidual = {
  -21.66666666666666, -30.00000000000000, -8.333333333333332,
  -25.00000000000000, -45.00000000000001, -19.99999999999999,
  -3.333333333333332, -15.00000000000000, -11.66666666666667,
  -10.00000000000001, -15.00000000000000, -4.999999999999993,
  -5.000000000000004,  0.000000000000000,  4.999999999999980,
   5.000000000000002,  15.00000000000001,  9.99999999999999,
   11.66666666666667,  14.99999999999999,  3.333333333333336,
   20.00000000000000,  45.00000000000005,  25.00000000000000,
   8.333333333333321,  29.99999999999999,  21.66666666666667
  };

  for(int iNode=0; iNode<int(tGoldResidual.size()); iNode++){
    if(tGoldResidual[iNode] == 0.0){
      TEST_ASSERT(fabs(tHostResidual[iNode]) < 1e-11);
    } else {
      TEST_FLOATING_EQUALITY(tHostResidual[iNode], tGoldResidual[iNode], 1e-13);
    }
  }


  // compute and test constraint gradient wrt state, tState. (i.e., jacobian)
  //
  auto tJacobian = tResidualFunction.jacobianState(tDatabase,/*cycle=*/0.0);

  auto tJacobianEntries = tJacobian->entries();
  auto tHostJacobianEntries = Kokkos::create_mirror_view( tJacobianEntries );
  Kokkos::deep_copy(tHostJacobianEntries, tJacobianEntries);

  std::vector<Plato::Scalar> tGoldJacobianEntries = {
   49.9999999999999858, -16.6666666666666643, -16.6666666666666643, 0,
  -16.6666666666666643, 0, 0, 0, -16.6666666666666643,
   83.3333333333333002, -16.6666666666666643, -24.9999999999999964, 0,
  -24.9999999999999964, 0, 0, 0, -16.6666666666666643,
   33.3333333333333286, -8.33333333333333215, -8.33333333333333215, 0
  };

  int tJacobianEntriesSize = tGoldJacobianEntries.size();
  for(int i=0; i<tJacobianEntriesSize; i++){
    TEST_FLOATING_EQUALITY(tHostJacobianEntries(i), tGoldJacobianEntries[i], 1.0e-15);
  }


  // compute and test gradient wrt control, control
  //
  auto tJacobianControl = tResidualFunction.jacobianControl(tDatabase,/*cycle=*/0.0);

  auto tJacobianControlEntries = tJacobianControl->entries();
  auto tHostJacobianControlEntries = Kokkos::create_mirror_view( tJacobianControlEntries );
  Kokkos::deep_copy(tHostJacobianControlEntries, tJacobianControlEntries);

  std::vector<Plato::Scalar> tGoldJacobianControlEntries = {
   -5.41666666666666607, -2.08333333333333304, -0.833333333333333259,
   -2.91666666666666696,  2.91666666666666563,  0.833333333333333037,
    2.08333333333333304,  5.41666666666666785, -0.416666666666666630,
   -7.50000000000000000, -2.08333333333333304, -2.08333333333333348,
   -2.91666666666666563,  4.16666666666666607,  0.833333333333334370,
    4.58333333333333304,  5.41666666666666519, -0.416666666666666741,
   -2.08333333333333304, -1.24999999999999956,  1.25000000000000044,
    2.49999999999999911
  };

  int tJacobianControlEntriesSize = tGoldJacobianControlEntries.size();
  for(int i=0; i<tJacobianControlEntriesSize; i++){
    TEST_FLOATING_EQUALITY(tHostJacobianControlEntries(i), tGoldJacobianControlEntries[i], 1.0e-14);
  }

  // compute and test gradient wrt node position, x
  //
  auto tJacobianConfig = tResidualFunction.jacobianConfig(tDatabase,/*cycle=*/0.0);

  auto tJacobianConfigEntries = tJacobianConfig->entries();
  auto tHostJacobianConfigEntries = Kokkos::create_mirror_view( tJacobianConfigEntries );
  Kokkos::deep_copy(tHostJacobianConfigEntries, tJacobianConfigEntries);

  std::vector<Plato::Scalar> tGoldJacobianConfigEntries = {
   -90.0000000000000000, -29.9999999999999929, -10.0000000000000000,
    28.3333333333333357,  8.33333333333332860,  23.3333333333333321,
    25.0000000000000000,  26.6666666666666679, -1.66666666666667052,
   -6.66666666666666696,  15.0000000000000018,  15.0000000000000018
  };

  int tJacobianConfigEntriesSize = tGoldJacobianConfigEntries.size();
  for(int i=0; i<tJacobianConfigEntriesSize; i++){
    if(fabs(tGoldJacobianConfigEntries[i]) < 1e-10){
      TEST_ASSERT(fabs(tHostJacobianConfigEntries[i]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(tHostJacobianConfigEntries(i), tGoldJacobianConfigEntries[i], 1.0e-13);
    }
  }
}

/******************************************************************************/
/*!
  \brief Compute value and both gradients (wrt state and control) of
         InternalThermalEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Morphorm, InternalThermalEnergy3D )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Physics' type='string' value='Thermostatics'/>             \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Internal Thermal Energy'>                            \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Thermal Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Thermal Conduction'>                               \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='100.0'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create test mesh
  //
  constexpr int tMeshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> tHostControl( tMesh->NumNodes(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    tHostControlView(tHostControl.data(),tHostControl.size());
  auto tControl = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), tHostControlView);

  // create mesh based temperature from host data
  //
  Plato::OrdinalType tNumDofs = tMesh->NumNodes();
  Plato::ScalarVector tState("states", tNumDofs);
  auto tHostState = Kokkos::create_mirror_view( tState );
  Plato::Scalar tTemp = 0.0, tDval = 0.1;
  for(Plato::OrdinalType i=0; i<tNumDofs; i++)
  {
      tHostState(i) = (tTemp += tDval);
  }
  Kokkos::deep_copy(tState, tHostState);

  // create database
  //
  Plato::Database tDatabase;
  tDatabase.vector("states"  , tState);
  tDatabase.vector("controls", tControl);

  // create criterion
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  std::string tMyFunctionName("Internal Thermal Energy");
  Plato::exp::ScalarFunction<Plato::exp::PhysicsThermal<Plato::Tet4>>
      tScalarFunction(tMyFunctionName, tSpatialModel, tDataMap, *tParamList);

  // compute and test criterion value
  //
  auto tValue = tScalarFunction.value(tDatabase, /*cycle=*/0.0);

  Plato::Scalar tGoldValue = 363.999999999999829;
  TEST_FLOATING_EQUALITY(tValue, tGoldValue, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  auto tGradState = tScalarFunction.gradientState(tDatabase, /*cycle=*/0.0);

  auto tHostGradState = Kokkos::create_mirror_view( tGradState );
  Kokkos::deep_copy( tHostGradState, tGradState );

  std::vector<Plato::Scalar> tGoldGradState = {
  -43.33333333333333, -60.00000000000000, -16.66666666666666,
  -49.99999999999999, -90.00000000000001, -39.99999999999999,
  -6.666666666666664, -30.00000000000000, -23.33333333333334,
  -20.00000000000001, -29.99999999999999, -9.99999999999999,
  -10.00000000000001,  0.000000000000000,  9.99999999999996,
   10.00000000000000,  30.00000000000003,  19.99999999999998,
   23.33333333333334,  29.99999999999996,  6.666666666666671,
   39.99999999999999,  90.00000000000009,  50.00000000000000,
   16.66666666666664,  59.99999999999997,  43.33333333333335
  };

  for(int iNode=0; iNode<int(tGoldGradState.size()); iNode++){
    if(tGoldGradState[iNode] == 0.0){
      TEST_ASSERT(fabs(tHostGradState[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tHostGradState[iNode], tGoldGradState[iNode], 1e-13);
    }
  }


  // compute and test criterion gradient wrt control, control
  //
  auto tGradControl = tScalarFunction.gradientControl(tDatabase, /*cycle=*/0.0);

  auto tHostGradControl = Kokkos::create_mirror_view( tGradControl );
  Kokkos::deep_copy( tHostGradControl, tGradControl );

  std::vector<Plato::Scalar> tGoldGradControl = {
  11.37500000000000, 15.16666666666666, 3.791666666666666,
  15.16666666666667, 22.74999999999999, 7.583333333333331,
  3.791666666666667, 7.583333333333333, 3.791666666666666,
  15.16666666666667, 22.75000000000000, 7.583333333333334,
  22.75000000000000, 45.50000000000001, 22.75000000000000,
  7.583333333333332, 22.75000000000000, 15.16666666666667,
  3.791666666666667, 7.583333333333337, 3.791666666666667,
  7.583333333333334, 22.75000000000001, 15.16666666666667,
  3.791666666666666, 15.16666666666667, 11.37500000000001
  };

  for(int iNode=0; iNode<int(tGoldGradControl.size()); iNode++){
    TEST_FLOATING_EQUALITY(tHostGradControl[iNode], tGoldGradControl[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto tGradConfig = tScalarFunction.gradientConfig(tDatabase, /*cycle=*/0.0);

  auto tHostGradConfig = Kokkos::create_mirror_view( tGradConfig );
  Kokkos::deep_copy(tHostGradConfig, tGradConfig);

  std::vector<Plato::Scalar> tGoldGradConfig = {
  47.66666666666666, -4.333333333333325, -21.66666666666667,
  62.50000000000003, -9.500000000000027,  12.00000000000000,
  14.83333333333334, -5.166666666666668,  33.66666666666667,
  44.50000000000000,  29.99999999999999, -35.50000000000000,
  71.00000000000001,  54.00000000000001,  18.00000000000001,
  26.49999999999999,  24.00000000000000,  53.50000000000001,
 -3.166666666666664,  34.33333333333334, -13.83333333333333,
  8.499999999999988,  63.50000000000001,  5.999999999999993,
  11.66666666666666,  29.16666666666667,  19.83333333333333,
  36.00000000000003, -33.49999999999999, -41.50000000000001,
  53.99999999999994, -73.00000000000003,  6.000000000000016,
  17.99999999999999, -39.50000000000001,  47.50000000000001
  };

  for(int iNode=0; iNode<int(tGoldGradConfig.size()); iNode++){
    TEST_FLOATING_EQUALITY(tHostGradConfig[iNode], tGoldGradConfig[iNode], 1e-13);
  }
}


TEUCHOS_UNIT_TEST( Morphorm, InternalThermalEnergyGrad_3D_TET4 )
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    // create input
    //
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                          \n"
      "  <ParameterList name='Spatial Model'>                                        \n"
      "    <ParameterList name='Domains'>                                            \n"
      "      <ParameterList name='Design Volume'>                                     \n"
      "        <Parameter name='Element Block' type='string' value='body'/>          \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
      "      </ParameterList>                                                        \n"
      "    </ParameterList>                                                          \n"
      "  </ParameterList>                                                            \n"
      "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
      "  <Parameter name='Physics' type='string' value='Thermostatics'/>             \n"
      "  <ParameterList name='Criteria'>                                             \n"
      "    <ParameterList name='Internal Thermal Energy'>                            \n"
      "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Internal Thermal Energy'/>  \n"
      "      <ParameterList name='Penalty Function'>                                 \n"
      "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
      "        <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
      "      </ParameterList>                                                        \n"
      "    </ParameterList>                                                          \n"
      "  </ParameterList>                                                            \n"
      "  <ParameterList name='Material Models'>                                      \n"
      "    <ParameterList name='Unobtainium'>                                        \n"
      "      <ParameterList name='Thermal Conduction'>                               \n"
      "        <Parameter name='Thermal Conductivity' type='double' value='100.0'/>  \n"
      "      </ParameterList>                                                        \n"
      "    </ParameterList>                                                          \n"
      "  </ParameterList>                                                            \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                         \n"
      "    <ParameterList  name='Flux Boundary Condition'>                           \n"
      "      <Parameter  name='Type'     type='string'   value='Uniform'/>           \n"
      "      <Parameter  name='Value'    type='double'   value='1.0'/>               \n"
      "      <Parameter  name='Sides'    type='string'   value='x+'/>                \n"
      "    </ParameterList>                                                          \n"
      "  </ParameterList>                                                            \n"
      "  <ParameterList  name='Essential Boundary Conditions'>                       \n"
      "    <ParameterList  name='Fixed Temperature Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>          \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                  \n"
      "    </ParameterList>                                                          \n"
      "  </ParameterList>                                                            \n"
      "</ParameterList>                                                              \n"
    );

    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    Plato::exp::Problem<Plato::exp::PhysicsThermal<Plato::Tet4>> tThermalProblem(tMesh, *tParamList, tMachine);

    auto tError = Plato::test_criterion_grad_wrt_control(tThermalProblem, tMesh, "Internal Thermal Energy");
    TEST_ASSERT(tError < 1e-4);
}


TEUCHOS_UNIT_TEST(Morphorm, Elastostatics_Nitsche)
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", tMeshWidth);

    // create input
    //
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                             \n"
      "  <ParameterList name='Spatial Model'>                                           \n"
      "    <ParameterList name='Domains'>                                               \n"
      "      <ParameterList name='Design Volume'>                                       \n"
      "        <Parameter name='Element Block' type='string' value='body'/>             \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
      "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
      "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                      \n"
      "  <ParameterList name='Elliptic'>                                                \n"
      "    <ParameterList name='Penalty Function'>                                      \n"
      "      <Parameter name='Exponent' type='double' value='1.0'/>                     \n"
      "      <Parameter name='Minimum Value' type='double' value='0.0'/>                \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                        \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList name='Material Models'>                                         \n"
      "    <ParameterList name='Unobtainium'>                                           \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
      "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
      "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>           \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                            \n"
      "    <ParameterList  name='Traction Vector Boundary Condition'>                   \n"
      "      <Parameter  name='Type'     type='string'        value='Uniform'/>         \n"
      "      <Parameter  name='Values'   type='Array(double)' value='{1.0, 0.0, 0.0}'/> \n"
      "      <Parameter  name='Sides'    type='string'        value='x+'/>              \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList  name='Essential Boundary Conditions'>                          \n"
      "    <ParameterList  name='X Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='0'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='1'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "    <ParameterList  name='Z Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='2'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList  name='Nitsche Boundary Conditions'>                            \n"
      "    <ParameterList  name='Mechanical Nitsche Boundary Conditions'>                   \n"
      "      <Parameter  name='Material Model'   type='string' value='Unobtainium'/> \n"
      "      <Parameter  name='Variable'   type='string' value='displacements'/> \n"
      "      <Parameter  name='Sides'    type='string'        value='x-'/>              \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "</ParameterList>                                                                 \n"
    );

    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    Plato::exp::Problem<Plato::exp::PhysicsMechanics<Plato::Tet10>>
        tElasticityProblem(tMesh, *tParamList, tMachine);

    // SOLVE ELASTOSTATICS EQUATIONS
    auto tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    auto tElasticitySolution = tElasticityProblem.solution(tControl);
    //tElasticityProblem.output("output_weak");

    // TEST RESULTS
    constexpr Plato::OrdinalType tTimeStep = 0;
    auto tState = tElasticitySolution.get("State");
    auto tSolution = Kokkos::subview(tState, tTimeStep, Kokkos::ALL());
    auto tHostSolution = Kokkos::create_mirror_view(tSolution);
    Kokkos::deep_copy(tHostSolution, tSolution);

    std::vector<Plato::Scalar> tGold =
        {9.22995e-8, 1.0634e-6, -6.65862e-08, 1.18648e-8,
         1.06597e-6, -6.93321e-8, -6.76512e-8,
         1.07269e-6, -6.88842e-8, -1.44575e-7,
         1.06555e-6, -1.37668e-7, 1.69763e-7,
         1.06655e-6, -1.39516e-7, 9.30324e-8,
         1.07056e-6, -1.44331e-7, 1.14471e-8,
         1.07329e-6, -1.46262e-7, -6.73653e-8,
         1.08639e-6, -1.49161e-7, -1.47672e-7};

    constexpr Plato::Scalar tTolerance = 1e-4;
    constexpr Plato::OrdinalType tDofOffset = 350; // comparing only the last 25 dofs
    for(Plato::OrdinalType tDofIndex=0; tDofIndex < tGold.size(); tDofIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostSolution(tDofOffset+tDofIndex), tGold[tDofIndex], tTolerance);
    }
}

}
// namespace IfemTests
