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

// immersus includes
#include "Simp.hpp"
#include "FadTypes.hpp"
#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "SurfaceArea.hpp"
#include "PlatoMathExpr.hpp"
#include "SpatialModel.hpp"
#include "SmallStrain.hpp"
#include "LinearStress.hpp"
#include "PlatoMeshExpr.hpp"
#include "GradientMatrix.hpp"
#include "PlatoUtilities.hpp"
#include "ApplyWeighting.hpp"
#include "PlatoStaticsTypes.hpp"
#include "ElasticModelFactory.hpp"
#include "GeneralStressDivergence.hpp"

namespace Plato
{

template<typename Type>
struct Range
{
private:
    std::unordered_map<std::string, Type> mRange; /*!< map from data name to two-dimensional array of pod type */
    std::unordered_map<std::string, Plato::OrdinalType> mDataID2NumDofs; /*!< map from data name to number of degrees of freedom */
    std::unordered_map<std::string, std::vector<std::string>> mDataID2DofNames; /*!< map from data name to degrees of freedom names */

public:
    Range(){};
    ~Range(){}

    std::vector<std::string> tags() const
    {
        std::vector<std::string> tTags;
        for(auto& tPair : mRange)
        {
            tTags.push_back(tPair.first);
        }
        return tTags;
    }
    Type get(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mRange.find(tLowerTag);
        if(tItr == mRange.end())
        {
            ANALYZE_THROWERR(std::string("Data with tag '") + aTag + "' is not defined in Range associative map")
        }
        return tItr->second;
    }
    void set(const std::string& aTag, const Type& aData)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mRange[tLowerTag] = aData;
    }
    Plato::OrdinalType dofs(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mDataID2NumDofs.find(tLowerTag);
        if(tItr == mDataID2NumDofs.end())
        {
            ANALYZE_THROWERR(std::string("Data with tag '") + aTag + "' is not defined in Range associative map")
        }
        return tItr->second;
    }
    std::vector<std::string> dof_names(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mDataID2DofNames.find(tLowerTag);
        if(tItr == mDataID2DofNames.end())
        {
            return std::vector<std::string>(0);
        }
        return tItr->second;
    }
    void print() const
    {
        if(mRange.empty())
        { return; }
        for(auto& tPair : mRange)
        { Plato::print_array_2D(tPair.second, tPair.first); }
    }
    bool empty() const
    {
        return mRange.empty();
    }
};
// struct Range

struct Domain
{
    std::unordered_map<std::string,Plato::Scalar> scalars; /*!< map to scalar quantities of interest */
    std::unordered_map<std::string,Plato::ScalarVector> vectors; /*!< map to scalar quantities of interest */
};


/******************************************************************************//**
 * \brief Factory for creating linear elastic material models.
 *
 * \tparam SpatialDim spatial dimensions: options 1D, 2D, and 3D
 *
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class ElasticMaterialFactory
{
public:
    /******************************************************************************//**
    * \brief Linear elastic material model factory constructor.
    * \param [in] aParamList input parameter list
    **********************************************************************************/
    ElasticMaterialFactory(const Teuchos::ParameterList& aParamList) :
        mParamList(aParamList){}

    /******************************************************************************//**
    * \brief Create a linear elastic material model.
    * \param [in] aModelName name of the model to be created.
    * \return Teuchos reference counter pointer to linear elastic material model
    **********************************************************************************/
    std::shared_ptr<Plato::LinearElasticMaterial<SpatialDim>>
    create(std::string aModelName)
    {
        if (!mParamList.isSublist("Material Models"))
        {
            REPORT("'Material Models' list not found! Returning 'nullptr'");
            return std::make_shared<Plato::LinearElasticMaterial<SpatialDim>>(nullptr);
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
                return std::make_shared<Plato::IsotropicLinearElasticMaterial<SpatialDim>>(tModelParamList.sublist("Isotropic Linear Elastic"));
            }
            else if(tModelParamList.isSublist("Orthotropic Linear Elastic"))
            {
                return std::make_shared<Plato::OrthotropicLinearElasticMaterial<SpatialDim>>(tModelParamList.sublist("Orthotropic Linear Elastic"));
            }
            return std::make_shared<Plato::LinearElasticMaterial<SpatialDim>>(nullptr);
        }
    }

private:
    const Teuchos::ParameterList& mParamList; /*!< Input parameter list */
};
// class ElasticModelFactory

enum class volume_force_t
{
    BODYLOAD,
    UNDEFINED
};

template<typename EvaluationType>
class AbstractVolumeForce
{
public:
    virtual ~AbstractVolumeForce(){}
    virtual Plato::volume_force_t category() const = 0;
    virtual void evaluate
    (const Plato::SpatialDomain & aSpatialDomain,
     const Plato::WorkSets      & aWorkSets,
           Plato::Scalar          aScale,
           Plato::Scalar          aCurrentTime) const=0;

    std::unordered_map<volume_force_t,std::string> AM =
        { {Plato::volume_force_t::BODYLOAD,"Body Load"} };
};

/******************************************************************************/
/*!
 \brief Class for essential boundary conditions.
 */
template<typename EvaluationType>
class BodyLoad : public Plato::AbstractVolumeForce<EvaluationType>
/******************************************************************************/
{
private:
    // set local element type definition
    using ElementType = typename EvaluationType::ElementType;

    // set local fad type definitions
    using ResultFadType  = typename EvaluationType::ResultScalarType;
    using ConfigFadType  = typename EvaluationType::ConfigScalarType;
    using ControlFadType = typename EvaluationType::ControlScalarType;

    // set local static types
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    static constexpr Plato::OrdinalType mNumDofsPerNode = ElementType::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumNodesPerCell = ElementType::mNumNodesPerCell; /*!< number of nodes per cell/element */

    const std::string mName;
    const std::string mFunction;
    const Plato::OrdinalType mDof;

public:
    BodyLoad<EvaluationType>(const std::string &aName, Teuchos::ParameterList &aProbParam) :
            mName(aName),
            mDof(aProbParam.get<Plato::OrdinalType>("Index", 0)),
            mFunction(aProbParam.get<std::string>("Function"))
    {
    }

    Plato::volume_force_t category() const { return Plato::volume_force_t::BODYLOAD; }

    void evaluate
    (const Plato::SpatialDomain & aSpatialDomain,
     const Plato::WorkSets      & aWorkSets,
           Plato::Scalar          aScale,
           Plato::Scalar          aCurrentTime) const
    {
        // get input worksets (i.e., domain for function evaluate)
        auto tConfig  = Plato::metadata<Plato::ScalarArray3DT<ConfigFadType>>(aWorkSets.get("configuration"));
        auto tResult  = Plato::metadata<Plato::ScalarMultiVectorT<ResultFadType>>(aWorkSets.get("result"));
        auto tControl = Plato::metadata<Plato::ScalarMultiVectorT<ControlFadType>>(aWorkSets.get("control"));

        // get integration points and weights
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        // map points to physical space
        Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
        Plato::ScalarArray3DT<ConfigFadType> tPhysicalPoints("cub points physical space", tNumCells, tNumPoints, mSpaceDim);
        Plato::mapPoints<ElementType>(tConfig, tPhysicalPoints);

        // get integrand values at quadrature points
        Plato::ScalarMultiVectorT<ConfigFadType> tFxnValues("function values", tNumCells*tNumPoints, 1);
        Plato::getFunctionValues<mSpaceDim>(tPhysicalPoints, mFunction, tFxnValues);

        // integrate and assemble
        //
        auto tDof = mDof;
        Plato::VectorEntryOrdinal<mSpaceDim, mSpaceDim> tVectorEntryOrdinal(aSpatialDomain.Mesh);
        Kokkos::parallel_for("compute body load", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tDetJ = Plato::determinant(ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal));

            ControlFadType tDensity(0.0);
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < ElementType::mNumNodesPerCell; tFieldOrdinal++)
            {
                tDensity += tBasisValues(tFieldOrdinal)*tControl(iCellOrdinal, tFieldOrdinal);
            }

            auto tEntryOffset = iCellOrdinal * tNumPoints;

            auto tFxnValue = tFxnValues(tEntryOffset + iGpOrdinal, 0);
            auto tWeight = aScale * tCubWeights(iGpOrdinal) * tDetJ;
            for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < ElementType::mNumNodesPerCell; tFieldOrdinal++)
            {
                Kokkos::atomic_add(&tResult(iCellOrdinal,tFieldOrdinal*mNumDofsPerNode+tDof),
                        tWeight * tFxnValue * tBasisValues(tFieldOrdinal) * tDensity);
            }
        });
    }
};
// end class BodyLoad

/******************************************************************************/
/*!
 \brief Contains list of BodyLoad objects.
 */
template<typename EvaluationType>
class VolumeForces
/******************************************************************************/
{
private:
    std::unordered_map<Plato::volume_force_t,std::vector< std::shared_ptr< AbstractVolumeForce<EvaluationType> > > > mVolumeForces;

public:

    /******************************************************************************//**
     * \brief Constructor that parses and creates a vector of BodyLoad objects based on
     *   the ParameterList.
     * \param aParams Body Loads sublist with input parameters
    **********************************************************************************/
    VolumeForces(Teuchos::ParameterList& aParams) :
        mVolumeForces()
    {
        this->bodyloads(aParams);
    }

    std::vector< std::shared_ptr< AbstractVolumeForce<EvaluationType> > >&
    get(Plato::volume_force_t aType) const
    {
        auto tItr = mVolumeForces.find(aType);
        if(tItr == mVolumeForces.end()){
            ANALYZE_THROWERR("Did not find requested volume force");
        }
        return tItr->second;
    }

    /**************************************************************************/
    /*!
     \brief Add the body load to the result workset
     */
    void evaluate
    (const Plato::SpatialDomain & aSpatialDomain,
     const Plato::WorkSets      & aWorkSets,
           Plato::Scalar          aScale = 1.0,
           Plato::Scalar          aCurrentTime = 0.0) const
    {
        for(const auto & tForce : mVolumeForces)
        {
            tForce.second->evaluate(aSpatialDomain, aWorkSets, aScale, aCurrentTime);
        }
    }

private:
    void bodyloads(Teuchos::ParameterList& aParams)
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
            auto tNewVolumeForce = std::make_shared<Plato::BodyLoad<EvaluationType>>(tName, tSublist);
            Plato::volume_force_t tType = tNewVolumeForce.category();
            mVolumeForces[tType].push_back(tNewVolumeForce);
        }
    }
};

enum class natural_bc_t
{
    SURFACE_FORCE,
    UNDEFINED
};

template<typename EvaluationType>
class AbstractNaturalBC
{
public:
    virtual ~AbstractNaturalBC(){}
    virtual Plato::natural_bc_t type() const = 0;
    virtual void evaluate
    (const Plato::SpatialDomain & aSpatialDomain,
     const Plato::WorkSets      & aWorkSets,
           Plato::Scalar          aScale = 1.0,
           Plato::Scalar          aCurrentTime = 0.0) const = 0;
};

template<typename EvaluationType>
class SurfaceForce : public Plato::AbstractNaturalBC<EvaluationType>
{
private:
    // set local element type definition
    using ElementType = typename EvaluationType::ElementType;

    // set local fad type definitions
    using ResultFadType  = typename EvaluationType::ResultScalarType;
    using ConfigFadType  = typename EvaluationType::ConfigScalarType;

    // set local parameter values
    using ElementType::mNumSpatialDims;
    using ElementType::mNumDofsPerNode;

    // allocate local member data
    const std::string mType;         /*!< natural boundary condition type */
    const std::string mBlockName;    /*!< input block of force parameters */
    const std::string mSideSetName;  /*!< side set name */
    const Plato::Array<mNumSpatialDims> mForceCoef; /*!< force coefficients */

    // allocate local member instances
    std::shared_ptr<Plato::MathExpr> mForceCoefExpr[mNumSpatialDims];

private:
    void initialize(Teuchos::ParameterList & aSubList)
    {
        auto tIsValue = aSubList.isType<Teuchos::Array<Plato::Scalar>>("Vector");
        auto tIsExpr  = aSubList.isType<Teuchos::Array<std::string>>("Vector");
        if (tIsValue)
        {
            auto tForceVal = aSubList.get<Teuchos::Array<Plato::Scalar>>("Vector");
            for(Plato::OrdinalType tDim=0; tDim<mNumSpatialDims; tDim++)
            {
                mForceCoef(tDim) = tForceVal[tDim];
            }
        }
        else
        if (tIsExpr)
        {
            auto tExpression = aSubList.get<Teuchos::Array<std::string>>("Vector");
            for(Plato::OrdinalType tDim=0; tDim<mNumSpatialDims; tDim++)
            {
                mForceCoefExpr[tDim] = std::make_shared<Plato::MathExpr>(tExpression[tDim]);
                mForceCoef(tDim) = mForceCoefExpr[tDim]->value(0.0);
            }
        }
    }
    void evalForceExpr(Plato::Scalar aCurrentTime)
    {
        for(int iDim=0; iDim<mNumSpatialDims; iDim++)
        {
            if(mForceCoefExpr[iDim])
            {
                mForceCoef(iDim) = mForceCoefExpr[iDim]->value(aCurrentTime);
            }
        }
    }

public:
    SurfaceForce
    (const std::string            & aBlockName,
           Teuchos::ParameterList & aSubList) :
        mBlockName(aBlockName),
        mType(aSubList.get<std::string>("Type")),
        mSideSetName(aSubList.get<std::string>("Sides")),
        mForceCoefExpr{nullptr}
    {
        this->initialize(aSubList)
    }
    Plato::natural_bc_t type() const
    {
        return Plato::natural_bc_t::SURFACE_FORCE;
    }
    void evaluate
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets     & aWorkSets,
           Plato::Scalar         aScale,
           Plato::Scalar         aCurrentTime)
    {
        this->evalForceExpr(aCurrentTime);

        // get input worksets (i.e., domain for function evaluate)
        auto tResult = Plato::metadata<Plato::ScalarMultiVectorT<ResultFadType>>(aWorkSets.get("result"));
        auto tConfig = Plato::metadata<Plato::ScalarArray3DT<ConfigFadType>>(aWorkSets.get("configuration"));
        // get side set connectivity
        auto tElementOrds = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
        auto tNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
        Plato::OrdinalType tNumFaces = tElementOrds.size();

        // create surface area functor
        Plato::SurfaceArea<ElementType> tComputeSurfaceArea;

        // get integration points and weights
        auto tForceCoef = mForceCoef;
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
          tComputeSurfaceArea(tElementOrdinal, tLocalNodeOrds, tBasisGrads, aConfig, tSurfaceArea);
          tSurfaceArea *= aScale;
          tSurfaceArea *= tCubatureWeight;

          // project into aResult workset
          for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
          {
              for( Plato::OrdinalType tDim=0; tDim<ElementType::mNumSpatialDims; tDim++)
              {
                  auto tElementDofOrdinal = tLocalNodeOrds[tNode] * ElementType::mNumDofsPerNode + tDim;
                  ResultScalarType tResult = tBasisValues(tNode)*tForceCoef[tDim]*tSurfaceArea;
                  Kokkos::atomic_add(&aResult(tElementOrdinal,tElementDofOrdinal), tResult);
              }
          }
        }, "surface force");
    }
};

enum class natural_bc_category_t
{
    UNIFORM,
    UNDEFINED
};

template<typename EvaluationType>
class NaturalBCs
{
// private member data
private:
    /*!< list of natural boundary condition */
    std::unordered_map<Plato::natural_bc_t,std::vector<std::shared_ptr<Plato::AbstractNaturalBC<EvaluationType> > > > mBCs;

public:
    NaturalBCs(Teuchos::ParameterList &aParams) :
        mBCs()
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
            // TODO: IMPLEMENT FACTORY
            this->flux(tName, tSubList);
        }
    }

    std::shared_ptr<Plato::SurfaceForce<EvaluationType>>
    flux(const std::string & aName, Teuchos::ParameterList &aSubList)
    {
        bool tValuesKeyword = aSubList.isType<Teuchos::Array<Plato::Scalar>>("Values");
        if (tValuesKeyword)
        {
            std::stringstream tMsg;
            tMsg << "'Values' parameter keyword is not defined in Natural Boundary Condition "
                << "Parameter Sublist with name '" << aName.c_str() << "'";
            ANALYZE_THROWERR(tMsg.str().c_str())
        }
        else if (tValuesKeyword)
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
                std::stringstream tMsg;
                tMsg << "Unexpected type encountered for 'Values' parameter keyword. "
                     << "Specify 'type' of 'Array(double)' or 'Array(string)'.";
                ANALYZE_THROWERR(tMsg.str().c_str())
            }
        }
        else
        {
            std::stringstream tMsg;
            tMsg << "Error while parsing surface force in Natural Boundary Condition "
                 << "Parameter Sublist with name '" << aName.c_str() << "'";
            ANALYZE_THROWERR(tMsg.str().c_str())
        }
        auto tBC = std::make_shared<Plato::SurfaceForce<EvaluationType>>(aName, aSubList);
        return tBC;
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
    using ElementType::mNumVoigtTerms; /*!< number of stress/strain terms */
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

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
    CauchyStress(const Plato::LinearElasticMaterial<mSpaceDim> & aMaterial) :
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

template<typename EvaluationType>
class AbstractResidual
{
private:
    using ResultFadType = typename EvaluationType::ResultScalarType;

protected:
    const Plato::SpatialDomain     & mSpatialDomain;  /*!< Plato spatial model containing mesh, meshsets, etc */
          Plato::DataMap           & mDataMap;        /*!< Plato Analyze database */
          std::vector<std::string>   mDofNames;       /*!< state dof names */

public:
    explicit AbstractResidual
    (const Plato::SpatialDomain & aSpatialDomain,
           Plato::DataMap       & aDataMap) :
        mSpatialDomain(aSpatialDomain),
        mDataMap(aDataMap)
    {}
    virtual ~AbstractResidual()
    {}

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate inner domain residual, exclude boundary terms.
     * \param [in] aWorkSets holds state and control worksets
     * \param [in/out] aResult   result workset
     ******************************************************************************/
    virtual void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultFadType> &aResult) const = 0;

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate boundary forces, not related to any prescribed boundary force,
     *        resulting from applying integration by part to the residual equation.
     * \param [in]  aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in]  aWorkSets     holds input worksets (e.g. states, control, etc)
     * \param [out] aResultWS     result/output workset
     ******************************************************************************/
    virtual void evaluate_boundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets     & aWorkSets,
     Plato::ScalarMultiVectorT<ResultFadType> &aResult) const = 0;

    /***************************************************************************//**
     * \fn void evaluatePrescribed
     * \brief Evaluate vector function on prescribed boundaries.
     * \param [in]  aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in]  aWorkSets     holds input worksets (e.g. states, control, etc)
     * \param [out] aResultWS     result/output workset
     ******************************************************************************/
    virtual void evaluate_prescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets     & aWorkSets,
     Plato::ScalarMultiVectorT<ResultFadType> &aResult) const = 0;
};
// class abstract residual


template<typename PhysicsType, typename EvaluationType>
class ElastostaticResidual : public Plato::AbstractResidual<EvaluationType>
{
private:
    // set local element type
    using ElementType = typename EvaluationType::ElementType;

    // set local static types
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    // access parent class data structures
    using ResidualBaseType = Plato::AbstractResidual<EvaluationType>;
    using ResidualBaseType::mSpatialDomain;
    using ResidualBaseType::mDataMap;
    using ResidualBaseType::mDofNames;

    // set local fad types
    using StateFadType   = typename EvaluationType::StateScalarType;
    using ResultFadType  = typename EvaluationType::ResultScalarType;
    using ConfigFadType  = typename EvaluationType::ConfigScalarType;
    using ControlFadType = typename EvaluationType::ControlScalarType;

    std::shared_ptr<Plato::VolumeForces<EvaluationType>> mBodyLoads;
    std::shared_ptr<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterial;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, Plato::MSIMP> mApplyWeighting;


public:
    ElastostaticResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aProbParams) :
         ResidualBaseType(aDomain, aDataMap)
    {
        // obligatory: define dof names in order
        mDofNames.push_back("displacement X");
        if(mNumSpatialDims > 1) mDofNames.push_back("displacement Y");
        if(mNumSpatialDims > 2) mDofNames.push_back("displacement Z");

        // create material model and get stiffness
        Plato::ElasticMaterialFactory<mNumSpatialDims> tMaterialFactory(aProbParams);
        mMaterial = tMaterialFactory.create(aDomain.getMaterialName());

        // parse body loads
        if(aProbParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::VolumeForces<EvaluationType>>(aProbParams.sublist("Body Loads"));
        }
    }

    void evaluate
    (const Plato::WorkSets                          & aWorkSets,
           Plato::ScalarMultiVectorT<ResultFadType> & aResult) const
    {
        // set strain fad type
        using StrainFadType = typename Plato::fad_type_t<ElementType, StateFadType, ConfigFadType>;

        // create local worksets
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVectorT<ConfigFadType> tCellVolume("volume", tNumCells);
        Plato::ScalarMultiVectorT<StrainFadType> tCellStrain("strain", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVectorT<ResultFadType> tCellStress("stress", tNumCells, mNumVoigtTerms);

        // create local functors
        Plato::SmallStrain<ElementType>             tComputeVoigtStrain;
        Plato::ComputeGradientMatrix<ElementType>   tComputeGradient;
        Plato::GeneralStressDivergence<ElementType> tComputeStressDivergence;
        Plato::CauchyStress<EvaluationType>         tComputeVoigtStress(mMaterial.operator*());

        // get input worksets (i.e., domain for function evaluate)
        auto tStateWS   = Plato::metadata<Plato::ScalarMultiVectorT<StateFadType>>( aWorkSets.get("state"));
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigFadType>>( aWorkSets.get("configuration"));
        auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlFadType>>( aWorkSets.get("control"));

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
            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigFadType> tGradient;
            Plato::Array<ElementType::mNumVoigtTerms, StrainFadType> tStrain(0.0);
            Plato::Array<ElementType::mNumVoigtTerms, ResultFadType> tStress(0.0);

            // get integration
            auto tCubPoint = tCubPoints(iGpOrdinal);

            // compute strains and stresses for this integration point
            tComputeGradient(iCellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
            tComputeVoigtStrain(iCellOrdinal, tStrain, tStateWS, tGradient);
            tComputeVoigtStress(tStress, tStrain);

            // add contribution to volume from this integration point
            tVolume *= tCubWeights(iGpOrdinal);

            // apply ersatz penalization
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            tApplyWeighting(iCellOrdinal, tControlWS, tBasisValues, tStress);

            // apply divergence to stress
            tComputeStressDivergence(iCellOrdinal, aResult, tStress, tGradient, tVolume);

            // compute cell stress and strain: aggregate stress and strain contribution from each integration point
            for(int i=0; i<ElementType::mNumVoigtTerms; i++)
            {
                Kokkos::atomic_add(&tCellStrain(iCellOrdinal,i), tVolume*tStrain(i));
                Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume*tStress(i));
            }
            // compute cell volume: aggregate volume contribution from each integration
            Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
        });

        // compute cell stress and strain values by multiplying by 1/volume factor
        Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
        {
            for(int i=0; i<ElementType::mNumVoigtTerms; i++)
            {
                tCellStrain(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
                tCellStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
            }
        });

        // add body loads contribution
        if( mBodyLoads != nullptr )
        {
            mBodyLoads->evaluate(mSpatialDomain,aWorkSets,-1.0);
        }
    }
};

class NewAbstractProblem
{
public:
    virtual ~NewAbstractProblem(){}

    /******************************************************************************//**
     * \brief Write results to output database.
     * \param [in] aFilename name of output database file
    **********************************************************************************/
    virtual void output(const std::string& aFilename) = 0;

    /******************************************************************************//**
     * \brief Solve numerical simulation
     * \param [in] aDomain independent variables
     * \return dependent variables
    **********************************************************************************/
    virtual Plato::Range<Plato::ScalarMultiVector>
    solution(const Plato::Domain& aDomain)=0;
};
// class Abstract Problem

/******************************************************************************//**
 * \brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename PhysicsType>
class Problem: public Plato::NewAbstractProblem
{
public:
    void output(const std::string& aFilename){}

    Plato::Range<Plato::ScalarMultiVector>
    solution(const Plato::Domain & aDomain)
    { return Range<Plato::ScalarMultiVector>(); }
};
// class Problem

}
// namespace immersus

namespace IfemTests
{

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_NewtonRaphsonStoppingCriterion)
{
    std::vector<double> tVector;
    immersus::Range<Plato::ScalarMultiVector> tRange();
}

}
// namespace IfemTests
