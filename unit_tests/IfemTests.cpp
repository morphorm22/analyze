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
#include "FadTypes.hpp"
#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "SpatialModel.hpp"
#include "SmallStrain.hpp"
#include "LinearStress.hpp"
#include "GradientMatrix.hpp"
#include "PlatoUtilities.hpp"
#include "PlatoStaticsTypes.hpp"
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

template<typename PhysicsT, typename EvaluationT>
class AbstractResidual
{
private:
    using ResultT = typename EvaluationT::ResultScalarType;

public:
    AbstractResidual(){}
    virtual ~AbstractResidual(){}

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate inner domain residual, exclude boundary terms.
     * \param [in] aWorkSets holds state and control worksets
     * \param [in/out] aResult   result workset
     ******************************************************************************/
    virtual void evaluate
    (const Plato::WorkSets &aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> &aResult) const = 0;

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate boundary forces, not related to any prescribed boundary force,
     *        resulting from applying integration by part to the residual equation.
     * \param [in]  aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in]  aWorkSets     holds input worksets (e.g. states, control, etc)
     * \param [out] aResultWS     result/output workset
     ******************************************************************************/
    virtual void evaluate_boundary
    (const Plato::SpatialModel &aSpatialModel,
     const Plato::WorkSets &aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> &aResult) const = 0;

    /***************************************************************************//**
     * \fn void evaluatePrescribed
     * \brief Evaluate vector function on prescribed boundaries.
     * \param [in]  aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in]  aWorkSets     holds input worksets (e.g. states, control, etc)
     * \param [out] aResultWS     result/output workset
     ******************************************************************************/
    virtual void evaluate_prescribed
    (const Plato::SpatialModel &aSpatialModel,
     const Plato::WorkSets &aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> &aResult) const = 0;
};
// class abstract residual


template<typename PhysicsT, typename EvaluationT>
class ElastostaticResidual : public Plato::AbstractResidual<PhysicsT, EvaluationT>
{
private:
    // set local element type
    using ElementType = typename EvaluationT::ElementType;

    // set local fad types
    using StateFadType  = typename EvaluationT::StateScalarType;
    using ResultFadType = typename EvaluationT::ResultScalarType;
    using ConfigFadType = typename EvaluationT::ConfigScalarType;

    const Plato::SpatialDomain& mSpatialDomain; /*!< spatial domain metadata */

public:
    ElastostaticResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aProbInputs) :
        mSpatialDomain(aDomain)
    {
    }

    void evaluate
    (const Plato::WorkSets &aDomain,
     Plato::ScalarMultiVectorT<ResultFadType> &aRange) const
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

        // get input worksets (i.e., domain for function evaluate)
        auto tStateWS  = Plato::metadata<Plato::ScalarMultiVectorT<StateFadType>>(aDomain.get("state"));
        auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ConfigFadType>>(aDomain.get("configuration"));

        // get element integration points and weights
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for("compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
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

            // apply divergence to stress
            tComputeStressDivergence(iCellOrdinal, aRange, tStress, tGradient, tVolume);

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
    }
};

class AbstractProb
{
public:
    virtual ~AbstractProb(){}

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
    solution(const Plato::Domain & aDomain)=0;
};
// class Abstract Problem

/******************************************************************************//**
 * \brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename PhysicsType>
class Problem: public Plato::AbstractProb
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
