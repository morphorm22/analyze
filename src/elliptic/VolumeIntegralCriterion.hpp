#pragma once

#include <algorithm>
#include <memory>

#include "Simp.hpp"
#include "PlatoStaticsTypes.hpp"
#include "WorksetBase.hpp"
#include "PlatoMathHelpers.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "ImplicitFunctors.hpp"
#include "AbstractLocalMeasure.hpp"
#include "BLAS1.hpp"


namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Volume integral criterion of field quantites (primarily for use with VolumeAverageCriterion)
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class VolumeIntegralCriterion :
    public EvaluationType::ElementType,
    public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain; /*!< mesh database */
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap; /*!< PLATO Engine output database */

    using StateT   = typename EvaluationType::StateScalarType;   /*!< state variables automatic differentiation type */
    using ConfigT  = typename EvaluationType::ConfigScalarType;  /*!< configuration variables automatic differentiation type */
    using ResultT  = typename EvaluationType::ResultScalarType;  /*!< result variables automatic differentiation type */
    using ControlT = typename EvaluationType::ControlScalarType; /*!< control variables automatic differentiation type */

    using FunctionBaseType = Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    std::string mSpatialWeightFunction;

    Plato::Scalar mPenalty;        /*!< penalty parameter in SIMP model */
    Plato::Scalar mMinErsatzValue; /*!< minimum ersatz material value in SIMP model */

    std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType>> mLocalMeasure; /*!< Volume averaged quantity with evaluation type */

private:
    /******************************************************************************//**
     * \brief Allocate member data
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputParams)
    {
        this->readInputs(aInputParams);
    }

    /******************************************************************************//**
     * \brief Read user inputs
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void readInputs(Teuchos::ParameterList & aInputParams)
    {
        Teuchos::ParameterList & tParams = aInputParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
        auto tPenaltyParams = tParams.sublist("Penalty Function");
        std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
        if (tPenaltyType != "SIMP")
        {
            ANALYZE_THROWERR("A penalty function type other than SIMP is not yet implemented for the VolumeIntegralCriterion.")
        }
        mPenalty        = tPenaltyParams.get<Plato::Scalar>("Exponent", 3.0);
        mMinErsatzValue = tPenaltyParams.get<Plato::Scalar>("Minimum Value", 1e-9);
    }

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aPlatoDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aFuncName user defined function name
     **********************************************************************************/
    VolumeIntegralCriterion(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aFuncName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aInputParams, aFuncName),
        mSpatialWeightFunction("1.0"),
        mPenalty(3),
        mMinErsatzValue(1.0e-9)
    {
        this->initialize(aInputParams);
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    VolumeIntegralCriterion(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, "Volume Integral Criterion"),
        mPenalty(3),
        mMinErsatzValue(0.0),
        mLocalMeasure(nullptr)
    {

    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~VolumeIntegralCriterion()
    {
    }

    /******************************************************************************//**
     * \brief Set volume integrated quanitity
     * \param [in] aInputEvaluationType evaluation type volume integrated quanitity
    **********************************************************************************/
    void setVolumeIntegratedQuantity(const std::shared_ptr<AbstractLocalMeasure<EvaluationType>> & aInput)
    {
        mLocalMeasure = aInput;
    }

    /******************************************************************************//**
     * \brief Set spatial weight function
     * \param [in] aInput math expression
    **********************************************************************************/
    void setSpatialWeightFunction(std::string aWeightFunctionString) override
    {
        mSpatialWeightFunction = aWeightFunctionString;
    }


    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void
    updateProblem(
        const Plato::ScalarMultiVector & aStateWS,
        const Plato::ScalarMultiVector & aControlWS,
        const Plato::ScalarArray3D     & aConfigWS
    ) override
    {
        // Perhaps update penalty exponent?
        WARNING("Penalty exponents not yet updated in VolumeIntegralCriterion.")
    }

    /******************************************************************************//**
     * \brief compute spatial weights
     * \param [in] aInput node location data
    **********************************************************************************/
    Plato::ScalarMultiVectorT<ConfigT>
    computeSpatialWeights(
        const Plato::ScalarArray3DT<ConfigT> & aConfig
    ) const
    {
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        auto tNumCells = mSpatialDomain.numCells();

        Plato::ScalarArray3DT<ConfigT> tPhysicalPoints("physical points", tNumCells, tNumPoints, mNumSpatialDims);
        Plato::mapPoints<ElementType>(aConfig, tPhysicalPoints);

        Plato::ScalarMultiVectorT<ConfigT> tFxnValues("function values", tNumCells*tNumPoints, 1);
        Plato::getFunctionValues<mNumSpatialDims>(tPhysicalPoints, mSpatialWeightFunction, tFxnValues);

        return tFxnValues;
    }

    /******************************************************************************//**
     * \brief Evaluate volume average criterion
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
        const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
              Plato::ScalarVectorT      <ResultT>  & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
        auto tSpatialWeights  = computeSpatialWeights(aConfigWS);

        auto tNumCells = mSpatialDomain.numCells();
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);
        Plato::ApplyWeighting<mNumNodesPerCell, /*num_terms=*/1, Plato::MSIMP> tApplyWeighting(tSIMP);

        // ****** COMPUTE VOLUME AVERAGED QUANTITIES AND STORE ON DEVICE ******
        Plato::ScalarVectorT<ResultT> tVolumeIntegratedQuantity("volume integrated quantity", tNumCells);
        (*mLocalMeasure)(aStateWS, aConfigWS, tVolumeIntegratedQuantity);
        
        Kokkos::parallel_for("compute volume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint  = tCubPoints(iGpOrdinal);
            auto tCubWeight = tCubWeights(iGpOrdinal);

            auto tJacobian = ElementType::jacobian(tCubPoint, aConfigWS, iCellOrdinal);

            ResultT tCellVolume = Plato::determinant(tJacobian);

            tCellVolume *= tCubWeight;

            ResultT tValue = tVolumeIntegratedQuantity(iCellOrdinal) * tCellVolume * tSpatialWeights(iCellOrdinal*tNumPoints + iGpOrdinal, 0);

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            tApplyWeighting(iCellOrdinal, aControlWS, tBasisValues, tValue);

            Kokkos::atomic_add(&aResultWS(iCellOrdinal), tValue);
        });
    }
};
// class VolumeIntegralCriterion

}
//namespace Elliptic

}
//namespace Plato
