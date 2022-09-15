#pragma once

#include "elliptic/VolumeIntegralCriterion_decl.hpp"

#include "Simp.hpp"
#include "PlatoMeshExpr.hpp"
#include "ApplyWeighting.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Elliptic
{

    /******************************************************************************//**
     * \brief Allocate member data
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    template<typename EvaluationType>
    void
    VolumeIntegralCriterion<EvaluationType>::
    initialize(Teuchos::ParameterList & aInputParams)
    {
        this->readInputs(aInputParams);
    }

    /******************************************************************************//**
     * \brief Read user inputs
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    template<typename EvaluationType>
    void
    VolumeIntegralCriterion<EvaluationType>::
    readInputs(Teuchos::ParameterList & aInputParams)
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

    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aPlatoDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aFuncName user defined function name
     **********************************************************************************/
    template<typename EvaluationType>
    VolumeIntegralCriterion<EvaluationType>::
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
    template<typename EvaluationType>
    VolumeIntegralCriterion<EvaluationType>::
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
    template<typename EvaluationType>
    VolumeIntegralCriterion<EvaluationType>::
    ~VolumeIntegralCriterion()
    {
    }

    /******************************************************************************//**
     * \brief Set volume integrated quanitity
     * \param [in] aInputEvaluationType evaluation type volume integrated quanitity
    **********************************************************************************/
    template<typename EvaluationType>
    void
    VolumeIntegralCriterion<EvaluationType>::
    setVolumeIntegratedQuantity(const std::shared_ptr<AbstractLocalMeasure<EvaluationType>> & aInput)
    {
        mLocalMeasure = aInput;
    }

    /******************************************************************************//**
     * \brief Set spatial weight function
     * \param [in] aInput math expression
    **********************************************************************************/
    template<typename EvaluationType>
    void
    VolumeIntegralCriterion<EvaluationType>::
    setSpatialWeightFunction(std::string aWeightFunctionString)
    {
        mSpatialWeightFunction = aWeightFunctionString;
    }


    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    template<typename EvaluationType>
    void
    VolumeIntegralCriterion<EvaluationType>::
    updateProblem(
        const Plato::ScalarMultiVector & aStateWS,
        const Plato::ScalarMultiVector & aControlWS,
        const Plato::ScalarArray3D     & aConfigWS
    )
    {
        // Perhaps update penalty exponent?
        WARNING("Penalty exponents not yet updated in VolumeIntegralCriterion.")
    }

    /******************************************************************************//**
     * \brief Evaluate volume average criterion
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    template<typename EvaluationType>
    void
    VolumeIntegralCriterion<EvaluationType>::
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
        const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
              Plato::ScalarVectorT      <ResultT>  & aResultWS,
              Plato::Scalar aTimeStep
    ) const
    {
        auto tSpatialWeights = Plato::computeSpatialWeights<ConfigT, ElementType>(mSpatialDomain, aConfigWS, mSpatialWeightFunction);

        auto tNumCells = mSpatialDomain.numCells();
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();

        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);
        Plato::ApplyWeighting<mNumNodesPerCell, /*num_terms=*/1, Plato::MSIMP> tApplyWeighting(tSIMP);

        // ****** COMPUTE VOLUME AVERAGED QUANTITIES AND STORE ON DEVICE ******
        Plato::ScalarVectorT<ResultT> tVolumeIntegratedQuantity("volume integrated quantity", tNumCells);
        (*mLocalMeasure)(aStateWS, aControlWS, aConfigWS, tVolumeIntegratedQuantity);
        
        Kokkos::parallel_for("compute volume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
}
//namespace Elliptic

}
//namespace Plato
