/*
 * AugLagStressCriterionTest.cpp
 *
 *  Created on: May 2, 2023
 */

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "Simp.hpp"
#include "BLAS1.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoMathTypes.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "AbstractLocalMeasure.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Apply augmented Lagragian method to model local constraints.
 * \tparam EvaluationType evaluation type for automatic differentiation tools; e.g., 
 *         type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class AugLagStressCriterion :
    public EvaluationType::ElementType,
    public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using StateT   = typename EvaluationType::StateScalarType;
    using ConfigT  = typename EvaluationType::ConfigScalarType;
    using ResultT  = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    using Residual = typename Plato::Elliptic::ResidualTypes<ElementType>;

    Plato::OrdinalType mNumLocalConstraints = 1;    /*!< number of local constraints */
 
    Plato::Scalar mMaterialPenalty = 3.0;           /*!< penalty for material penalty model */
    Plato::Scalar mMaxAugLagPenalty = 10000.0;      /*!< maximum penalty value allowed for AL formulation */
    Plato::Scalar mInitiaAugLagPenalty = 1.0;       /*!< initial Lagrange multipliers */
    Plato::Scalar mMinErsatzMaterialValue = 1e-9;   /*!<minimum ersatz material stiffness for material penalty model*/
    Plato::Scalar mAugLagPenaltyIncrement = 1.1;    /*!< increment multiplier for penalty values */
    Plato::Scalar mConservativeMultiplier = 0.25;   /*!< previous constraint multiplier used for penaly update */
    Plato::Scalar mInitialLagrangeMultiplier = 0.0; /*!< initial Lagrange multipliers */

    Plato::Array<1> mLocalMeasureLimits;            /*!< local measure limit */

    Plato::ScalarVector mLagrangeMultipliers;       /*!< Lagrange multipliers for augmented Lagragian formulation */
    Plato::ScalarVector mAugLagPenaltyValues;       /*!< penalty values for augmented Largangian formulation */
    Plato::ScalarVector mCurnConstraintValues;      /*!< current contraint values */
    Plato::ScalarVector mPrevConstraintValues;      /*!< previous contraint values */

    /*!< Local measure with FAD evaluation type */
    std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType>> mLocalMeasureEvaluationType;

    /*!< Local measure with POD type */
    std::shared_ptr<Plato::AbstractLocalMeasure<Residual>> mLocalMeasurePODType;

private:
    /******************************************************************************//**
     * \brief Allocate member data
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputParams)
    {
        this->parseScalars(aInputParams);  
        this->parseLimits(aInputParams); 
        this->allocateContainers();    

        Plato::blas1::fill(mInitiaAugLagPenalty, mAugLagPenaltyValues);
        Plato::blas1::fill(mInitialLagrangeMultiplier, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Read user inputs
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void parseScalars(Teuchos::ParameterList & aInputParams)
    {
        Teuchos::ParameterList & tParams = 
            aInputParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
        
        mNumLocalConstraints = tParams.get<Plato::Scalar>("Number of Local Constraints", 1);

        mMaterialPenalty = tParams.get<Plato::Scalar>("SIMP penalty", 3.0);
        mMaxAugLagPenalty = tParams.get<Plato::Scalar>("Penalty Upper Bound", 10000.0);
        mInitiaAugLagPenalty = tParams.get<Plato::Scalar>("Initial Penalty", 1.0);
        mMinErsatzMaterialValue = tParams.get<Plato::Scalar>("Min. Ersatz Material", 1e-9);
        mConservativeMultiplier = tParams.get<Plato::Scalar>(" Conservative Multiplier", 0.25);
        mAugLagPenaltyIncrement = tParams.get<Plato::Scalar>("Penalty Expansion Multiplier", 1.1);
        mInitialLagrangeMultiplier = tParams.get<Plato::Scalar>("Initial Lagrange Multiplier", 0.0);
    }

    void parseLimits(Teuchos::ParameterList & aInputParams)
    {
        bool tIsArray = aInputParams.isType<Teuchos::Array<Plato::Scalar>>("Limits");
        if (tIsArray)
        {
            Teuchos::Array<Plato::Scalar> tLimits = aInputParams.get<Teuchos::Array<Plato::Scalar>>("Limits");
            for(Plato::OrdinalType tIndex = 0; tIndex < tLimits.size(); tIndex++)
            {
                mLocalMeasureLimits(tIndex) = tLimits[tIndex];
            }
        }
        else
        {
            auto tMsg = std::string("Local constraints limits are not defined for stress constraint criterion. ") 
                + "Constraint limits are required to properly enforce the local constraints.";
            ANALYZE_THROWERR(tMsg)
        }
    }
    
    void allocateContainers()
    {
        auto tNumConstraints  = mNumLocalConstraints * mSpatialDomain.Mesh->NumElements();
        mCurnConstraintValues = Plato::ScalarVector("Current Constraints",tNumConstraints);
        mPrevConstraintValues = Plato::ScalarVector("Previous Constraints",tNumConstraints);
        mLagrangeMultipliers  = Plato::ScalarVector("Lagrange Multipliers",tNumConstraints);
        mAugLagPenaltyValues  = Plato::ScalarVector("Penalty Values",tNumConstraints);
    }

    void 
    evaluateCurrentConstraints(
        const Plato::ScalarMultiVector &aStateWS,
        const Plato::ScalarMultiVector &aControlWS,
        const Plato::ScalarArray3D     &aConfigWS
    )
    {
        // evaluate local measure
        const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();
        auto tNumConstraints = tNumCells * mNumLocalConstraints;
        Plato::ScalarVector tLocalMeasureValues("local measure values", tNumConstraints);
        (*mLocalMeasurePODType)(aStateWS, aControlWS, aConfigWS, tLocalMeasureValues);
        // copy global member data to local scope
        auto tLocalMeasureLimits  = mLocalMeasureLimits;
        auto tCurrentConstraints  = mCurnConstraintValues;
        auto tAugLagPenaltyValues = mAugLagPenaltyValues;
        auto tLagrangeMultipliers = mLagrangeMultipliers;
        // gather integration rule data
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();
        // evaluate local constraints
        Plato::MSIMP tSIMP(mMaterialPenalty,mMinErsatzMaterialValue);
        auto tNumLocalConstraints = mLocalMeasureLimits.size();
        Kokkos::parallel_for("local constraints",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCells,tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
        {
            // evaluate material penalty model
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            Plato::Scalar tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, aControlWS, tBasisValues);
            Plato::Scalar tStiffnessPenalty = tSIMP(tDensity);
            // evaluate local equality constraint
            for(Plato::OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumLocalConstraints; tConstraintIndex++)
            {
                // evaluate local inequality constraint 
                auto tLocalIndex = (iCellOrdinal * tNumLocalConstraints) + tConstraintIndex;
                const Plato::Scalar tLocalMeasureValueOverLimit = 
                    tLocalMeasureValues(tLocalIndex) / tLocalMeasureLimits(tLocalIndex);
                const Plato::Scalar tLocalMeasureValueOverLimitMinusOne = 
                    tLocalMeasureValueOverLimit - static_cast<Plato::Scalar>(1.0);
                const Plato::Scalar tConstraintValue = pow(tLocalMeasureValueOverLimitMinusOne, 2.0);
                // penalized local inequality constraint
                const Plato::Scalar tPenalizedConstraintValue = tStiffnessPenalty * tConstraintValue;
                // evaluate condition: \frac{-\lambda}{\theta}
                const Plato::Scalar tLagMultOverPenalty = ( static_cast<Plato::Scalar>(-1.0) * 
                    tLagrangeMultipliers(tLocalIndex) ) / tAugLagPenaltyValues(tLocalIndex);
                // evaluate equality constraint
                tCurrentConstraints(tLocalIndex) = tPenalizedConstraintValue > tLagMultOverPenalty ?
                    tPenalizedConstraintValue : tLagMultOverPenalty;
            }
        });
    }

    /******************************************************************************//**
     * \brief Update penalty values for AL formulation
    **********************************************************************************/
    void 
    updateAugLagPenaltyValues()
    {
        // copy global member data to local scope
        Plato::ScalarVector tAugLagPenaltyValues  = mAugLagPenaltyValues;
        Plato::ScalarVector tPrevConstraintValues = mPrevConstraintValues;
        Plato::ScalarVector tCurnConstraintValues = mCurnConstraintValues;

        Plato::Scalar tMaxAugLagPenalty       = mMaxAugLagPenalty; 
        Plato::Scalar tAugLagPenaltyIncrement = mAugLagPenaltyIncrement;
        Plato::Scalar tConservativeMultiplier = mConservativeMultiplier;

        // update penalty values
        const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();
        Plato::OrdinalType tNumLocalConstraints = mLocalMeasureLimits.size();
        Kokkos::parallel_for("update penalty values",Kokkos::RangePolicy<>(0,tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
        {
            for(Plato::OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumLocalConstraints; tConstraintIndex++)
            {
                Plato::OrdinalType tLocalIndex = (iCellOrdinal * tNumLocalConstraints) + tConstraintIndex;   
                // evaluate condition
                Plato::Scalar tCondition = tConservativeMultiplier * tPrevConstraintValues(tLocalIndex);
                Plato::Scalar tTrialPenalty = tCurnConstraintValues(tLocalIndex) > tCondition ? 
                    tAugLagPenaltyIncrement * tAugLagPenaltyValues(tLocalIndex) : tAugLagPenaltyValues(tLocalIndex); 
                tAugLagPenaltyValues(tLocalIndex) = 
                    tTrialPenalty < tMaxAugLagPenalty ? tTrialPenalty : tMaxAugLagPenalty;
            }
        });
    }

    void updateLagrangeMultipliers()
    {
        // copy global member data to local scope
        Plato::ScalarVector tAugLagPenaltyValues  = mAugLagPenaltyValues;
        Plato::ScalarVector tLagrangeMultipliers  = mLagrangeMultipliers;
        Plato::ScalarVector tCurnConstraintValues = mCurnConstraintValues;
        // update lagrange multipliers
        const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();
        Plato::OrdinalType tNumLocalConstraints = mLocalMeasureLimits.size();
        Kokkos::parallel_for("update lagrange multipliers",Kokkos::RangePolicy<>(0,tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
        {
            for(Plato::OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumLocalConstraints; tConstraintIndex++)
            {
                Plato::OrdinalType tLocalIndex = (iCellOrdinal * tNumLocalConstraints) + tConstraintIndex;
                tLagrangeMultipliers(tLocalIndex) = tLagrangeMultipliers(tLocalIndex) 
                    + ( tAugLagPenaltyValues(tLocalIndex) * tCurnConstraintValues(tLocalIndex) );
            }
        });
    }

public:
    AugLagStressCriterion(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aFuncName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aInputParams, aFuncName)
    {
        this->initialize(aInputParams);
    }
    ~AugLagStressCriterion(){}

    void
    updateProblem(
        const Plato::ScalarMultiVector & aStateWS,
        const Plato::ScalarMultiVector & aControlWS,
        const Plato::ScalarArray3D     & aConfigWS
    )
    {
        this->evaluateCurrentConstraints(aStateWS,aControlWS,aConfigWS);
        this->updateLagrangeMultipliers();
        this->updateAugLagPenaltyValues();
    }

    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
        const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
              Plato::ScalarVectorT      <ResultT>  & aResultWS,
              Plato::Scalar aTimeStep
    ) const
    {
    }
};

}
// namespace Plato

namespace AugLagStressCriterionTest
{
}
// namespace AugLagStressCriterionTest