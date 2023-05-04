/*
 * AugLagStressCriterionTest.cpp
 *
 *  Created on: May 2, 2023
 */

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "Analyze_Diagnostics.hpp"
#include "util/PlatoTestHelpers.hpp"

#include "Simp.hpp"
#include "Tri3.hpp"
#include "Tet4.hpp"
#include "ToMap.hpp"
#include "BLAS1.hpp"
#include "Mechanics.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoMathTypes.hpp"
#include "MechanicsElement.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "AbstractLocalMeasure.hpp"

#include "elliptic/MassMoment.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/WeightedSumFunction.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"

namespace Plato
{

class AugLagDataMng
{
public:
    Plato::OrdinalType mNumCells = 0;               /*!< number of cells */
    Plato::OrdinalType mNumLocalConstraints = 0;    /*!< number of local constraints */
 
    Plato::Scalar mMaxPenalty = 10000.0;            /*!< maximum penalty value allowed for AL formulation */
    Plato::Scalar mInitiaPenalty = 1.0;             /*!< initial Lagrange multipliers */
    Plato::Scalar mPenaltyIncrement = 1.1;          /*!< increment multiplier for penalty values */
    Plato::Scalar mPenaltyUpdateParameter = 0.25;   /*!< previous constraint multiplier used for penaly update */
    Plato::Scalar mInitialLagrangeMultiplier = 0.0; /*!< initial Lagrange multipliers */

    Plato::ScalarVector mPenaltyValues;             /*!< penalty values for augmented Largangian formulation */
    Plato::ScalarVector mLocalMeasureLimits;        /*!< local constraint limit */
    Plato::ScalarVector mLagrangeMultipliers;       /*!< Lagrange multipliers for augmented Lagragian formulation */
    Plato::ScalarVector mCurrentConstraintValues;   /*!< current contraint values */
    Plato::ScalarVector mPreviousConstraintValues;  /*!< previous contraint values */

public:
    AugLagDataMng(){}
    ~AugLagDataMng(){}
    
    void initialize()
    {
        Plato::blas1::fill(mInitiaPenalty, mPenaltyValues);
        Plato::blas1::fill(mInitialLagrangeMultiplier, mLagrangeMultipliers);
    }

    void updateLagrangeMultipliers()
    {
        // copy global member data to local scope
        Plato::ScalarVector tPenaltyValues = mPenaltyValues;
        Plato::ScalarVector tLagrangeMultipliers  = mLagrangeMultipliers;
        Plato::ScalarVector tCurrentConstraintValues = mCurrentConstraintValues;
        // update lagrange multipliers
        const Plato::OrdinalType tNumCells = mNumCells;
        Plato::OrdinalType tNumLocalConstraints = mNumLocalConstraints;
        Kokkos::parallel_for("update lagrange multipliers",Kokkos::RangePolicy<>(0,tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
        {
            for(Plato::OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumLocalConstraints; tConstraintIndex++)
            {
                Plato::OrdinalType tLocalIndex = (iCellOrdinal * tNumLocalConstraints) + tConstraintIndex;
                tLagrangeMultipliers(tLocalIndex) = tLagrangeMultipliers(tLocalIndex) 
                    + ( tPenaltyValues(tLocalIndex) * tCurrentConstraintValues(tLocalIndex) );
            }
        });
    }

    void updatePenaltyValues()
    {
        // copy member containers to local scope
        Plato::ScalarVector tPenaltyValues  = mPenaltyValues;
        Plato::ScalarVector tCurrentConstraintValues = mCurrentConstraintValues;
        Plato::ScalarVector tPreviousConstraintValues = mPreviousConstraintValues;
        // update penalty values
        const Plato::OrdinalType tNumCells = mNumCells;
        Plato::OrdinalType tNumLocalConstraints = mNumLocalConstraints;
        Kokkos::parallel_for("update penalty values",Kokkos::RangePolicy<>(0,tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
        {
            for(Plato::OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumLocalConstraints; tConstraintIndex++)
            {
                Plato::OrdinalType tLocalIndex = (iCellOrdinal * tNumLocalConstraints) + tConstraintIndex;   
                // evaluate condition
                Plato::Scalar tCondition = mPenaltyUpdateParameter * tPreviousConstraintValues(tLocalIndex);
                Plato::Scalar tTrialPenalty = tCurrentConstraintValues(tLocalIndex) > tCondition ? 
                    mPenaltyIncrement * tPenaltyValues(tLocalIndex) : tPenaltyValues(tLocalIndex); 
                tPenaltyValues(tLocalIndex) = tTrialPenalty < mMaxPenalty ? tTrialPenalty : mMaxPenalty;
            }
        });
    }

    void parseNumerics(Teuchos::ParameterList &aParams)
    {
        mMaxPenalty = aParams.get<Plato::Scalar>("Maximum Penalty", 10000.0);
        mInitiaPenalty = aParams.get<Plato::Scalar>("Initial Penalty", 1.0);
        mPenaltyIncrement = aParams.get<Plato::Scalar>("Penalty Increment", 1.1);
        mPenaltyUpdateParameter = aParams.get<Plato::Scalar>("Penalty Update Parameter", 0.25);
        mInitialLagrangeMultiplier = aParams.get<Plato::Scalar>("Initial Lagrange Multiplier", 0.0);
    }
    
    void parseLimits(Teuchos::ParameterList &aParams)
    {
        bool tIsArray = aParams.isType<Teuchos::Array<Plato::Scalar>>("Limits");
        if (tIsArray)
        {
            // parse inputs
            Teuchos::Array<Plato::Scalar> tLimits = aParams.get<Teuchos::Array<Plato::Scalar>>("Limits");
            // create mirror 
            mNumLocalConstraints = tLimits.size();
            mLocalMeasureLimits  = Plato::ScalarVector("Limits",mNumLocalConstraints);
            auto tHostArray = Kokkos::create_mirror_view(mLocalMeasureLimits);
            for( Plato::OrdinalType tIndex = 0; tIndex < mNumLocalConstraints; tIndex++)
            {
                tHostArray(tIndex) = tLimits[tIndex];
            }
            Kokkos::deep_copy(mLocalMeasureLimits,tHostArray);
        }
        else
        {
            auto tMsg = std::string("Local constraints limits are not defined in criterion block. ") 
                + "Constraint limits are required to properly enforce the local constraints.";
            ANALYZE_THROWERR(tMsg)
        }
    }

    void allocateContainers(const Plato::OrdinalType &aNumCells)
    {
        if(mNumLocalConstraints <= 0)
        {
            auto tMsg = std::string("Number of local constraints is not defined; i.e., ") 
                + "number of local constraints is set to " + std::to_string(mNumLocalConstraints) + ".";
            ANALYZE_THROWERR(tMsg)
        }
        mNumCells = aNumCells;
        if(mNumCells <= 0)
        {
            auto tMsg = std::string("Number of cells is not defined; i.e., ") 
                + "number of cells is set to " + std::to_string(mNumCells) + ".";
            ANALYZE_THROWERR(tMsg)
        }

        auto tNumConstraints = mNumLocalConstraints * mNumCells;
        mPenaltyValues = Plato::ScalarVector("Penalty Values",tNumConstraints);
        mLagrangeMultipliers = Plato::ScalarVector("Lagrange Multipliers",tNumConstraints);
        mCurrentConstraintValues = Plato::ScalarVector("Current Constraints",tNumConstraints);
        mPreviousConstraintValues = Plato::ScalarVector("Previous Constraints",tNumConstraints);
    }
};

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

    Plato::Scalar mMaterialPenalty = 3.0;         /*!< penalty for material penalty model */
    Plato::Scalar mMinErsatzMaterialValue = 1e-9; /*!< minimum ersatz material stiffness for material penalty model*/

    Plato::AugLagDataMng mAugLagDataMng;    /*!< contains all relevant data associated with the AL method*/

    /*!< Local measure with FAD evaluation type */
    std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType>> mLocalMeasureEvaluationType;

    /*!< Local measure with POD type */
    std::shared_ptr<Plato::AbstractLocalMeasure<Residual>> mLocalMeasurePODType;

private:
    /******************************************************************************//**
     * \brief Allocate member data
     * \param [in] aParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aParams)
    {
        this->parseNumerics(aParams);
        this->parseLimits(aParams); 
        auto tNumCells = mSpatialDomain.Mesh->NumElements();
        mAugLagDataMng.allocateContainers(tNumCells);
        mAugLagDataMng.initialize();
    }

    /******************************************************************************//**
     * \brief Read user inputs
     * \param [in] aParams input parameters database
    **********************************************************************************/
    void parseNumerics(Teuchos::ParameterList & aParams)
    {
        Teuchos::ParameterList & tParams = 
            aParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
        mMaterialPenalty = tParams.get<Plato::Scalar>("Exponent", 3.0);
        mMinErsatzMaterialValue = tParams.get<Plato::Scalar>("Minimum Value", 1e-9);
        mAugLagDataMng.parseNumerics(tParams);
    }

    void parseLimits(Teuchos::ParameterList & aParams)
    {
        Teuchos::ParameterList &tParams = 
            aParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
        mAugLagDataMng.parseLimits(tParams);
    }

    void 
    evaluateCurrentConstraints(
        const Plato::ScalarMultiVector &aStateWS,
        const Plato::ScalarMultiVector &aControlWS,
        const Plato::ScalarArray3D     &aConfigWS
    )
    {
        // update previous constraint container
        Plato::blas1::copy(mAugLagDataMng.mCurrentConstraintValues,mAugLagDataMng.mPreviousConstraintValues);
        // evaluate local measure
        auto tNumCells = mAugLagDataMng.mNumCells;
        auto tNumConstraints = tNumCells * mAugLagDataMng.mNumLocalConstraints;
        Plato::ScalarVector tLocalMeasureValues("local measure values", tNumConstraints);
        (*mLocalMeasurePODType)(aStateWS, aControlWS, aConfigWS, tLocalMeasureValues);
        // copy global member data to local scope
        auto tLocalMeasureLimits  = mAugLagDataMng.mLocalMeasureLimits;
        auto tCurrentConstraints  = mAugLagDataMng.mCurrentConstraintValues;
        auto tPenaltyValues = mAugLagDataMng.mPenaltyValues;
        auto tLagrangeMultipliers = mAugLagDataMng.mLagrangeMultipliers;
        // gather integration rule data
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();
        // evaluate local constraints
        Plato::MSIMP tSIMP(mMaterialPenalty,mMinErsatzMaterialValue);
        auto tNumLocalConstraints = mAugLagDataMng.mNumLocalConstraints;
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
                    tLocalMeasureValues(tLocalIndex) / tLocalMeasureLimits(tConstraintIndex);
                const Plato::Scalar tLocalMeasureValueOverLimitMinusOne = 
                    tLocalMeasureValueOverLimit - static_cast<Plato::Scalar>(1.0);
                const Plato::Scalar tConstraintValue = pow(tLocalMeasureValueOverLimitMinusOne, 2.0);
                // penalized local inequality constraint
                const Plato::Scalar tPenalizedConstraintValue = tStiffnessPenalty * tConstraintValue;
                // evaluate condition: \frac{-\lambda}{\theta}
                const Plato::Scalar tLagMultOverPenalty = ( static_cast<Plato::Scalar>(-1.0) * 
                    tLagrangeMultipliers(tLocalIndex) ) / tPenaltyValues(tLocalIndex);
                // evaluate equality constraint
                tCurrentConstraints(tLocalIndex) = tPenalizedConstraintValue > tLagMultOverPenalty ?
                    tPenalizedConstraintValue : tLagMultOverPenalty;
            }
        });
    }

public:
    AugLagStressCriterion(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParams,
        const std::string            & aFuncName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aParams, aFuncName)
    {
        this->initialize(aParams);
    }
    ~AugLagStressCriterion(){}

    void
    setLocalMeasure(
        const std::shared_ptr<AbstractLocalMeasure<EvaluationType>> & aInputEvaluationType,
        const std::shared_ptr<AbstractLocalMeasure<Residual>>       & aInputPODType
    )
    {
        mLocalMeasurePODType        = aInputPODType;
        mLocalMeasureEvaluationType = aInputEvaluationType;
    }

    void
    updateProblem(
        const Plato::ScalarMultiVector & aStateWS,
        const Plato::ScalarMultiVector & aControlWS,
        const Plato::ScalarArray3D     & aConfigWS
    )
    {
        this->evaluateCurrentConstraints(aStateWS,aControlWS,aConfigWS);
        mAugLagDataMng.updateLagrangeMultipliers();
        mAugLagDataMng.updatePenaltyValues();
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
        using StrainT = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;
        // compute local measure and store on device
        const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();
        const Plato::OrdinalType tNumConstraints = tNumCells * mAugLagDataMng.mNumLocalConstraints;
        Plato::ScalarVectorT<ResultT> tLocalMeasureValues("local measure values", tNumConstraints);
        (*mLocalMeasureEvaluationType)(aStateWS, aControlWS, aConfigWS, tLocalMeasureValues);
        Plato::ScalarVectorT<ResultT> 
            tOutputPenalizedLocalMeasure("output penalized local measure values",tNumConstraints);
        // transfer member data to local scope
        auto tPenaltyMultipliers = mAugLagDataMng.mPenaltyValues;
        auto tLocalMeasureLimits = mAugLagDataMng.mLocalMeasureLimits;
        auto tLagrangeMultipliers = mAugLagDataMng.mLagrangeMultipliers;
        // access cubature/integration rule data
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();
        // evaluate augmented lagrangian penalty term 
        Plato::MSIMP tSIMP(mMaterialPenalty,mMinErsatzMaterialValue);
        auto tNumLocalConstraints = mAugLagDataMng.mNumLocalConstraints;
        Plato::Scalar tNormalization = static_cast<Plato::Scalar>(1.0/tNumCells);
        Kokkos::parallel_for("augmented lagrangian penalty", 
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
            KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            // evaluate material penalty model
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, aControlWS, tBasisValues);
            ControlT tStiffnessPenalty = tSIMP(tDensity);
            // evaluate local equality constraint
            for(Plato::OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumLocalConstraints; tConstraintIndex++)
            {
                // evaluate local inequality constraint 
                const Plato::OrdinalType tLocalIndex = 
                    (iCellOrdinal * tNumLocalConstraints) + tConstraintIndex;
                const ResultT tLocalMeasureValueOverLimit = 
                    tLocalMeasureValues(tLocalIndex) / tLocalMeasureLimits(tConstraintIndex);
                const ResultT tLocalMeasureValueOverLimitMinusOne = 
                    tLocalMeasureValueOverLimit - static_cast<Plato::Scalar>(1.0);
                const ResultT tConstraintValue = pow(tLocalMeasureValueOverLimitMinusOne, 2.0);
                // penalized local inequality constraint
                const ResultT tPenalizedConstraintValue = tStiffnessPenalty * tConstraintValue;
                // evaluate condition: \frac{-\lambda}{\theta}
                const Plato::Scalar tLagMultOverPenalty = ( static_cast<Plato::Scalar>(-1.0) * 
                    tLagrangeMultipliers(tLocalIndex) ) / tPenaltyMultipliers(tLocalIndex);
                // evaluate augmented Lagrangian equality constraint
                const ResultT tConstraint = tPenalizedConstraintValue > tLagMultOverPenalty ?
                    tPenalizedConstraintValue : static_cast<ResultT>(tLagMultOverPenalty);
                // evaluate augmented Lagrangian penalty term
                const ResultT tResult = tNormalization * ( ( tLagrangeMultipliers(tLocalIndex) *
                    tConstraint ) + ( static_cast<Plato::Scalar>(0.5) * tPenaltyMultipliers(tLocalIndex) *
                    tConstraint * tConstraint ) );
                Kokkos::atomic_add(&aResultWS(iCellOrdinal), tResult);
                // evaluate outputs
                const ResultT tOutput = mMaterialPenalty * tLocalMeasureValues(iCellOrdinal);
                Kokkos::atomic_add(&tOutputPenalizedLocalMeasure(iCellOrdinal),tOutput);
            }
        });
        // save penalized output local measure values to output data map 
        Plato::toMap(mDataMap,tOutputPenalizedLocalMeasure,mLocalMeasureEvaluationType->getName(),mSpatialDomain);
    }
};

}
// namespace Plato

namespace AugLagStressCriterionTest
{

   Teuchos::RCP<Teuchos::ParameterList> tGenericParamListOne = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                                                  \n"
    "<ParameterList name='Spatial Model'>                                                                \n"
      "<ParameterList name='Domains'>                                                                    \n"
        "<ParameterList name='Design Volume'>                                                            \n"
          "<Parameter name='Element Block' type='string' value='body'/>                                  \n"
          "<Parameter name='Material Model' type='string' value='Mystic'/>                               \n"
        "</ParameterList>                                                                                \n"
      "</ParameterList>                                                                                  \n"
    "</ParameterList>                                                                                    \n"
    "<ParameterList name='Material Models'>                                                              \n"
      "<ParameterList name='Mystic'>                                                                     \n"
        "<ParameterList name='Isotropic Linear Elastic'>                                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>                                \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>                              \n"
        "</ParameterList>                                                                                \n"
      "</ParameterList>                                                                                  \n"
    "</ParameterList>                                                                                    \n"
    "<ParameterList name='Criteria'>                                                                     \n"
    "  <ParameterList name='Objective'>                                                                  \n"
    "    <Parameter name='Type' type='string' value='Weighted Sum'/>                                     \n"
    "    <Parameter name='Functions' type='Array(string)' value='{My Mass,My Stress}'/>                  \n"
    "    <Parameter name='Weights' type='Array(double)' value='{1.0,1.0}'/>                              \n"
    "  </ParameterList>                                                                                  \n"
    "  <ParameterList name='My Mass'>                                                                    \n"
    "    <Parameter name='Type'                 type='string' value='Scalar Function'/>                  \n"
    "    <Parameter name='Scalar Function Type' type='string' value='Mass'/>                             \n"
    "  </ParameterList>                                                                                  \n"
    "  <ParameterList name='My Stress'>                                                                  \n"
    "    <Parameter name='Type'                        type='string'        value='Scalar Function'/>    \n"
    "    <Parameter name='Scalar Function Type'        type='string'        value='Strength Constraint'/>\n"
    "    <Parameter name='Exponent'                    type='double'        value='2.0'/>                \n"
    "    <Parameter name='Minimum Value'               type='double'        value='1.0e-6'/>             \n"
    "    <Parameter name='Maximum Penalty'             type='double'        value='100'/>                \n"
    "    <Parameter name='Initial Penalty'             type='double'        value='2.0'/>                \n"
    "    <Parameter name='Penalty Increment'           type='double'        value='1.5'/>                \n"
    "    <Parameter name='Penalty Update Parameter'    type='double'        value='0.15'/>               \n"
    "    <Parameter name='Initial Lagrange Multiplier' type='double'        value='0.1'/>                \n"
    "    <Parameter name='Limits'                      type='Array(double)' value='{4.5,5.0}'/>          \n"
    "  </ParameterList>                                                                                  \n"
    "</ParameterList>                                                                                    \n"
  "</ParameterList>                                                                                      \n"
  );

   Teuchos::RCP<Teuchos::ParameterList> tGenericParamListTwo = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                                                  \n"
    "<ParameterList name='Spatial Model'>                                                                \n"
      "<ParameterList name='Domains'>                                                                    \n"
        "<ParameterList name='Design Volume'>                                                            \n"
          "<Parameter name='Element Block' type='string' value='body'/>                                  \n"
          "<Parameter name='Material Model' type='string' value='Mystic'/>                               \n"
        "</ParameterList>                                                                                \n"
      "</ParameterList>                                                                                  \n"
    "</ParameterList>                                                                                    \n"
    "<ParameterList name='Material Models'>                                                              \n"
      "<ParameterList name='Mystic'>                                                                     \n"
        "<ParameterList name='Isotropic Linear Elastic'>                                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>                                \n"
          "<Parameter  name='Youngs Modulus' type='double' value='4.0'/>                                 \n"
          "<Parameter  name='Density'        type='double' value='0.5'/>                                 \n"
        "</ParameterList>                                                                                \n"
      "</ParameterList>                                                                                  \n"
    "</ParameterList>                                                                                    \n"
    "<ParameterList name='Criteria'>                                                                     \n"
    "  <ParameterList name='Objective'>                                                                  \n"
    "    <Parameter name='Type' type='string' value='Weighted Sum'/>                                     \n"
    "    <Parameter name='Functions' type='Array(string)' value='{My Mass,My Stress}'/>                  \n"
    "    <Parameter name='Weights' type='Array(double)' value='{1.0,1.0}'/>                              \n"
    "  </ParameterList>                                                                                  \n"
    "  <ParameterList name='My Mass'>                                                                    \n"
    "    <Parameter name='Type'                 type='string' value='Scalar Function'/>                  \n"
    "    <Parameter name='Scalar Function Type' type='string' value='Mass'/>                             \n"
    "  </ParameterList>                                                                                  \n"
    "  <ParameterList name='My Stress'>                                                                  \n"
    "    <Parameter name='Type'                        type='string'        value='Scalar Function'/>    \n"
    "    <Parameter name='Scalar Function Type'        type='string'        value='Strength Constraint'/>\n"
    "    <Parameter name='Local Measure'               type='string'        value='VonMises'/>           \n"
    "    <Parameter name='Exponent'                    type='double'        value='2.0'/>                \n"
    "    <Parameter name='Minimum Value'               type='double'        value='1.0e-6'/>             \n"
    "    <Parameter name='Maximum Penalty'             type='double'        value='100'/>                \n"
    "    <Parameter name='Initial Penalty'             type='double'        value='2.0'/>                \n"
    "    <Parameter name='Penalty Increment'           type='double'        value='1.5'/>                \n"
    "    <Parameter name='Penalty Update Parameter'    type='double'        value='0.15'/>               \n"
    "    <Parameter name='Initial Lagrange Multiplier' type='double'        value='0.1'/>                \n"
    "    <Parameter name='Limits'                      type='Array(double)' value='{1.0}'/>              \n"
    "  </ParameterList>                                                                                  \n"
    "</ParameterList>                                                                                    \n"
  "</ParameterList>                                                                                      \n"
  );

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagDataMng_ParseDefaults)
{
    Teuchos::RCP<Teuchos::ParameterList> tGenericParamList = 
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>                                                            \n"
              "<ParameterList name='Spatial Model'>                                                          \n"
                "<ParameterList name='Domains'>                                                              \n"
                  "<ParameterList name='Design Volume'>                                                      \n"
                    "<Parameter name='Element Block' type='string' value='body'/>                            \n"
                    "<Parameter name='Material Model' type='string' value='Mystic'/>                         \n"
                  "</ParameterList>                                                                          \n"
                "</ParameterList>                                                                            \n"
              "</ParameterList>                                                                              \n"
              "<ParameterList name='Criteria'>                                                               \n"
              "  <ParameterList name='Objective'>                                                            \n"
              "    <Parameter name='Type' type='string' value='Weighted Sum'/>                               \n"
              "    <Parameter name='Functions' type='Array(string)' value='{My Stress}'/>                    \n"
              "    <Parameter name='Weights' type='Array(double)' value='{1.0}'/>                            \n"
              "  </ParameterList>                                                                            \n"
              "  <ParameterList name='My Stress'>                                                            \n"
              "    <Parameter name='Type'                 type='string'        value='Scalar Function'/>     \n"
              "    <Parameter name='Scalar Function Type' type='string'        value='Strength Constraint'/> \n"
              "    <Parameter name='Local Measure'        type='string'        value='VonMises'/>            \n"
              "    <Parameter name='Limits'               type='Array(double)' value='{1.0}'/>               \n"
              "  </ParameterList>                                                                            \n"
              "</ParameterList>                                                                              \n"
            "</ParameterList>                                                                                \n"
        );
    Plato::AugLagDataMng tDataMng;
    Teuchos::ParameterList & tParams = 
        tGenericParamList->sublist("Criteria").get<Teuchos::ParameterList>("My Stress");
    // parse numerics
    tDataMng.parseNumerics(tParams);
    // test floating data
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(10000., tDataMng.mMaxPenalty               , tTolerance);
    TEST_FLOATING_EQUALITY(1.0   , tDataMng.mInitiaPenalty            , tTolerance);
    TEST_FLOATING_EQUALITY(1.1   , tDataMng.mPenaltyIncrement         , tTolerance);
    TEST_FLOATING_EQUALITY(0.25  , tDataMng.mPenaltyUpdateParameter   , tTolerance);
    TEST_FLOATING_EQUALITY(0.0   , tDataMng.mInitialLagrangeMultiplier, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagDataMng_ParseInputs)
{
    Plato::AugLagDataMng tDataMng;
    Teuchos::ParameterList & tParams = 
        tGenericParamListOne->sublist("Criteria").get<Teuchos::ParameterList>("My Stress");
    // parse numerics
    tDataMng.parseNumerics(tParams);
    // test floating data
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(100.0, tDataMng.mMaxPenalty               , tTolerance);
    TEST_FLOATING_EQUALITY(2.0  , tDataMng.mInitiaPenalty            , tTolerance);
    TEST_FLOATING_EQUALITY(1.5  , tDataMng.mPenaltyIncrement         , tTolerance);
    TEST_FLOATING_EQUALITY(0.15 , tDataMng.mPenaltyUpdateParameter   , tTolerance);
    TEST_FLOATING_EQUALITY(0.1  , tDataMng.mInitialLagrangeMultiplier, tTolerance);
    // parse limits
    tDataMng.parseLimits(tParams);
    TEST_EQUALITY(2, tDataMng.mLocalMeasureLimits.size());
    TEST_EQUALITY(2, tDataMng.mNumLocalConstraints);
    auto tHostArray = Kokkos::create_mirror_view(tDataMng.mLocalMeasureLimits);
    std::vector<double> tGoldLimits = {4.5,5.0};
    for(Plato::OrdinalType tIndex = 0; tIndex < tDataMng.mLocalMeasureLimits.size(); tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldLimits[tIndex], tHostArray(tIndex), tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagDataMng_AllocateContainers)
{
    Plato::AugLagDataMng tDataMng;
    tDataMng.mNumLocalConstraints = 2;
    tDataMng.allocateContainers(5/*num cells*/);
    // test
    TEST_EQUALITY(10, tDataMng.mPenaltyValues.size());
    TEST_EQUALITY(10, tDataMng.mLagrangeMultipliers.size());
    TEST_EQUALITY(10, tDataMng.mCurrentConstraintValues.size());
    TEST_EQUALITY(10, tDataMng.mPreviousConstraintValues.size());
    // test throw: zero cells
    TEST_THROW(tDataMng.allocateContainers(0),std::runtime_error);
    // test throw: zero local constraints
    tDataMng.mNumLocalConstraints = 0;
    TEST_THROW(tDataMng.allocateContainers(1),std::runtime_error);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagDataMng_Initialize)
{
    Plato::AugLagDataMng tDataMng;
    Teuchos::ParameterList & tParams = 
        tGenericParamListOne->sublist("Criteria").get<Teuchos::ParameterList>("My Stress");
    // parse numerics
    tDataMng.parseNumerics(tParams);
    tDataMng.parseLimits(tParams);
    tDataMng.allocateContainers(3/*num cells*/);
    tDataMng.initialize();
    // test
    constexpr double tTolerance = 1e-4;
    TEST_EQUALITY(6,tDataMng.mPenaltyValues.size());
    auto tHostPM = Kokkos::create_mirror_view(tDataMng.mPenaltyValues);
    TEST_EQUALITY(6,tDataMng.mLagrangeMultipliers.size());
    auto tHostLM = Kokkos::create_mirror_view(tDataMng.mLagrangeMultipliers);
    for(Plato::OrdinalType tIndex = 0; tIndex < tDataMng.mLagrangeMultipliers.size(); tIndex++)
    {
        TEST_FLOATING_EQUALITY(2.0,tHostPM(tIndex),tTolerance);
        TEST_FLOATING_EQUALITY(0.1,tHostLM(tIndex),tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagDataMng_UpdateLagrangeMultipliers)
{
    Plato::AugLagDataMng tDataMng;
    Teuchos::ParameterList & tParams = 
        tGenericParamListOne->sublist("Criteria").get<Teuchos::ParameterList>("My Stress");
    // parse numeric inputs
    tDataMng.parseNumerics(tParams);
    tDataMng.parseLimits(tParams);
    tDataMng.allocateContainers(3/*num cells*/);
    tDataMng.initialize();
    // update lagrange multipliers
    Plato::blas1::fill(1.0,tDataMng.mCurrentConstraintValues);
    tDataMng.updateLagrangeMultipliers();
    // test
    constexpr double tTolerance = 1e-4;
    TEST_EQUALITY(6,tDataMng.mLagrangeMultipliers.size());
    auto tHostLM = Kokkos::create_mirror_view(tDataMng.mLagrangeMultipliers);
    for(Plato::OrdinalType tIndex = 0; tIndex < tDataMng.mLagrangeMultipliers.size(); tIndex++)
    {
        TEST_FLOATING_EQUALITY(2.1,tHostLM(tIndex),tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagDataMng_UpdatePenaltyValues_1)
{
    Plato::AugLagDataMng tDataMng;
    Teuchos::ParameterList & tParams = 
        tGenericParamListOne->sublist("Criteria").get<Teuchos::ParameterList>("My Stress");
    // parse numeric inputs
    tDataMng.parseNumerics(tParams);
    tDataMng.parseLimits(tParams);
    tDataMng.allocateContainers(3/*num cells*/);
    tDataMng.initialize();
    // update penalty values
    Plato::blas1::fill(1.0,tDataMng.mCurrentConstraintValues);
    Plato::blas1::fill(2.0,tDataMng.mPreviousConstraintValues);
    tDataMng.updatePenaltyValues();
    // test
    constexpr double tTolerance = 1e-4;
    TEST_EQUALITY(6,tDataMng.mPenaltyValues.size());
    auto tHostPV = Kokkos::create_mirror_view(tDataMng.mPenaltyValues);
    for(Plato::OrdinalType tIndex = 0; tIndex < tDataMng.mPenaltyValues.size(); tIndex++)
    {
        TEST_FLOATING_EQUALITY(3.0,tHostPV(tIndex),tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagDataMng_UpdatePenaltyValues_2)
{
    Plato::AugLagDataMng tDataMng;
    Teuchos::ParameterList & tParams = 
        tGenericParamListOne->sublist("Criteria").get<Teuchos::ParameterList>("My Stress");
    // parse numeric inputs
    tDataMng.parseNumerics(tParams);
    tDataMng.parseLimits(tParams);
    tDataMng.allocateContainers(3/*num cells*/);
    tDataMng.initialize();
    // update penalty values
    Plato::blas1::fill(0.25,tDataMng.mCurrentConstraintValues);
    Plato::blas1::fill(2.0,tDataMng.mPreviousConstraintValues);
    tDataMng.updatePenaltyValues();
    // test
    constexpr double tTolerance = 1e-4;
    TEST_EQUALITY(6,tDataMng.mPenaltyValues.size());
    auto tHostPV = Kokkos::create_mirror_view(tDataMng.mPenaltyValues);
    for(Plato::OrdinalType tIndex = 0; tIndex < tDataMng.mPenaltyValues.size(); tIndex++)
    {
        TEST_FLOATING_EQUALITY(2.0,tHostPV(tIndex),tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StrengthConstraintCriterion_Evaluate_VonMises2D)
{
    // create mesh
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

    //set ad-types
    using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    // create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);
    // create control workset
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);
    // create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::blas1::fill(0.1, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
            {   tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal);}, "fill state");
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);
    // create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);
    
    // create spatial model
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamListTwo, tDataMap);

    // create criterion
    auto tOnlyDomainDefined = tSpatialModel.Domains.front();
    Plato::AugLagStressCriterion<Residual> tCriterion(tOnlyDomainDefined,tDataMap,*tGenericParamListTwo,"My Stress");
    
    // create material model
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    auto tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    
    // create local measure
    const auto tLocalMeasure = std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(
                                 tOnlyDomainDefined,tDataMap,tCellStiffMatrix,"VonMises");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure); 
    
    // evaluate criterion
    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // test result/output
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.00740563, 0.00740563};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }
    // test sum over cell contributions
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(0.0148113, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassCriterion_Evaluate)
{
    // create mesh
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

    //set ad-types
    using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
    using StateT   = typename Residual::StateScalarType;
    using ConfigT  = typename Residual::ConfigScalarType;
    using ResultT  = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    // Create control workset
    const Plato::Scalar tPseudoDensity = 1.0;
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(tPseudoDensity, tControl);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarMultiVector tStates("States", /*numStates=*/ 1, tNumDofs);
    Kokkos::deep_copy(tStates, 0.1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
            { tStates(0, aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal) * 2; }, "fill state");
    
    // create spatial model
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamListTwo, tDataMap);

    // create criterion
    auto tOnlyDomainDefined = tSpatialModel.Domains.front();
    const auto tCriterion =
      std::make_shared<Plato::Elliptic::MassMoment<Residual>>(tOnlyDomainDefined,tDataMap,*tGenericParamListTwo);
    tCriterion->setCalculationType("Mass");

    // Append mass function
    const auto tPhysicsScalarFuncMass =
      std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tri3>>>(tSpatialModel, tDataMap);
    tPhysicsScalarFuncMass->setEvaluator(tCriterion, tOnlyDomainDefined.getDomainName());

    // create weighted sum function
    Plato::Elliptic::WeightedSumFunction<Plato::Mechanics<Plato::Tri3>> tWeightedSum(tSpatialModel, tDataMap);
    const Plato::Scalar tMassFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncMass);
    tWeightedSum.appendFunctionWeight(tMassFunctionWeight);

    // evaluate function
    Plato::Solutions tSolution;
    tSolution.set("State", tStates);
    auto tObjFuncVal = tWeightedSum.value(tSolution, tControl, 0.0);

    // gold mass
    Plato::Scalar tMaterialDensity = 0.5;
    Plato::Scalar tMassGoldValue = pow(static_cast<Plato::Scalar>(tMeshWidth), tSpaceDim)
                                   * tPseudoDensity * tMassFunctionWeight * tMaterialDensity;
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(0.375, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StrengthConstraintCriterion_Evaluate_VonMises3D)
{
    // create mesh
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    //set ad-types
    using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tet4>>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    // create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    TEST_EQUALITY(6,tNumCells);
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);
    // create control workset
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);
    // create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::blas1::fill(0.1, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
            {   tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal);}, "fill state");
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);
    // create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);
    
    // create spatial model
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamListTwo, tDataMap);

    // create criterion
    auto tOnlyDomainDefined = tSpatialModel.Domains.front();
    Plato::AugLagStressCriterion<Residual> tCriterion(tOnlyDomainDefined,tDataMap,*tGenericParamListTwo,"My Stress");
    
    // create material model
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    auto tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    
    // create local measure
    const auto tLocalMeasure = std::make_shared<Plato::VonMisesLocalMeasure<Residual>>
                               (tOnlyDomainDefined,tDataMap,tCellStiffMatrix,"VonMises");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure); 
    
    // evaluate criterion
    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // test result/output
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.0718548,0.0718548,0.0718548,0.0718548,0.0718548,0.0718548};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }
    // test sum over cell contributions
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(0.431129, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StrengthConstraintCriterion_VonMises_GradZ_2D)
{
    // create mesh
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);

    // create weighthed sum scalar function
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamListTwo, tDataMap);
    auto tOnlyDomainDefined = tSpatialModel.Domains.front();
    Plato::Elliptic::WeightedSumFunction<Plato::Mechanics<Plato::Tri3>> tWeightedSum(tSpatialModel, tDataMap);

    // set ad-types
    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;
    using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;
  
    // create material model
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);

    // create von mises local measure
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    const auto tLocalMeasureGradZ = std::make_shared<Plato::VonMisesLocalMeasure<GradientZ>> 
                                      (tOnlyDomainDefined, tDataMap, tCellStiffMatrix, "VonMises");
    const auto tLocalMeasurePODType = std::make_shared<Plato::VonMisesLocalMeasure<Residual>> 
                                        (tOnlyDomainDefined, tDataMap, tCellStiffMatrix, "VonMises");

    // create stress criterion
    const auto tCriterionResidual = std::make_shared<Plato::AugLagStressCriterion<Residual>>(
                                      tOnlyDomainDefined,tDataMap,*tGenericParamListTwo,"My Stress");
    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    const auto tCriterionGradZ = std::make_shared<Plato::AugLagStressCriterion<GradientZ>>(
                                    tOnlyDomainDefined,tDataMap,*tGenericParamListTwo,"My Stress");
    tCriterionGradZ->setLocalMeasure(tLocalMeasureGradZ, tLocalMeasurePODType);

    // append stress criterion to weighthed scalar function
    const auto tPhysicsScalarFuncVonMises =
       std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tri3>>>(tSpatialModel,tDataMap);
    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionResidual, tOnlyDomainDefined.getDomainName());
    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionGradZ, tOnlyDomainDefined.getDomainName());
    const Plato::Scalar tVonMisesFunctionWeight = 1.0;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncVonMises);
    tWeightedSum.appendFunctionWeight(tVonMisesFunctionWeight);
    // test partial wrt control variables
    Plato::test_partial_control<GradientZ,ElementType>(tMesh, tWeightedSum);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StrengthConstraintCriterion_VonMises_GradZ_3D)
{
    // create mesh
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    // create weighthed sum scalar function
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamListTwo, tDataMap);
    auto tOnlyDomainDefined = tSpatialModel.Domains.front();
    Plato::Elliptic::WeightedSumFunction<Plato::Mechanics<Plato::Tet4>> tWeightedSum(tSpatialModel, tDataMap);

    // set ad-types
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;
  
    // create material model
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);

    // create von mises local measure
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    const auto tLocalMeasureGradZ = std::make_shared<Plato::VonMisesLocalMeasure<GradientZ>> 
                                      (tOnlyDomainDefined, tDataMap, tCellStiffMatrix, "VonMises");
    const auto tLocalMeasurePODType = std::make_shared<Plato::VonMisesLocalMeasure<Residual>> 
                                        (tOnlyDomainDefined, tDataMap, tCellStiffMatrix, "VonMises");

    // create stress criterion
    const auto tCriterionResidual = std::make_shared<Plato::AugLagStressCriterion<Residual>>(
                                      tOnlyDomainDefined,tDataMap,*tGenericParamListTwo,"My Stress");
    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    const auto tCriterionGradZ = std::make_shared<Plato::AugLagStressCriterion<GradientZ>>(
                                    tOnlyDomainDefined,tDataMap,*tGenericParamListTwo,"My Stress");
    tCriterionGradZ->setLocalMeasure(tLocalMeasureGradZ, tLocalMeasurePODType);

    // append stress criterion to weighthed scalar function
    const auto tPhysicsScalarFuncVonMises =
       std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tet4>>>(tSpatialModel,tDataMap);
    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionResidual, tOnlyDomainDefined.getDomainName());
    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionGradZ, tOnlyDomainDefined.getDomainName());
    const Plato::Scalar tVonMisesFunctionWeight = 1.0;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncVonMises);
    tWeightedSum.appendFunctionWeight(tVonMisesFunctionWeight);
    // test partial wrt control variables
    Plato::test_partial_control<GradientZ,ElementType>(tMesh, tWeightedSum);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StrengthConstraintCriterion_VonMises_GradU_2D)
{
    // create mesh
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);

    // create weighthed sum scalar function
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamListTwo, tDataMap);
    auto tOnlyDomainDefined = tSpatialModel.Domains.front();
    Plato::Elliptic::WeightedSumFunction<Plato::Mechanics<Plato::Tri3>> tWeightedSum(tSpatialModel, tDataMap);

    // set ad-types
    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;
    using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using GradientU = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
  
    // create material model
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);

    // create von mises local measure
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    const auto tLocalMeasureGradZ = std::make_shared<Plato::VonMisesLocalMeasure<GradientU>> 
                                      (tOnlyDomainDefined, tDataMap, tCellStiffMatrix, "VonMises");
    const auto tLocalMeasurePODType = std::make_shared<Plato::VonMisesLocalMeasure<Residual>> 
                                        (tOnlyDomainDefined, tDataMap, tCellStiffMatrix, "VonMises");

    // create stress criterion
    const auto tCriterionResidual = std::make_shared<Plato::AugLagStressCriterion<Residual>>(
                                      tOnlyDomainDefined,tDataMap,*tGenericParamListTwo,"My Stress");
    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    const auto tCriterionGradZ = std::make_shared<Plato::AugLagStressCriterion<GradientU>>(
                                    tOnlyDomainDefined,tDataMap,*tGenericParamListTwo,"My Stress");
    tCriterionGradZ->setLocalMeasure(tLocalMeasureGradZ, tLocalMeasurePODType);

    // append stress criterion to weighthed scalar function
    const auto tPhysicsScalarFuncVonMises =
       std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tri3>>>(tSpatialModel,tDataMap);
    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionResidual, tOnlyDomainDefined.getDomainName());
    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionGradZ, tOnlyDomainDefined.getDomainName());
    const Plato::Scalar tVonMisesFunctionWeight = 1.0;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncVonMises);
    tWeightedSum.appendFunctionWeight(tVonMisesFunctionWeight);
    // test partial wrt control variables
    Plato::test_partial_state<GradientU,ElementType>(tMesh, tWeightedSum);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StrengthConstraintCriterion_VonMises_GradU_3D)
{
    // create mesh
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    // create weighthed sum scalar function
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamListTwo, tDataMap);
    auto tOnlyDomainDefined = tSpatialModel.Domains.front();
    Plato::Elliptic::WeightedSumFunction<Plato::Mechanics<Plato::Tet4>> tWeightedSum(tSpatialModel, tDataMap);

    // set ad-types
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using GradientU = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
  
    // create material model
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);

    // create von mises local measure
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    const auto tLocalMeasureGradZ = std::make_shared<Plato::VonMisesLocalMeasure<GradientU>> 
                                      (tOnlyDomainDefined, tDataMap, tCellStiffMatrix, "VonMises");
    const auto tLocalMeasurePODType = std::make_shared<Plato::VonMisesLocalMeasure<Residual>> 
                                        (tOnlyDomainDefined, tDataMap, tCellStiffMatrix, "VonMises");

    // create stress criterion
    const auto tCriterionResidual = std::make_shared<Plato::AugLagStressCriterion<Residual>>(
                                      tOnlyDomainDefined,tDataMap,*tGenericParamListTwo,"My Stress");
    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    const auto tCriterionGradZ = std::make_shared<Plato::AugLagStressCriterion<GradientU>>(
                                    tOnlyDomainDefined,tDataMap,*tGenericParamListTwo,"My Stress");
    tCriterionGradZ->setLocalMeasure(tLocalMeasureGradZ, tLocalMeasurePODType);

    // append stress criterion to weighthed scalar function
    const auto tPhysicsScalarFuncVonMises =
       std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tet4>>>(tSpatialModel,tDataMap);
    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionResidual, tOnlyDomainDefined.getDomainName());
    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionGradZ, tOnlyDomainDefined.getDomainName());
    const Plato::Scalar tVonMisesFunctionWeight = 1.0;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncVonMises);
    tWeightedSum.appendFunctionWeight(tVonMisesFunctionWeight);
    // test partial wrt control variables
    Plato::test_partial_state<GradientU,ElementType>(tMesh, tWeightedSum);
}

}
// namespace AugLagStressCriterionTest