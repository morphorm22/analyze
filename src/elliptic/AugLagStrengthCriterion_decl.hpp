/*
 * AugLagStrengthCriterion_decl.hpp
 *
 *  Created on: May 4, 2023
 */

#pragma once

#include <string>
#include <memory>

#include <Teuchos_ParameterList.hpp>

#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"
#include "AbstractLocalMeasure.hpp"

#include "optimizer/AugLagDataMng.hpp"
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
class AugLagStrengthCriterion :
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

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aSpatialDomain holds spatial domain; i.e., element block, data
     * \param [in] aDataMap       holds output data map
     * \param [in] aParams        input parameters database
     * \param [in] aFuncName      user-defined criterion name
    **********************************************************************************/
    AugLagStrengthCriterion(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParams,
        const std::string            & aFuncName
    );

    /******************************************************************************//**
     * \brief Destructor
    **********************************************************************************/
    ~AugLagStrengthCriterion(){}

    void 
    setLocalMeasure(
        const std::shared_ptr<AbstractLocalMeasure<EvaluationType>> & aInputEvaluationType,
        const std::shared_ptr<AbstractLocalMeasure<Residual>>       & aInputPODType
    );

    void 
    updateProblem(
        const Plato::ScalarMultiVector & aStateWS,
        const Plato::ScalarMultiVector & aControlWS,
        const Plato::ScalarArray3D     & aConfigWS
    );

    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
        const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
              Plato::ScalarVectorT      <ResultT>  & aResultWS,
              Plato::Scalar aTimeStep
    ) const;

private:
    /******************************************************************************//**
     * \brief Allocate member data
     * \param [in] aParams input parameters database
    **********************************************************************************/
    void 
    initialize(
        Teuchos::ParameterList & aParams
    );

    /******************************************************************************//**
     * \brief Parse numeric inputs from input file
     * \param [in] aParams input parameters database
    **********************************************************************************/
    void 
    parseNumerics(
        Teuchos::ParameterList & aParams
    );

    /******************************************************************************//**
     * \brief Parse limits on strength constraint
     * \param [in] aParams input parameters database
    **********************************************************************************/
    void 
    parseLimits(
        Teuchos::ParameterList & aParams
    );

    /******************************************************************************//**
     * \brief Evaluate current local strength constraints
     * \param [in] aStateWS   state workset
     * \param [in] aControlWS control workset
     * \param [in] aConfigWS  configuration workset
    **********************************************************************************/
    void 
    evaluateCurrentConstraints(
        const Plato::ScalarMultiVector &aStateWS,
        const Plato::ScalarMultiVector &aControlWS,
        const Plato::ScalarArray3D     &aConfigWS
    );
};

}
// namespace Plato