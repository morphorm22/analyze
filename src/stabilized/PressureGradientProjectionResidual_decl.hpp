#pragma once

#include "ApplyWeighting.hpp"
#include "stabilized/AbstractVectorFunction.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************//**
 * \brief Evaluate pressure gradient projection residual (reference Chiumenti et al. (2004))
 *
 *               \langle \nabla{p},\eta \rangle - <\Phi,\eta> = 0
 *
 **********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class PressureGradientProjectionResidual :
    public EvaluationType::ElementType,
    public Plato::Stabilized::AbstractVectorFunction<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = Plato::Stabilized::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction; /*!< material penalty function */
    Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyVectorWeighting; /*!< apply penalty to vector function */

    Plato::Scalar mPressureScaling; /*!< Pressure scaling term */

private:
    /***************************************************************************//**
     * \brief Initialize member data
     * \param [in] aProblemParams input XML data, i.e. parameter list
    *******************************************************************************/
    void initialize(Teuchos::ParameterList &aProblemParams);

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aDataMap output data map
     * \param [in] aProblemParams input XML data
     * \param [in] aPenaltyParams penalty function input XML data
    **********************************************************************************/
    PressureGradientProjectionResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    );

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override;

    /******************************************************************************//**
     * \brief Evaluate stabilized elastostatics residual
     * \param [in] aNodalPGradWS pressure gradient workset on H^1(\Omega)
     * \param [in] aPressureWS pressure gradient workset on H^1(\Omega)
     * \param [in] aControlWS control workset
     * \param [in] aConfigWS configuration workset
     * \param [in/out] aResultWS result, e.g. residual workset
     * \param [in] aTimeStep time step
    **********************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>     & aNodalPGradWS,
        const Plato::ScalarMultiVectorT <NodeStateScalarType> & aPressureWS,
        const Plato::ScalarMultiVectorT <ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT     <ConfigScalarType>    & aConfigWS,
              Plato::ScalarMultiVectorT <ResultScalarType>    & aResultWS,
              Plato::Scalar                                     aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Update physics-based data within a frequency of optimization iterations
     * \param [in] aState     global state variables
     * \param [in] aControl   control variables, e.g. design variables
     * \param [in] aTimeStep  pseudo time step
    **********************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aState,
                       const Plato::ScalarVector & aControl,
                       Plato::Scalar aTimeStep = 0.0) override;
};
// class PressureGradientProjectionResidual

} // namespace Stabilized
} // namespace Plato
