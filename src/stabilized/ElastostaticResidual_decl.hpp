#pragma once

#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "ApplyWeighting.hpp"
#include "ElasticModelFactory.hpp"
#include "stabilized/AbstractVectorFunction.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************//**
 * \brief Stabilized elastostatic residual (reference: M. Chiumenti et al. (2004))
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ElastostaticResidual :
        public EvaluationType::ElementType,
        public Plato::Stabilized::AbstractVectorFunction<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumSpatialDims;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumMechDims  = mNumSpatialDims;
    static constexpr Plato::OrdinalType mNumPressDims = 1;
    static constexpr Plato::OrdinalType mMechDofOffset = 0;
    static constexpr Plato::OrdinalType mPressDofOffset = mNumSpatialDims;

    using FunctionBaseType = Plato::Stabilized::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction; /*!< material penalty function */
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms,  IndicatorFunctionType> mApplyTensorWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyVectorWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, 1 /* number of pressure dofs per node */, IndicatorFunctionType> mApplyScalarWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads; /*!< body loads interface */
    std::shared_ptr<Plato::NaturalBCs<ElementType, mNumMechDims, mNumDofsPerNode, mMechDofOffset>> mBoundaryLoads; /*!< boundary loads interface */

    Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel; /*!< material model interface */

    std::vector<std::string> mPlottable; /*!< array with output data identifiers */

private:
    /******************************************************************************//**
     * \brief initialize material, loads and output data
     * \param [in] aProblemParams input XML data
    **********************************************************************************/
    void initialize(Teuchos::ParameterList& aProblemParams);

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh mesh metadata
     * \param [in] aDataMap output data map
     * \param [in] aProblemParams input XML data
     * \param [in] aPenaltyParams penalty function input XML data
    **********************************************************************************/
    ElastostaticResidual(
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
     * \param [in] aStateWS state, {disp_x, disp_y, disp_z, pressure}, workset
     * \param [in] aPressGradWS pressure gradient workset
     * \param [in] aControlWS control workset
     * \param [in] aConfigWS configuration workset
     * \param [in/out] aResultWS result, e.g. residual, workset
     * \param [in] aTimeStep time step
    **********************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>     & aStateWS,
        const Plato::ScalarMultiVectorT <NodeStateScalarType> & aPressGradWS,
        const Plato::ScalarMultiVectorT <ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT     <ConfigScalarType>    & aConfigWS,
              Plato::ScalarMultiVectorT <ResultScalarType>    & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate stabilized elastostatics boundary terms residual
     * \param [in] aStateWS state, {disp_x, disp_y, disp_z, pressure}, workset
     * \param [in] aPressGradWS pressure gradient workset
     * \param [in] aControlWS control workset
     * \param [in] aConfigWS configuration workset
     * \param [in/out] aResultWS result, e.g. residual, workset
     * \param [in] aTimeStep time step
    **********************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                             & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType>     & aStateWS,
        const Plato::ScalarMultiVectorT <NodeStateScalarType> & aPressGradWS,
        const Plato::ScalarMultiVectorT <ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT     <ConfigScalarType>    & aConfigWS,
              Plato::ScalarMultiVectorT <ResultScalarType>    & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override;
};
// class ElastostaticResidual

} // namespace Stabilized
} // namespace Plato
