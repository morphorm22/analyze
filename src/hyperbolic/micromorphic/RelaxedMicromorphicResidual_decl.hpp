#pragma once

#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "ApplyWeighting.hpp"

#include "hyperbolic/VectorFunction.hpp"
#include "hyperbolic/EvaluationTypes.hpp"

#include "hyperbolic/micromorphic/MicromorphicElasticModelFactory.hpp"
#include "hyperbolic/micromorphic/MicromorphicInertiaModelFactory.hpp"

namespace Plato
{

namespace Hyperbolic
{

template<typename EvaluationType, typename IndicatorFunctionType>
class RelaxedMicromorphicResidual :
  public EvaluationType::ElementType,
  public Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>
{
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumSpatialDims;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;

    using ElementType::mNumFullTerms;
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumSkwTerms;

    using FunctionBaseType = Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using StateScalarType       = typename EvaluationType::StateScalarType;
    using StateDotScalarType    = typename EvaluationType::StateDotScalarType;
    using StateDotDotScalarType = typename EvaluationType::StateDotDotScalarType;
    using ControlScalarType     = typename EvaluationType::ControlScalarType;
    using ConfigScalarType      = typename EvaluationType::ConfigScalarType;
    using ResultScalarType      = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyMassWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;
    std::shared_ptr<Plato::NaturalBCs<ElementType>> mBoundaryLoads;

    bool mRayleighDamping;

    Teuchos::RCP<Plato::MicromorphicLinearElasticMaterial<mNumSpatialDims>> mMaterialModel;
    Teuchos::RCP<Plato::MicromorphicInertiaMaterial<mNumSpatialDims>>       mInertiaModel;

    std::vector<std::string> mPlotTable;

  public:
    RelaxedMicromorphicResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ); 

    void
    checkTimeIntegrator(Teuchos::ParameterList & aIntegratorParams);

    Plato::Scalar
    getMaxEigenvalue(const Plato::ScalarArray3D & aConfig) const override;

    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override;

    void
    evaluate(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0,
              Plato::Scalar aCurrentTime = 0.0
    ) const override;

    void
    evaluateWithoutDamping(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0,
              Plato::Scalar aCurrentTime = 0.0
    ) const;

    void
    evaluateWithDamping(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0,
              Plato::Scalar aCurrentTime = 0.0
    ) const;

    void
    evaluate_boundary(
        const Plato::SpatialModel                                & aSpatialModel,
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0,
              Plato::Scalar aCurrentTime = 0.0
    ) const override;
};

}

}

