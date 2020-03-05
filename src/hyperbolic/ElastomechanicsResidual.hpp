#ifndef HYPERBOLIC_ELASTOMECHANICS_RESIDUAL_HPP
#define HYPERBOLIC_ELASTOMECHANICS_RESIDUAL_HPP
#include "Simp.hpp"
#include "Ramp.hpp"
#include "ToMap.hpp"
#include "Strain.hpp"
#include "Heaviside.hpp"
#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "Plato_Solve.hpp"
#include "LinearStress.hpp"
#include "ProjectToNode.hpp"
#include "ApplyWeighting.hpp"
#include "StressDivergence.hpp"
#include "SimplexMechanics.hpp"
#include "ApplyConstraints.hpp"
#include "PlatoMathHelpers.hpp"
#include "ScalarFunctionBase.hpp"
#include "InterpolateFromNodal.hpp"
#include "PlatoAbstractProblem.hpp"
#include "LinearElasticMaterial.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/Newmark.hpp"
#include "hyperbolic/InertialContent.hpp"
#include "hyperbolic/HyperbolicVectorFunction.hpp"

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class TransientMechanicsResidual :
  public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
  public Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexMechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexMechanics<SpaceDim>::mNumDofsPerCell;
    using Plato::SimplexMechanics<SpaceDim>::mNumDofsPerNode;

    using Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>::mMesh;
    using Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>::mDataMap;
    using Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>::mMeshSets;

    using DisplacementScalarType = typename EvaluationType::DisplacementScalarType;
    using VelocityScalarType     = typename EvaluationType::VelocityScalarType;
    using AccelerationScalarType = typename EvaluationType::AccelerationScalarType;
    using ControlScalarType      = typename EvaluationType::ControlScalarType;
    using ConfigScalarType       = typename EvaluationType::ConfigScalarType;
    using ResultScalarType       = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim, mNumVoigtTerms,  IndicatorFunctionType> mApplyStressWeighting;
    Plato::ApplyWeighting<SpaceDim, SpaceDim,        IndicatorFunctionType> mApplyMassWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType>> mBodyLoads;
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<SpaceDim>> mCubatureRule;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim,mNumDofsPerNode>> mBoundaryLoads;

    Teuchos::RCP<Plato::LinearElasticMaterial<SpaceDim>> mMaterialModel;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    TransientMechanicsResidual(
      Omega_h::Mesh& aMesh,
      Omega_h::MeshSets& aMeshSets,
      Plato::DataMap& aDataMap,
      Teuchos::ParameterList& aProblemParams,
      Teuchos::ParameterList& aPenaltyParams) :
     Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap,
        {"Displacement X", "Displacement Y", "Displacement Z"}),
     mIndicatorFunction(aPenaltyParams),
     mApplyStressWeighting(mIndicatorFunction),
     mApplyMassWeighting(mIndicatorFunction),
     mBodyLoads(nullptr),
     mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<SpaceDim>>()),
     mBoundaryLoads(nullptr)
    /**************************************************************************/
    {
        // create material model and get stiffness
        //
        Plato::ElasticModelFactory<SpaceDim> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create();
        // parse body loads
        // 
        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType>>
                         (aProblemParams.sublist("Body Loads"));
        }

        // parse boundary Conditions
        // 
        if(aProblemParams.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim,mNumDofsPerNode>>
                             (aProblemParams.sublist("Natural Boundary Conditions"));
        }

        auto tResidualParams = aProblemParams.sublist("Hyperbolic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
          mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();

    }

    /**************************************************************************/
    void
    evaluate( const Plato::ScalarMultiVectorT< DisplacementScalarType > & aDisplacement,
              const Plato::ScalarMultiVectorT< VelocityScalarType     > & aVelocity,
              const Plato::ScalarMultiVectorT< AccelerationScalarType > & aAcceleration,
              const Plato::ScalarMultiVectorT< ControlScalarType      > & aControl,
              const Plato::ScalarArray3DT    < ConfigScalarType       > & aConfig,
                    Plato::ScalarMultiVectorT< ResultScalarType       > & aResult,
                    Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto tNumCells = mMesh.nelems();

      using StrainScalarType =
          typename Plato::fad_type_t<Plato::SimplexMechanics<EvaluationType::SpatialDim>, DisplacementScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<SpaceDim> tComputeGradient;
      Plato::Strain<SpaceDim>                 tComputeVoigtStrain;
      Plato::LinearStress<SpaceDim>           tComputeVoigtStress(mMaterialModel);
      Plato::StressDivergence<SpaceDim>       tComputeStressDivergence;

      auto tCellDensity = mMaterialModel->getMassDensity();
      Plato::InertialContent<SpaceDim>        tInertialContent(tCellDensity);

      Plato::ProjectToNode<SpaceDim, mNumDofsPerNode>        tProjectInertialContent;

      Plato::InterpolateFromNodal<SpaceDim, mNumDofsPerNode, /*offset=*/0, SpaceDim> tInterpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType>
        tStrain("strain",tNumCells,mNumVoigtTerms);

      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("gradient",tNumCells,mNumNodesPerCell,SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tStress("stress",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<AccelerationScalarType> 
        tAccelerationGP("acceleration at Gauss point", tNumCells, SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tInertialContentGP("Inertial content at Gauss point", tNumCells, SpaceDim);

      auto tBasisFunctions = mCubatureRule->getBasisFunctions();

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto& tApplyStressWeighting = mApplyStressWeighting;
      auto& tApplyMassWeighting = mApplyMassWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute strain
        tComputeVoigtStrain(aCellOrdinal, tStrain, aDisplacement, tGradient);

        // compute stress
        tComputeVoigtStress(aCellOrdinal, tStress, tStrain);

        // apply weighting
        tApplyStressWeighting(aCellOrdinal, tStress, aControl);

        // compute stress divergence
        tComputeStressDivergence(aCellOrdinal, aResult, tStress, tGradient, tCellVolume);

        // compute accelerations at gausspoints
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aAcceleration, tAccelerationGP);

        // compute inertia at gausspoints
        tInertialContent(aCellOrdinal, tInertialContentGP, tAccelerationGP);

        // apply weighting
        tApplyMassWeighting(aCellOrdinal, tInertialContentGP, aControl);

        // project to nodes
        tProjectInertialContent(aCellOrdinal, tCellVolume, tBasisFunctions, tInertialContentGP, aResult);

      }, "Compute Residual");

      if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) toMap(mDataMap, tStress, "stress");
      if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) toMap(mDataMap, tStress, "strain");

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mMesh, aDisplacement, aControl, aResult, -1.0 );
      }

      if( mBoundaryLoads != nullptr )
      {
          mBoundaryLoads->get( &mMesh, mMeshSets, aDisplacement, aControl, aConfig, aResult, -1.0 );
      }
    }
};

#endif
