#pragma once

#include "ToMap.hpp"
#include "ScalarGrad.hpp"
#include "UtilsOmegaH.hpp"
#include "UtilsTeuchos.hpp"
#include "FluxDivergence.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoMathHelpers.hpp"
#include "SimplexFadTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "InterpolateFromNodal.hpp"
#include "SurfaceIntegralUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "helmholtz/AddMassTerm.hpp"
#include "helmholtz/HelmholtzFlux.hpp"
#include "helmholtz/SimplexHelmholtz.hpp"
#include "helmholtz/AbstractVectorFunction.hpp"

namespace Plato
{

namespace Helmholtz
{

/******************************************************************************/
template<typename EvaluationType>
class HelmholtzResidual : 
  public Plato::SimplexHelmholtz<EvaluationType::SpatialDim>,
  public Plato::Helmholtz::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    using PhysicsType = typename Plato::SimplexHelmholtz<mSpaceDim>;
    using PhysicsType::mNumDofsPerCell;
    using PhysicsType::mNumDofsPerNode;
    using PhysicsType::mNumNodesPerFace;
    using PhysicsType::mNumSpatialDimsOnFace;

    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;

    using Plato::Helmholtz::AbstractVectorFunction<EvaluationType>::mSpatialDomain;
    using Plato::Helmholtz::AbstractVectorFunction<EvaluationType>::mDataMap;
    
    /*!< local automatic differentiaton parameters */
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::LinearTetCubRuleDegreeOne<mSpaceDim> mCubatureRule; /*!< volume cubature rule */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace> mSurfaceCubatureRule; /*!< surface cubature rule */

    Plato::Scalar mLengthScale = 0.5; /*!< volume length scale */
    Plato::Scalar mSurfaceLengthScale = 0.0; /*!< surface length scale multiplier, 0 \leq \alpha \leq 1 */
    std::vector<std::string> mSymmetryPlaneSides; /*!< entity sets where symmetry constraints are applied */

  public:
    /**************************************************************************/
    HelmholtzResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams
    ) :
        Plato::Helmholtz::AbstractVectorFunction<EvaluationType>(aSpatialDomain, aDataMap),
        mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mSpaceDim>()),
        mSurfaceCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>())
    /**************************************************************************/
    {
        // parse length scale parameter
        if (!aProblemParams.isSublist("Parameters"))
        {
            ANALYZE_THROWERR("NO PARAMETERS SUBLIST WAS PROVIDED FOR THE HELMHOLTZ FILTER.");
        }
        else
        {
          auto tParamList = aProblemParams.get < Teuchos::ParameterList > ("Parameters");
          mLengthScale = tParamList.get<Plato::Scalar>("Length Scale", 0.5);
          mSurfaceLengthScale = tParamList.get<Plato::Scalar>("Surface Length Scale", 0.0);
          mSymmetryPlaneSides = Plato::teuchos::parse_array<std::string>("Symmetry Plane Sides", tParamList);
        }
    }

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override
    {
      Plato::ScalarMultiVector tFilteredDensity = aSolutions.get("State");
      Plato::Solutions tSolutionsOutput(aSolutions.physics(), aSolutions.pde());
      tSolutionsOutput.set("Filtered Density", tFilteredDensity);
      tSolutionsOutput.setNumDofs("Filtered Density", 1);
      return tSolutionsOutput;
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexHelmholtz<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Kokkos::View<GradScalarType**, Plato::Layout, Plato::MemSpace>
        tGrad("filtered density gradient",tNumCells,mSpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("basis function gradient",tNumCells,mNumNodesPerCell,mSpaceDim);

      Kokkos::View<ResultScalarType**, Plato::Layout, Plato::MemSpace>
        tFlux("filtered density flux",tNumCells,mSpaceDim);

      Plato::ScalarVectorT<StateScalarType> 
        tFilteredDensity("Gauss point filtered density", tNumCells);

      Plato::ScalarVectorT<ControlScalarType> 
        tUnfilteredDensity("Gauss point unfiltered density", tNumCells);

      // create a bunch of functors:
      Plato::ComputeGradientWorkset<mSpaceDim>    tComputeGradient;
      Plato::ScalarGrad<mSpaceDim>                tScalarGrad;
      Plato::Helmholtz::HelmholtzFlux<mSpaceDim>  tHelmholtzFlux(mLengthScale);
      Plato::FluxDivergence<mSpaceDim>            tFluxDivergence;
      Plato::Helmholtz::AddMassTerm<mSpaceDim>    tAddMassTerm;

      Plato::InterpolateFromNodal<mSpaceDim, mNumDofsPerNode> tInterpolateFromNodal;

      auto tQuadratureWeight = mCubatureRule.getCubWeight();
      auto tBasisFunctions   = mCubatureRule.getBasisFunctions();

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;
        
        // compute filtered and unfiltered densities
        //
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aState, tFilteredDensity);
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aControl, tUnfilteredDensity);

        // compute filtered density gradient
        //
        tScalarGrad(aCellOrdinal, tGrad, aState, tGradient);
    
        // compute flux (scale by length scale squared)
        //
        tHelmholtzFlux(aCellOrdinal, tFlux, tGrad);
    
        // compute flux divergence
        //
        tFluxDivergence(aCellOrdinal, aResult, tFlux, tGradient, tCellVolume);
        
        // add mass term
        //
        tAddMassTerm(aCellOrdinal, aResult, tFilteredDensity, tUnfilteredDensity, tBasisFunctions, tCellVolume);

      },"helmholtz residual");
    }

    /**************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                           & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      if(mSurfaceLengthScale <= static_cast<Plato::Scalar>(0.0))
        { return; }

      // set local functors
      Plato::CalculateSurfaceArea<mSpaceDim> tCalculateSurfaceArea;
      Plato::NodeCoordinate<mSpaceDim> tCoords(aSpatialModel.Mesh);
      Plato::CalculateSurfaceJacobians<mSpaceDim> tCalculateSurfaceJacobians;

      // get sideset faces
      auto tElementOrds = aSpatialModel.Mesh->GetSideSetElementsComplement(mSymmetryPlaneSides);
      auto tNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodesComplement(mSymmetryPlaneSides);
      auto tNumFaces = tElementOrds.size();

      Plato::ScalarArray3DT<ConfigScalarType> tJacobians("jacobian", tNumFaces, mNumSpatialDimsOnFace, mSpaceDim);
      auto tNumCells = aSpatialModel.Mesh->NumElements();
      Plato::ScalarVectorT<StateScalarType> tFilteredDensity("filtered density", tNumCells);

      // evaluate integral
      auto tLengthScale = mLengthScale;
      const auto tNodesPerFace = mNumNodesPerFace;
      auto tSurfaceLengthScale = mSurfaceLengthScale;
      auto tSurfaceCubatureWeight = mSurfaceCubatureRule.getCubWeight();
      auto tSurfaceBasisFunctions = mSurfaceCubatureRule.getBasisFunctions();
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aSideOrdinal)
      {
          auto tElementOrdinal = tElementOrds(aSideOrdinal);

          Plato::OrdinalType tLocalNodeOrds[tNodesPerFace];
          for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<tNodesPerFace; tNodeOrd++)
          {
              tLocalNodeOrds[tNodeOrd] = tNodeOrds(aSideOrdinal*tNodesPerFace+tNodeOrd);
          }

          // calculate surface jacobians
          ConfigScalarType tSurfaceAreaTimesCubWeight(0.0);
          tCalculateSurfaceJacobians(tElementOrdinal, aSideOrdinal, tLocalNodeOrds, aConfig, tJacobians);
          tCalculateSurfaceArea(aSideOrdinal, tSurfaceCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

          // project filtered density field onto surface
          tFilteredDensity(tElementOrdinal) = 0.0;
          for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++)
          {
            auto tLocalCellNode = tLocalNodeOrds[tNode];
            tFilteredDensity(tElementOrdinal) += tSurfaceBasisFunctions(tNode) * aState(tElementOrdinal, tLocalCellNode);
          }

          for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++ )
          {
            auto tLocalCellNode = tLocalNodeOrds[tNode];
            aResult(tElementOrdinal, tLocalCellNode) += tSurfaceLengthScale * tLengthScale * tFilteredDensity(tElementOrdinal) *
              tSurfaceBasisFunctions(tNode) * tSurfaceAreaTimesCubWeight;
          }
      }, "add surface mass to left-hand-side");
    }
};
// class HelmholtzResidual

} // namespace Helmholtz

} // namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::Helmholtz::HelmholtzResidual<Plato::ResidualTypes<Plato::SimplexHelmholtz<1>>>;
extern template class Plato::Helmholtz::HelmholtzResidual<Plato::JacobianTypes<Plato::SimplexHelmholtz<1>>>;
#endif
#ifdef PLATOANALYZE_2D
extern template class Plato::Helmholtz::HelmholtzResidual<Plato::ResidualTypes<Plato::SimplexHelmholtz<2>>>;
extern template class Plato::Helmholtz::HelmholtzResidual<Plato::JacobianTypes<Plato::SimplexHelmholtz<2>>>;
#endif
#ifdef PLATOANALYZE_3D
extern template class Plato::Helmholtz::HelmholtzResidual<Plato::ResidualTypes<Plato::SimplexHelmholtz<3>>>;
extern template class Plato::Helmholtz::HelmholtzResidual<Plato::JacobianTypes<Plato::SimplexHelmholtz<3>>>;
#endif
