#pragma once

#include "ToMap.hpp"
#include "FadTypes.hpp"
#include "ScalarGrad.hpp"
#include "UtilsTeuchos.hpp"
#include "GradientMatrix.hpp"
#include "PlatoMathHelpers.hpp"
#include "ImplicitFunctors.hpp"
#include "InterpolateFromNodal.hpp"
#include "GeneralFluxDivergence.hpp"
#include "SurfaceIntegralUtilities.hpp"

#include "utilities/SurfaceArea.hpp"

#include "helmholtz/AddMassTerm.hpp"
#include "helmholtz/HelmholtzFlux.hpp"
#include "helmholtz/EvaluationTypes.hpp"
#include "helmholtz/HelmholtzElement.hpp"
#include "helmholtz/AbstractVectorFunction.hpp"

namespace Plato
{

namespace Helmholtz
{

/******************************************************************************/
template<typename EvaluationType>
class HelmholtzResidual : 
  public EvaluationType::ElementType,
  public Plato::Helmholtz::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumDofsPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumNodesPerFace;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDimsOnFace;

    using FunctionBaseType = Plato::Helmholtz::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;
    
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

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
        FunctionBaseType(aSpatialDomain, aDataMap)
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
      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      // create a bunch of functors:
      Plato::ComputeGradientMatrix<ElementType>    computeGradient;
      Plato::ScalarGrad<ElementType>               scalarGrad;
      Plato::Helmholtz::HelmholtzFlux<ElementType> helmholtzFlux(mLengthScale);
      Plato::GeneralFluxDivergence<ElementType>    fluxDivergence;
      Plato::Helmholtz::AddMassTerm<ElementType>   addMassTerm;

      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode> tInterpolateFromNodal;

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      Kokkos::parallel_for("helmholtz residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        ConfigScalarType tVolume(0.0);
        StateScalarType tFilteredDensity;
        ControlScalarType tUnfilteredDensity;
        Plato::Array<ElementType::mNumSpatialDims, GradScalarType> tGrad;
        Plato::Array<ElementType::mNumSpatialDims, ResultScalarType> tFlux;

        Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

        auto tCubPoint = tCubPoints(iGpOrdinal);

        computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

        tVolume *= tCubWeights(iGpOrdinal);
        
        // compute filtered and unfiltered densities
        //
        auto tBasisValues = ElementType::basisValues(tCubPoint);
        tInterpolateFromNodal(iCellOrdinal, tBasisValues, aState, tFilteredDensity);
        tInterpolateFromNodal(iCellOrdinal, tBasisValues, aControl, tUnfilteredDensity);

        // compute filtered density gradient
        //
        scalarGrad(iCellOrdinal, tGrad, aState, tGradient);
    
        // compute flux (scale by length scale squared)
        //
        helmholtzFlux(iCellOrdinal, tFlux, tGrad);
    
        // compute flux divergence
        //
        fluxDivergence(iCellOrdinal, aResult, tFlux, tGradient, tVolume);
        
        // add mass term
        //
        addMassTerm(iCellOrdinal, aResult, tFilteredDensity, tUnfilteredDensity, tBasisValues, tVolume);

      });
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
      Plato::SurfaceArea<ElementType> surfaceArea;

      // get sideset faces
      auto tElementOrds = aSpatialModel.Mesh->GetSideSetElementsComplement(mSymmetryPlaneSides);
      auto tNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodesComplement(mSymmetryPlaneSides);
      Plato::OrdinalType tNumFaces = tElementOrds.size();

      // evaluate integral
      auto tLengthScale = mLengthScale;
      const auto tNodesPerFace = mNumNodesPerFace;
      auto tSurfaceLengthScale = mSurfaceLengthScale;
      auto tCubatureWeights = ElementType::Face::getCubWeights();
      auto tCubaturePoints  = ElementType::Face::getCubPoints();
      auto tNumPoints = tCubatureWeights.size();

      Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
      {
          auto tElementOrdinal = tElementOrds(aSideOrdinal);

          Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
          for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<tNodesPerFace; tNodeOrd++)
          {
              tLocalNodeOrds(tNodeOrd) = tNodeOrds(aSideOrdinal*tNodesPerFace+tNodeOrd);
          }

          auto tCubatureWeight = tCubatureWeights(aPointOrdinal);
          auto tCubaturePoint = tCubaturePoints(aPointOrdinal);
          auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);
          auto tBasisGrads  = ElementType::Face::basisGrads(tCubaturePoint);

          // calculate surface jacobians
          ResultScalarType tSurfaceArea(0.0);
          surfaceArea(tElementOrdinal, tLocalNodeOrds, tBasisGrads, aConfig, tSurfaceArea);
          tSurfaceArea *= tCubatureWeight;

          // project filtered density field onto surface
          StateScalarType tFilteredDensity(0.0);
          for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++)
          {
            auto tLocalCellNode = tLocalNodeOrds(tNode);
            tFilteredDensity += tBasisValues(tNode) * aState(tElementOrdinal, tLocalCellNode);
          }

          for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++ )
          {
            auto tLocalCellNode = tLocalNodeOrds(tNode);
            Kokkos::atomic_add(&aResult(tElementOrdinal, tLocalCellNode), tSurfaceLengthScale * tLengthScale * tFilteredDensity *
              tBasisValues(tNode) * tSurfaceArea);
          }
      }, "add surface mass to left-hand-side");
    }
};
// class HelmholtzResidual

} // namespace Helmholtz

} // namespace Plato

#include "helmholtz/ExpInstMacros.hpp"

PLATO_HELMHOLTZ_DEF_3(Plato::Helmholtz::HelmholtzResidual, Plato::HelmholtzElement)
