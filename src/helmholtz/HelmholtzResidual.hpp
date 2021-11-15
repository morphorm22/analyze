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

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::LinearTetCubRuleDegreeOne<mSpaceDim> mCubatureRule;
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace> mSurfaceCubatureRule;

    Plato::Scalar mLengthScale = 1.0;
    Plato::Scalar mSurfacePenalty = 0.0;
    std::vector<std::string> mSymmetryPlaneSides;

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
          THROWERR("NO PARAMETERS SUBLIST WAS PROVIDED FOR THE HELMHOLTZ FILTER.");
        }
        else
        {
          auto tParamList = aProblemParams.get < Teuchos::ParameterList > ("Parameters");
          mLengthScale = tParamList.get<Plato::Scalar>("Length Scale", 1.0);
          mSurfacePenalty = tParamList.get<Plato::Scalar>("Surface Length Scale", 0.0);
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
      if(mSurfacePenalty <= static_cast<Plato::Scalar>(0.0))
        { return; }

      // get mesh vertices
      auto tFace2Verts = aSpatialModel.Mesh.ask_verts_of(mNumSpatialDimsOnFace);
      auto tCell2Verts = aSpatialModel.Mesh.ask_elem_verts();

      // get face to element graph
      auto tFace2eElems = aSpatialModel.Mesh.ask_up(mNumSpatialDimsOnFace, mSpaceDim);
      auto tFace2Elems_map   = tFace2eElems.a2ab;
      auto tFace2Elems_elems = tFace2eElems.ab2b;

      // get element to face map
      auto tElem2Faces = aSpatialModel.Mesh.ask_down(mSpaceDim, mNumSpatialDimsOnFace).ab2b;

      // set local functors
      Plato::CalculateSurfaceArea<mSpaceDim> tCalculateSurfaceArea;
      Plato::NodeCoordinate<mSpaceDim> tCoords(&(aSpatialModel.Mesh));
      Plato::CalculateSurfaceJacobians<mSpaceDim> tCalculateSurfaceJacobians;
      Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<mSpaceDim> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

      // get sideset faces
      auto tFaceLocalOrdinals = 
        Plato::omega_h::get_boundary_entities<mNumSpatialDimsOnFace,Omega_h::SIDE_SET>(mSymmetryPlaneSides, aSpatialModel.Mesh, aSpatialModel.MeshSets);
      auto tNumFaces = tFaceLocalOrdinals.size();
      Plato::ScalarArray3DT<ConfigScalarType> tJacobians("jacobian", tNumFaces, mNumSpatialDimsOnFace, mSpaceDim);
      auto tNumCells = aSpatialModel.Mesh.nelems();
      Plato::ScalarVectorT<StateScalarType> tFilteredDensity("filtered density", tNumCells);

      // evaluate integral
      auto tLengthScale = mLengthScale;
      auto tSurfaceCubatureWeight = mSurfaceCubatureRule.getCubWeight();
      auto tSurfaceBasisFunctions = mSurfaceCubatureRule.getBasisFunctions();
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
      {
        auto tFaceOrdinal = tFaceLocalOrdinals[aFaceI];

        // for each element that the face is connected to: (either 1 or 2 elements)
        for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal + 1]; tElem++ )
        {
          // create a map from face local node index to elem local node index
          Plato::OrdinalType tLocalNodeOrd[mSpaceDim];
          auto tCellOrdinal = tFace2Elems_elems[tElem];
          tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrd);

          // calculate surface jacobians
          ConfigScalarType tSurfaceAreaTimesCubWeight(0.0);
          tCalculateSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrd, aConfig, tJacobians);
          tCalculateSurfaceArea(aFaceI, tSurfaceCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

          // project filtered density field onto surface
          tFilteredDensity(tCellOrdinal) = 0.0;
          for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++)
          {
            auto tLocalCellNode = tLocalNodeOrd[tNode];
            tFilteredDensity(tCellOrdinal) += tSurfaceBasisFunctions(tNode) * aState(tCellOrdinal, tLocalCellNode);
          }

          for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++ )
          {
            auto tLocalCellNode = tLocalNodeOrd[tNode];
            aResult(tCellOrdinal, tLocalCellNode) += tLengthScale * tFilteredDensity(tCellOrdinal) *
              tSurfaceBasisFunctions(tNode) * tSurfaceAreaTimesCubWeight;
          }
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
