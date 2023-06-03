/*
 * MomentumSurfaceForces.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include "BLAS2.hpp"
#include "WorkSets.hpp"
#include "NaturalBCs.hpp"
#include "UtilsOmegaH.hpp"
#include "SpatialModel.hpp"
#include "ExpInstMacros.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/fluids/AbstractVolumeIntegrand.hpp"
#include "hyperbolic/fluids/SimplexFluids.hpp"
#include "hyperbolic/fluids/SimplexFluidsFadTypes.hpp"

namespace Plato
{

namespace Fluids
{

/******************************************************************************//**
 * \tparam SpaceDims (integer) number of spatial dimensions
 * \tparam NumNodes  (integer) number of nodes on surface/face
 * \tparam InStateType  input state type
 * \tparam OutStateType output state type
 *
 * \fn device_type inline project_vector_field_onto_surface
 *
 * \brief Project vector field onto surface/face
 *
 * \param [in] aCellOrdinal       cell/element ordinal
 * \param [in] aBasisFunctions    basis functions
 * \param [in] aLocalNodeOrdinals local cell node ordinals
 * \param [in] aInputState        input state
 * \param [in/out] aInputState    output state
**********************************************************************************/
template<Plato::OrdinalType SpaceDims,
         Plato::OrdinalType NumNodes,
         typename InStateType,
         typename OutStateType>
KOKKOS_INLINE_FUNCTION void
project_vector_field_onto_surface
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::OrdinalType aLocalNodeOrdinals[NumNodes],
 const Plato::ScalarMultiVectorT<InStateType> & aInputState,
 const Plato::ScalarMultiVectorT<OutStateType> & aOutputState)
{
    for(Plato::OrdinalType tDim = 0; tDim < SpaceDims; tDim++)
    {
        aOutputState(aCellOrdinal, tDim) = 0.0;
        for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
        {
            auto tLocalCellNode = aLocalNodeOrdinals[tNode];
            auto tLocalCellDof = (SpaceDims * tLocalCellNode) + tDim;
            aOutputState(aCellOrdinal, tDim) +=
                aBasisFunctions(tNode) * aInputState(aCellOrdinal, tLocalCellDof);
        }
    }
}
// function project_vector_field_onto_surface

/***************************************************************************//**
 * \class MomentumSurfaceForces
 *
 * \tparam PhysicsT    Physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \brief Class responsible for the evaluation of the surface momentum forces.
 *   This surface integral is evaluated during the calculation of the pressure
 *   residual (i.e. mass conservation equation).
 *
 * \f[
 *   \alpha\int_{\Gamma_e} v^h \left( u_i^n n_i \right) d\Gamma_e
 * \f]
 *
 * where \f$ v^h \f$ is the test function, \f$ u_i^n \f$ is the i-th velocity
 * component at time step n, \f$\f$ is the i-th unit normal component and
 * \f$ \alpha \f$ is a scalar multiplier.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class MomentumSurfaceForces
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace; /*!< number of nodes per face */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace; /*!< number of spatial dimensions on face */

    // forward automatic differentiation types
    using ResultT = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */
    using ConfigT = typename EvaluationT::ConfigScalarType; /*!< configuration FAD evaluation type */
    using PrevVelT = typename EvaluationT::PreviousMomentumScalarType; /*!< previous momentum FAD evaluation type */

    const std::string mEntitySetName; /*!< entity set name, defined by the surfaces where Dirichlet boundary conditions are applied */
    const Plato::SpatialDomain& mSpatialDomain; /*!< spatial domain metadata */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace> mSurfaceCubatureRule; /*!< surface cubature integration rule */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain        spatial domain metadata
     * \param [in] aEntitySetName entity set name (e.g. side set name)
     ******************************************************************************/
    MomentumSurfaceForces
    (const Plato::SpatialDomain & aDomain,
     const std::string & aEntitySetName) :
         mEntitySetName(aEntitySetName),
         mSpatialDomain(aDomain)
    {
    }

    /***************************************************************************//**
     * \fn void operator()
     * \brief Evaluate surface integral.
     * \param [in] aWorkSets   holds input worksets (e.g. states, control, etc)
     * \param [in] aMultiplier scalar multiplier (default = 1.0)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void operator()
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult,
     Plato::Scalar aMultiplier = 1.0) const
    {
        auto tElementOrds = mSpatialDomain.Mesh->GetSideSetElements(mEntitySetName);
        auto tFaceOrds    = mSpatialDomain.Mesh->GetSideSetFaces(mEntitySetName);
        auto tNodeOrds    = mSpatialDomain.Mesh->GetSideSetLocalNodes(mEntitySetName);

        auto tNumFaces = tElementOrds.size();

        // set local functors
        Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(mSpatialDomain.Mesh);
        Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode,   0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        Plato::ScalarArray3DT<ConfigT> tJacobians("jacobian", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);
        auto tNumCells = mSpatialDomain.Mesh->NumElements();
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity", tNumCells, mNumVelDofsPerNode);

        // set input state worksets
        auto tConfigWS = Plato::unpack<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS = Plato::unpack<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tCriticalTimeStep = Plato::unpack<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // evaluate integral
        auto tSurfaceCubatureWeight = mSurfaceCubatureRule.getCubWeight();
        auto tSurfaceBasisFunctions = mSurfaceCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal)
        {
            auto tElementOrdinal = tElementOrds(aSideOrdinal);
            auto tElemFaceOrdinal = tFaceOrds(aSideOrdinal);

            Plato::OrdinalType tLocalNodeOrds[mNumNodesPerFace];
            for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<mNumNodesPerFace; tNodeOrd++)
            {
                tLocalNodeOrds[tNodeOrd] = tNodeOrds(aSideOrdinal*mNumNodesPerFace+tNodeOrd);
            }

            // calculate surface jacobians
            ConfigT tSurfaceAreaTimesCubWeight(0.0);
            tCalculateSurfaceJacobians(tElementOrdinal, aSideOrdinal, tLocalNodeOrds, tConfigWS, tJacobians);
            tCalculateSurfaceArea(aSideOrdinal, tSurfaceCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

            // compute unit normal vector
            auto tUnitNormalVec = Plato::omega_h::unit_normal_vector(tElementOrdinal, tElemFaceOrdinal, tCoords);

            // project velocity field onto surface
            Plato::Fluids::project_vector_field_onto_surface<mNumSpatialDims,mNumNodesPerFace>
               (tElementOrdinal, tSurfaceBasisFunctions, tLocalNodeOrds, tPrevVelWS, tPrevVelGP);

            auto tMultiplier = aMultiplier / tCriticalTimeStep(0);
            for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++ )
            {
                auto tLocalCellNode = tLocalNodeOrds[tNode];
                for( Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++ )
                {
                    aResult(tElementOrdinal, tLocalCellNode) += tMultiplier * tUnitNormalVec(tDim) *
                        tPrevVelGP(tElementOrdinal, tDim) * tSurfaceBasisFunctions(tNode) * tSurfaceAreaTimesCubWeight;
                }
            }
        }, "calculate surface momentum integral");
    }
};
// class MomentumSurfaceForces

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::MomentumSurfaceForces, Plato::MassConservation, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::MomentumSurfaceForces, Plato::MassConservation, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::MomentumSurfaceForces, Plato::MassConservation, Plato::SimplexFluids, 3, 1)
#endif
