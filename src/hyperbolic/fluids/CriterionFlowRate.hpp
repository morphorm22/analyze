/*
 * CriterionFlowRate.hpp
 *
 *  Created on: Apr 28, 2021
 */

#pragma once

#include "UtilsOmegaH.hpp"
#include "SpatialModel.hpp"
#include "UtilsTeuchos.hpp"
#include "ExpInstMacros.hpp"
#include "SurfaceIntegralUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/fluids/SimplexFluids.hpp"
#include "hyperbolic/fluids/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/fluids/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \tparam PhysicsT    Plato physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \class CriterionFlowRate
 *
 * \brief Evaluatie volumetric flow rate along specidied side sets (e.g. entity 
 *        set). The volumetric flow rate is defined as
 *
 *                  \f[ \int_{\Gamma_e} u_i^n n_i d\Gamma_e \f],
 *
 * where \f$ n \f$ denotes the current time step, \f$ u_i \f$ dis the i-th 
 * velocity component and \f$ n_i \f$ is the unit normal.
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class CriterionFlowRate : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace;   /*!< number of spatial dimensions on face */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;        /*!< number of nodes per face */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of pressure dofs per node */

    using ResultT = typename EvaluationT::ResultScalarType; /*!< result FAD type */
    using ConfigT = typename EvaluationT::ConfigScalarType; /*!< configuration FAD type */
    using CurVelT = typename EvaluationT::CurrentMomentumScalarType; /*!< current velocity FAD type */

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace>; /*!< local short name for cubature rule class */

    // member metadata
    Plato::DataMap& mDataMap; /*!< holds output metadata */
    CubatureRule mSurfaceCubatureRule; /*!< cubature integration rule on surface */
    const Plato::SpatialDomain& mSpatialDomain; /*!< holds mesh and entity sets metadata for a domain (i.e. element block) */

    // member parameters
    std::string mFuncName; /*!< scalar funciton name */
    std::vector<std::string> mSideSets; /*!< sideset names corresponding to the surfaces associated with the surface integral */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aName    scalar function name
     * \param [in] aDomain  holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in] aDataMap holds output metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    CriterionFlowRate
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSurfaceCubatureRule(CubatureRule()),
         mSpatialDomain(aDomain),
         mFuncName(aName)
    {
        auto tMyCriteria = aInputs.sublist("Criteria").sublist(aName);
        mSideSets = Plato::teuchos::parse_array<std::string>("Sides", tMyCriteria);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~CriterionFlowRate(){}

    /***************************************************************************//**
     * \fn std::string name
     * \brief Returns scalar function name
     * \return scalar function name
     ******************************************************************************/
    std::string name() const override { return mFuncName; }

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate scalar function inside the computational domain \f$ \Omega \f$.
     * \param [in] aWorkSets holds state work sets initialized with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
    void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    { return; }

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate scalar function along a set of user-defined boudaries \f$ d\Gamma \f$.
     * \param [in] aSpatialModel mesh and entity sets (e.g. node and side sets) metadata
     * \param [in] aWorkSets     state work sets initialized with correct FAD types
     * \param [in] aResult       1D output work set of size number of cells
     ******************************************************************************/
    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel, 
     const Plato::WorkSets & aWorkSets, 
     Plato::ScalarVectorT<ResultT> & aResult) 
    const override
    {
        // allocate local functors
        Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(mSpatialDomain.Mesh);
        Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;

        // set input worksets
        auto tConfigWS = Plato::unpack<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurrentVelocityWS = Plato::unpack<Plato::ScalarMultiVectorT<CurVelT>>(aWorkSets.get("current velocity"));

        // transfer member data to device
        auto tCubatureWeight = mSurfaceCubatureRule.getCubWeight();
        auto tBasisFunctions = mSurfaceCubatureRule.getBasisFunctions();
        for(auto& tName : mSideSets)
        {
            auto tElementOrds = aSpatialModel.Mesh->GetSideSetElements(tName);
            auto tFaceOrds    = aSpatialModel.Mesh->GetSideSetFaces(tName);
            auto tNodeOrds    = aSpatialModel.Mesh->GetSideSetLocalNodes(tName);

            auto tNumFaces = tFaceOrds.size();
            auto tNumElements = mSpatialDomain.Mesh->NumElements();

            // set local worksets
            Plato::ScalarArray3DT<ConfigT> tJacobians("face Jacobians", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);
            Plato::ScalarMultiVectorT<CurVelT> tCurVelGP("current velocity at Gauss points", tNumElements, mNumSpatialDims);

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal)
            {
                auto tElementOrdinal = tElementOrds(aSideOrdinal);
                auto tElemFaceOrdinal = tFaceOrds(aSideOrdinal);

                Plato::OrdinalType tLocalNodeOrdinals[mNumSpatialDims];
                for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<mNumNodesPerFace; tNodeOrd++)
                {
                    tLocalNodeOrdinals[tNodeOrd] = tNodeOrds(aSideOrdinal*mNumNodesPerFace+tNodeOrd);
                }

                // calculate surface Jacobian and surface integral weight
                ConfigT tSurfaceAreaTimesCubWeight(0.0);
                tCalculateSurfaceJacobians(tElementOrdinal, aSideOrdinal, tLocalNodeOrdinals, tConfigWS, tJacobians);
                tCalculateSurfaceArea(aSideOrdinal, tCubatureWeight, tJacobians, tSurfaceAreaTimesCubWeight);

                // compute unit normal vector
                auto tUnitNormalVec = Plato::omega_h::unit_normal_vector(tElementOrdinal, tElemFaceOrdinal, tCoords);

                // project current velocity onto surface
                for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++)
                {
                    for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                    {
                        auto tLocalElementNode = tLocalNodeOrdinals[tNode];
                        auto tLocalElementDof = (tLocalElementNode * mNumSpatialDims) + tDim;
                        tCurVelGP(tElementOrdinal, tDim) += tBasisFunctions(tNode) * tCurrentVelocityWS(tElementOrdinal, tLocalElementDof);
                    }
                }

                // calculate flow rate, which is defined as \int_{\Gamma_e}N_p^a (u_i^n n_i) d\Gamma_e
                for( Plato::OrdinalType tNode=0; tNode < mNumNodesPerFace; tNode++)
                {
                    for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
                    {
                        aResult(tElementOrdinal) += tBasisFunctions(tNode) * tCurVelGP(tElementOrdinal, tDim) * tUnitNormalVec(tDim) * tSurfaceAreaTimesCubWeight;
                    }
                }
            }, "flow rate");
        }
    }
};
// class CriterionFlowRate

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionFlowRate, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionFlowRate, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionFlowRate, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
#endif
