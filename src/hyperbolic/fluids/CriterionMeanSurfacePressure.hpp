/*
 * CriterionMeanSurfacePressure.hpp
 *
 *  Created on: Apr 6, 2021
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
 * \class CriterionMeanSurfacePressure
 *
 * \brief Class responsible for the evaluation of the mean surface pressure
 *   along the user-specified entity sets (e.g. side sets).
 *
 *                  \f[ \int_{\Gamma_e} p^n d\Gamma_e \f],
 *
 * where \f$ n \f$ denotes the current time step and \f$ p \f$ denotes pressure.
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class CriterionMeanSurfacePressure : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumSpatialDimsOnFace = PhysicsT::SimplexT::mNumSpatialDimsOnFace;   /*!< number of spatial dimensions on face */
    static constexpr auto mNumNodesPerFace      = PhysicsT::SimplexT::mNumNodesPerFace;        /*!< number of nodes per face */
    static constexpr auto mNumPressDofsPerNode  = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of pressure dofs per node */

    using ResultT   = typename EvaluationT::ResultScalarType;      /*!< result FAD type */
    using ConfigT   = typename EvaluationT::ConfigScalarType;      /*!< configuration FAD type */
    using PressureT = typename EvaluationT::CurrentMassScalarType; /*!< pressure FAD type */

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
    CriterionMeanSurfacePressure
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
    virtual ~CriterionMeanSurfacePressure(){}

    /***************************************************************************//**
     * \fn std::string name
     * \brief Returns scalar function name
     * \return scalar function name
     ******************************************************************************/
    std::string name() const override { return mFuncName; }

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate scalar function inside the computational domain \f$ \Omega \f$.
     * \param [in] aWorkSets holds state work sets initialize with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
    void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    { return; }

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate scalar function along the computational boudary \f$ \Gamma \f$.
     * \param [in] aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in] aWorkSets holds state work sets initialize with correct FAD types
     * \param [in] aResult   1D output work set of size number of cells
     ******************************************************************************/
    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel, 
     const Plato::WorkSets & aWorkSets, 
     Plato::ScalarVectorT<ResultT> & aResult) 
    const override
    {
        auto tNumCells = mSpatialDomain.Mesh->NumElements();
        if(tNumCells != aResult.extent(0)) 
        {
            ANALYZE_THROWERR( std::string("Dimension mismatch. 'Result View' and 'Spatial Domain' cell/element number do not match. ") 
                + "'Result View' has '" + std::to_string(aResult.extent(0)) + "' elements and 'Spatial Domain' has '" 
                + std::to_string(tNumCells) + "' elements." )
        }

        // allocate local functors
        Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
        Plato::NodeCoordinate<mNumSpatialDims> tCoords(mSpatialDomain.Mesh);
        Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;

        // set input worksets
        auto tConfigWS = Plato::unpack<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurrentPressWS = Plato::unpack<Plato::ScalarMultiVectorT<PressureT>>(aWorkSets.get("current pressure"));

        // transfer member data to device
        auto tCubatureWeight = mSurfaceCubatureRule.getCubWeight();
        auto tBasisFunctions = mSurfaceCubatureRule.getBasisFunctions();
        for(auto& tName : mSideSets)
        {
            auto tElementOrds = aSpatialModel.Mesh->GetSideSetElements(tName);
            auto tFaceOrds    = aSpatialModel.Mesh->GetSideSetFaces(tName);
            auto tNodeOrds    = aSpatialModel.Mesh->GetSideSetLocalNodes(tName);

            auto tNumFaces = tFaceOrds.size();

            Plato::ScalarArray3DT<ConfigT> tJacobians("face Jacobians", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

            // set local worksets
            Plato::ScalarVectorT<ResultT> tResult("temp results", tNumCells);
            Plato::ScalarVectorT<ConfigT> tSurfaceAreaSum("surface area sum", 1);
            Plato::ScalarVectorT<ConfigT> tSurfaceArea("surface area", tNumFaces);
            Plato::ScalarVectorT<PressureT> tCurrentPressGP("current pressure at Gauss point", tNumCells);

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumFaces), KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal)
            {
                auto tElementOrdinal = tElementOrds(aSideOrdinal);

                Plato::OrdinalType tLocalNodeOrdinals[mNumSpatialDims];
                for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<mNumNodesPerFace; tNodeOrd++)
                {
                    tLocalNodeOrdinals[tNodeOrd] = tNodeOrds(aSideOrdinal*mNumNodesPerFace+tNodeOrd);
                }

                // calculate surface Jacobian and surface integral weight
                tCalculateSurfaceJacobians(tElementOrdinal, aSideOrdinal, tLocalNodeOrdinals, tConfigWS, tJacobians);
                tCalculateSurfaceArea(aSideOrdinal, tCubatureWeight, tJacobians, tSurfaceArea);

                Kokkos::atomic_add(&tSurfaceAreaSum(0), tSurfaceArea(aSideOrdinal));

                // project current pressure onto surface
                for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerFace; tNode++)
                {
                    auto tLocalElementNode = tLocalNodeOrdinals[tNode];
                    tCurrentPressGP(tElementOrdinal) += tBasisFunctions(tNode) * tCurrentPressWS(tElementOrdinal, tLocalElementNode);
                }

                // calculate surface integral, which is defined as \int_{\Gamma_e}N_p^a p^h d\Gamma_e
                for( Plato::OrdinalType tNode=0; tNode < mNumNodesPerFace; tNode++)
                {
                    tResult(tElementOrdinal) += tBasisFunctions(tNode) * tCurrentPressGP(tElementOrdinal) * tSurfaceArea(aSideOrdinal);
                }
            }, "integrate surface pressure");

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aElementOrdinal)
            {
                aResult(aElementOrdinal) = ( static_cast<Plato::Scalar>(1.0) / tSurfaceAreaSum(0) ) * tResult(aElementOrdinal);
            }, "calculate mean surface pressure");
        }
    }
};
// class CriterionMeanSurfacePressure

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionMeanSurfacePressure, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionMeanSurfacePressure, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::CriterionMeanSurfacePressure, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
#endif
