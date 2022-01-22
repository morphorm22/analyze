/*
 * SurfacePressureIntegral.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include "UtilsOmegaH.hpp"
#include "SpatialModel.hpp"
#include "ImplicitFunctors.hpp"
#include "SurfaceIntegralUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Class for the evaluation of natural boundary condition surface integrals
 * of type: UNIFORM PRESSURE.
 *
 * \tparam SpatialDim   spatial dimension
 * \tparam NumDofs      number degrees of freedom per natural boundary condition force vector
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs=SpatialDim, Plato::OrdinalType DofsPerNode=NumDofs, Plato::OrdinalType DofOffset=0>
class SurfacePressureIntegral
{
private:
    /*!< number of spatial dimensions */
    static constexpr auto mNumSpatialDims = SpatialDim;
    /*!< number of spatial dimensions on face */
    static constexpr auto mNumSpatialDimsOnFace = mNumSpatialDims - static_cast<Plato::OrdinalType>(1);

    const std::string mSideSetName; /*!< side set name */
    const Plato::Array<NumDofs> mFlux; /*!< force vector values */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDimsOnFace> mCubatureRule; /*!< integration rule */

public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    SurfacePressureIntegral(const std::string & aSideSetName, const Plato::Array<NumDofs>& aFlux);

    /***************************************************************************//**
     * \brief Evaluate natural boundary condition surface integrals.
     *
     * \tparam StateScalarType   state forward automatically differentiated (FAD) type
     * \tparam ControlScalarType control FAD type
     * \tparam ConfigScalarType  configuration FAD type
     * \tparam ResultScalarType  result FAD type
     *
     * \param [in]  aSpatialModel Plato Analyze spatial model.
     * \param [in]  aState        2-D view of state variables.
     * \param [in]  aControl      2-D view of control variables.
     * \param [in]  aConfig       3-D view of configuration variables.
     * \param [out] aResult       Assembled vector to which the boundary terms will be added
     * \param [in]  aScale        scalar multiplier
     *
    *******************************************************************************/
    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void operator()(
        const Plato::SpatialModel                          & aSpatialModel,
        const Plato::ScalarMultiVectorT<  StateScalarType> & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType> & aConfig,
        const Plato::ScalarMultiVectorT< ResultScalarType> & aResult,
              Plato::Scalar aScale) const;
};
// class SurfacePressureIntegral

/***************************************************************************//**
 * \brief SurfacePressureIntegral::SurfacePressureIntegral constructor definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
SurfacePressureIntegral<SpatialDim,NumDofs,DofsPerNode,DofOffset>::SurfacePressureIntegral
(const std::string & aSideSetName, const Plato::Array<NumDofs>& aFlux) :
    mSideSetName(aSideSetName),
    mFlux(aFlux),
    mCubatureRule()
{
}
// class SurfacePressureIntegral::SurfacePressureIntegral

/***************************************************************************//**
 * \brief SurfacePressureIntegral::operator() function definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void SurfacePressureIntegral<SpatialDim,NumDofs,DofsPerNode,DofOffset>::operator()(
    const Plato::SpatialModel                          & aSpatialModel,
    const Plato::ScalarMultiVectorT<  StateScalarType>& aState,
    const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
    const Plato::ScalarArray3DT    < ConfigScalarType>& aConfig,
    const Plato::ScalarMultiVectorT< ResultScalarType>& aResult,
          Plato::Scalar aScale
) const
{
    auto tElementOrds = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tFaceOrds    = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    auto tNodeOrds    = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);

    auto tNumFaces = tElementOrds.size();

    Plato::CalculateSurfaceArea<mNumSpatialDims> tCalculateSurfaceArea;
    Plato::NodeCoordinate<mNumSpatialDims> tCoords(aSpatialModel.Mesh);
    Plato::CalculateSurfaceJacobians<mNumSpatialDims> tCalculateSurfaceJacobians;
    Plato::ScalarArray3DT<ConfigScalarType> tJacobians("jacobian", tNumFaces, mNumSpatialDimsOnFace, mNumSpatialDims);

    auto tFlux = mFlux;
    auto tNumDofs = NumDofs;
    auto tNodesPerFace = mNumSpatialDims;
    auto tCubatureWeight = mCubatureRule.getCubWeight();
    auto tBasisFunctions = mCubatureRule.getBasisFunctions();
    if(std::isfinite(tCubatureWeight) == false)
    {
        ANALYZE_THROWERR("Surface Pressure Integral: A non-finite cubature weight was detected.")
    }
    auto tCubWeightTimesScale = aScale * tCubatureWeight;

    // pressure forces should act towards the surface; thus, -1.0 is used to invert the outward facing normal inwards.
    Plato::Scalar tNormalMultiplier(-1.0);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aSideOrdinal)
    {
        auto tElementOrdinal = tElementOrds(aSideOrdinal);
        auto tElemFaceOrdinal = tFaceOrds(aSideOrdinal);

        Plato::OrdinalType tLocalNodeOrds[mNumSpatialDims];
        for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<tNodesPerFace; tNodeOrd++)
        {
            tLocalNodeOrds[tNodeOrd] = tNodeOrds(aSideOrdinal*tNodesPerFace+tNodeOrd);
        }

        ConfigScalarType tSurfaceAreaTimesCubWeight(0.0);
        tCalculateSurfaceJacobians(tElementOrdinal, aSideOrdinal, tLocalNodeOrds, aConfig, tJacobians);
        tCalculateSurfaceArea(aSideOrdinal, tCubWeightTimesScale, tJacobians, tSurfaceAreaTimesCubWeight);

        // compute unit normal vector
        auto tUnitNormalVec = Plato::omega_h::unit_normal_vector(tElementOrdinal, tElemFaceOrdinal, tCoords);

        // project into aResult workset
        for( Plato::OrdinalType tNode=0; tNode<tNodesPerFace; tNode++)
        {
              for( Plato::OrdinalType tDof=0; tDof<tNumDofs; tDof++)
              {
                auto tElementDofOrdinal = (tLocalNodeOrds[tNode] * DofsPerNode) + tDof + DofOffset;
                aResult(tElementOrdinal,tElementDofOrdinal) += tNormalMultiplier * tBasisFunctions(tNode) *
                    ( tUnitNormalVec(tDof) * tFlux(tDof) ) * tSurfaceAreaTimesCubWeight;
            }
        }
    }, "surface pressure integral");
}
// class SurfacePressureIntegral::operator()

}
// namespace Plato
