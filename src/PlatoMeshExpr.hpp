#pragma once

#include <Omega_h_expr.hpp>

#include "ExpressionEvaluator.hpp"
#include "SpatialModel.hpp"
#include "UtilsOmegaH.hpp"
#include "ImplicitFunctors.hpp"
#include "Plato_TopOptFunctors.hpp"

namespace Plato
{

/******************************************************************************/
template<Plato::OrdinalType SpaceDim, typename ScalarType>
void
getFunctionValues(
          Plato::ScalarArray3DT<ScalarType>        aPoints,
    const std::string                            & aFuncString,
          Plato::ScalarMultiVectorT<ScalarType>  & aFxnValues
)
/******************************************************************************/
{
    Plato::OrdinalType tNumCells = aPoints.extent(0);
    Plato::OrdinalType tNumPoints = aPoints.extent(1);

    Plato::ScalarVectorT<ScalarType> x_coords("x coordinates", tNumCells*tNumPoints);
    Plato::ScalarVectorT<ScalarType> y_coords("y coordinates", tNumCells*tNumPoints);
    Plato::ScalarVectorT<ScalarType> z_coords("z coordinates", tNumCells*tNumPoints);

    Kokkos::parallel_for("fill coords", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
        Plato::OrdinalType tEntryOffset = iCellOrdinal * tNumPoints;
        x_coords(tEntryOffset+iGpOrdinal) = aPoints(iCellOrdinal, iGpOrdinal, 0);
        if (SpaceDim > 1) y_coords(tEntryOffset+iGpOrdinal) = aPoints(iCellOrdinal, iGpOrdinal, 1);
        if (SpaceDim > 2) z_coords(tEntryOffset+iGpOrdinal) = aPoints(iCellOrdinal, iGpOrdinal, 2);
    });

    ExpressionEvaluator<Plato::ScalarMultiVectorT<ScalarType>,
                        Plato::ScalarMultiVectorT<ScalarType>,
                        Plato::ScalarVectorT<ScalarType>,
                        Plato::Scalar> tExpEval;

    tExpEval.parse_expression(aFuncString.c_str());
    tExpEval.setup_storage(tNumCells*tNumPoints, /*num vals to eval =*/ 1);
    Kokkos::parallel_for("", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
        Plato::OrdinalType tEntryOrdinal = iCellOrdinal * tNumPoints + iGpOrdinal;
        tExpEval.set_variable("x", x_coords, tEntryOrdinal);
        tExpEval.set_variable("y", y_coords, tEntryOrdinal);
        tExpEval.set_variable("z", z_coords, tEntryOrdinal);
        tExpEval.evaluate_expression( tEntryOrdinal, aFxnValues );
    });
    Kokkos::fence();
    tExpEval.clear_storage();

}

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
void
getFunctionValues(
          Plato::ScalarArray3D   aPoints,
    const std::string          & aFuncString,
          Omega_h::Reals       & aFxnValues
)
/******************************************************************************/
{
    Plato::OrdinalType numCells = aPoints.extent(0);
    Plato::OrdinalType numPoints = aPoints.extent(1);

    auto x_coords = Plato::omega_h::create_omega_h_write_array<Plato::Scalar>("forcing function x coords", numCells * numPoints);
    auto y_coords = Plato::omega_h::create_omega_h_write_array<Plato::Scalar>("forcing function y coords", numCells * numPoints);
    auto z_coords = Plato::omega_h::create_omega_h_write_array<Plato::Scalar>("forcing function z coords", numCells * numPoints);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, numCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
        Plato::OrdinalType entryOffset = aCellOrdinal * numPoints;
        for (Plato::OrdinalType ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
        {
            if (SpaceDim > 0) x_coords[entryOffset+ptOrdinal] = aPoints(aCellOrdinal,ptOrdinal,0);
            if (SpaceDim > 1) y_coords[entryOffset+ptOrdinal] = aPoints(aCellOrdinal,ptOrdinal,1);
            if (SpaceDim > 2) z_coords[entryOffset+ptOrdinal] = aPoints(aCellOrdinal,ptOrdinal,2);
        }
    }, "fill coords");

    Omega_h::ExprReader reader(numCells * numPoints, SpaceDim);
    if(SpaceDim > 0)
        reader.register_variable("x", Omega_h::any(Omega_h::Reals(x_coords)));
    if(SpaceDim > 1)
        reader.register_variable("y", Omega_h::any(Omega_h::Reals(y_coords)));
    if(SpaceDim > 2)
        reader.register_variable("z", Omega_h::any(Omega_h::Reals(z_coords)));

    auto result = reader.read_string(aFuncString, "Integrand");
    reader.repeat(result);
    aFxnValues = Omega_h::any_cast<Omega_h::Reals>(result);
}

/******************************************************************************/
template<typename ElementType, typename ConfigScalarType>
void
mapPoints(
    const Plato::OrdinalVector                    & aCellOrdinals,
    const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
          Plato::ScalarArray3DT<ConfigScalarType>   aMappedPoints
)
/******************************************************************************/
{
    Plato::OrdinalType tNumCells = aCellOrdinals.size();

    auto tCubPoints  = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints  = tCubWeights.size();

    Kokkos::deep_copy(aMappedPoints, Plato::Scalar(0.0)); // initialize to 0

    Kokkos::parallel_for("map points", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
        auto aCellGlobalOrdinal = aCellOrdinals[iCellOrdinal];

        auto tCubPoint = tCubPoints(iGpOrdinal);
        auto tBasisValues = ElementType::basisValues(tCubPoint);

        for (Plato::OrdinalType iNode=0; iNode<ElementType::mNumNodesPerCell; iNode++)
        {
            for (Plato::OrdinalType iDim=0; iDim<ElementType::mNumSpatialDims; iDim++)
            {
                aMappedPoints(aCellGlobalOrdinal, iGpOrdinal, iDim) += tBasisValues(iNode)*aConfig(aCellGlobalOrdinal, iNode, iDim);
            }
        }
    });
}
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
void
mapPoints(
    const Plato::SpatialDomain     & aSpatialDomain,
          Plato::ScalarMultiVector   aRefPoints,
          Plato::ScalarArray3D       aMappedPoints
)
/******************************************************************************/
{
    Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
    Plato::OrdinalType tNumPoints = aMappedPoints.extent(1);

    Kokkos::deep_copy(aMappedPoints, Plato::Scalar(0.0)); // initialize to 0

    Plato::NodeCoordinate<SpaceDim> tNodeCoordinate(aSpatialDomain.Mesh);

    auto tCellOrdinals = aSpatialDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for (Plato::OrdinalType ptOrdinal=0; ptOrdinal<tNumPoints; ptOrdinal++)
        {
            Plato::OrdinalType tNodeOrdinal;
            Scalar tFinalNodeValue = 1.0;
            for (tNodeOrdinal=0; tNodeOrdinal<SpaceDim; tNodeOrdinal++)
            {
                Scalar tNodeValue = aRefPoints(ptOrdinal,tNodeOrdinal);
                tFinalNodeValue -= tNodeValue;
                for (Plato::OrdinalType d=0; d<SpaceDim; d++)
                {
                    aMappedPoints(aCellOrdinal, ptOrdinal, d) += tNodeValue * tNodeCoordinate(tCellOrdinal, tNodeOrdinal, d);
                }
            }
            tNodeOrdinal = SpaceDim;
            for (Plato::OrdinalType d=0; d<SpaceDim; d++)
            {
                aMappedPoints(aCellOrdinal, ptOrdinal, d) += tFinalNodeValue * tNodeCoordinate(tCellOrdinal, tNodeOrdinal, d);
            }
        }
    });
}

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
void
mapPoints(
    const Plato::SpatialModel      & aSpatialModel,
          Plato::ScalarMultiVector   aRefPoints,
          Plato::ScalarArray3D       aMappedPoints
)
/******************************************************************************/
{
    Plato::OrdinalType tNumCells = aSpatialModel.Mesh->NumElements();
    Plato::OrdinalType tNumPoints = aMappedPoints.extent(1);

    Kokkos::deep_copy(aMappedPoints, Plato::Scalar(0.0)); // initialize to 0

    Plato::NodeCoordinate<SpaceDim> tNodeCoordinate(&(aSpatialModel.Mesh));

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
        for (Plato::OrdinalType ptOrdinal=0; ptOrdinal<tNumPoints; ptOrdinal++)
        {
            Plato::OrdinalType tNodeOrdinal;
            Scalar tFinalNodeValue = 1.0;
            for (tNodeOrdinal=0; tNodeOrdinal<SpaceDim; tNodeOrdinal++)
            {
                Scalar tNodeValue = aRefPoints(ptOrdinal,tNodeOrdinal);
                tFinalNodeValue -= tNodeValue;
                for (Plato::OrdinalType d=0; d<SpaceDim; d++)
                {
                    aMappedPoints(aCellOrdinal, ptOrdinal, d) += tNodeValue * tNodeCoordinate(aCellOrdinal, tNodeOrdinal, d);
                }
            }
            tNodeOrdinal = SpaceDim;
            for (Plato::OrdinalType d=0; d<SpaceDim; d++)
            {
                aMappedPoints(aCellOrdinal, ptOrdinal, d) += tFinalNodeValue * tNodeCoordinate(aCellOrdinal, tNodeOrdinal, d);
            }
        }
    });
}

}
