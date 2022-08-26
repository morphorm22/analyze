#pragma once

#include "ExpressionEvaluator.hpp"
#include "SpatialModel.hpp"
#include "ImplicitFunctors.hpp"
#include "Plato_TopOptFunctors.hpp"

namespace Plato
{

/******************************************************************************/
template<Plato::OrdinalType SpaceDim, typename ScalarType>
void
getFunctionValues(
    const Plato::ScalarArray3DT<ScalarType>      & aPoints,
    const std::string                            & aFuncString,
    const Plato::ScalarMultiVectorT<ScalarType>  & aFxnValues
)
/******************************************************************************/
{
    Plato::OrdinalType tNumCells = aPoints.extent(0);
    Plato::OrdinalType tNumPoints = aPoints.extent(1);

    Plato::ScalarVectorT<ScalarType> x_coords("x coordinates", tNumCells*tNumPoints);
    Plato::ScalarVectorT<ScalarType> y_coords("y coordinates", tNumCells*tNumPoints);
    Plato::ScalarVectorT<ScalarType> z_coords("z coordinates", tNumCells*tNumPoints);

    Kokkos::parallel_for("fill coords", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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

    tExpEval.set_variable("x", x_coords);
    tExpEval.set_variable("y", y_coords);
    tExpEval.set_variable("z", z_coords);

    auto tNumTotalPoints = tNumCells*tNumPoints;
    Kokkos::parallel_for("", Kokkos::RangePolicy<>(0, tNumTotalPoints), KOKKOS_LAMBDA(const Plato::OrdinalType iEntryOrdinal)
    {
        tExpEval.evaluate_expression( iEntryOrdinal, aFxnValues );
    });
    Kokkos::fence();
    tExpEval.clear_storage();

}

/******************************************************************************/
template<typename ElementType, typename ConfigScalarType>
void
mapPoints(
    const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
          Plato::ScalarArray3DT<ConfigScalarType>   aMappedPoints
)
/******************************************************************************/
{
    Plato::OrdinalType tNumCells = aConfig.extent(0);

    auto tCubPoints  = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints  = tCubWeights.size();

    Kokkos::deep_copy(aMappedPoints, Plato::Scalar(0.0)); // initialize to 0

    Kokkos::parallel_for("map points", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
        auto tCubPoint = tCubPoints(iGpOrdinal);
        auto tBasisValues = ElementType::basisValues(tCubPoint);

        for (Plato::OrdinalType iNode=0; iNode<ElementType::mNumNodesPerCell; iNode++)
        {
            for (Plato::OrdinalType iDim=0; iDim<ElementType::mNumSpatialDims; iDim++)
            {
                aMappedPoints(iCellOrdinal, iGpOrdinal, iDim) += tBasisValues(iNode)*aConfig(iCellOrdinal, iNode, iDim);
            }
        }
    });
}
/******************************************************************************/
template<typename ElementType>
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

    Plato::NodeCoordinate<ElementType::mNumSpatialDims, ElementType::mNumNodesPerCell> tNodeCoordinate(aSpatialDomain.Mesh);

    auto tCellOrdinals = aSpatialDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(Plato::OrdinalType aCellOrdinal)
    {
        auto tCellOrdinal = tCellOrdinals[aCellOrdinal];
        for (Plato::OrdinalType ptOrdinal=0; ptOrdinal<tNumPoints; ptOrdinal++)
        {
            Plato::OrdinalType tNodeOrdinal;
            Scalar tFinalNodeValue = 1.0;
            for (tNodeOrdinal=0; tNodeOrdinal<ElementType::mNumSpatialDims; tNodeOrdinal++)
            {
                Scalar tNodeValue = aRefPoints(ptOrdinal,tNodeOrdinal);
                tFinalNodeValue -= tNodeValue;
                for (Plato::OrdinalType d=0; d<ElementType::mNumSpatialDims; d++)
                {
                    aMappedPoints(aCellOrdinal, ptOrdinal, d) += tNodeValue * tNodeCoordinate(tCellOrdinal, tNodeOrdinal, d);
                }
            }
            tNodeOrdinal = ElementType::mNumSpatialDims;
            for (Plato::OrdinalType d=0; d<ElementType::mNumSpatialDims; d++)
            {
                aMappedPoints(aCellOrdinal, ptOrdinal, d) += tFinalNodeValue * tNodeCoordinate(tCellOrdinal, tNodeOrdinal, d);
            }
        }
    });
}

/******************************************************************************/
template<typename ElementType>
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

    Plato::NodeCoordinate<ElementType::mNumSpatialDims, ElementType::mNumNodesPerCell> tNodeCoordinate(&(aSpatialModel.Mesh));

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(Plato::OrdinalType aCellOrdinal)
    {
        for (Plato::OrdinalType ptOrdinal=0; ptOrdinal<tNumPoints; ptOrdinal++)
        {
            Plato::OrdinalType tNodeOrdinal;
            Scalar tFinalNodeValue = 1.0;
            for (tNodeOrdinal=0; tNodeOrdinal<ElementType::mNumSpatialDims; tNodeOrdinal++)
            {
                Scalar tNodeValue = aRefPoints(ptOrdinal,tNodeOrdinal);
                tFinalNodeValue -= tNodeValue;
                for (Plato::OrdinalType d=0; d<ElementType::mNumSpatialDims; d++)
                {
                    aMappedPoints(aCellOrdinal, ptOrdinal, d) += tNodeValue * tNodeCoordinate(aCellOrdinal, tNodeOrdinal, d);
                }
            }
            tNodeOrdinal = ElementType::mNumSpatialDims;
            for (Plato::OrdinalType d=0; d<ElementType::mNumSpatialDims; d++)
            {
                aMappedPoints(aCellOrdinal, ptOrdinal, d) += tFinalNodeValue * tNodeCoordinate(aCellOrdinal, tNodeOrdinal, d);
            }
        }
    });
}

/******************************************************************************//**
 * \brief compute function values at gauss points
**********************************************************************************/
template<typename ConfigScalarType, typename ElementType>
Plato::ScalarMultiVectorT<ConfigScalarType>
computeSpatialWeights(
    const Plato::SpatialDomain                    & aSpatialDomain,
    const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
    const std::string                             & aFunction
)
{
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints  = tCubWeights.size();

    auto tNumCells = aSpatialDomain.numCells();

    Plato::ScalarArray3DT<ConfigScalarType> tPhysicalPoints("physical points", tNumCells, tNumPoints, ElementType::mNumSpatialDims);
    Plato::mapPoints<ElementType>(aConfig, tPhysicalPoints);

    Plato::ScalarMultiVectorT<ConfigScalarType> tFxnValues("function values", tNumCells*tNumPoints, 1);
    Plato::getFunctionValues<ElementType::mNumSpatialDims>(tPhysicalPoints, aFunction, tFxnValues);

    return tFxnValues;
}


}
