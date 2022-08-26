#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "Tet4.hpp"
#include "ToMap.hpp"
#include "BLAS1.hpp"
#include "SpatialModel.hpp"
#include "PlatoSequence.hpp"
#include "PlatoMathHelpers.hpp"
#include "elliptic/hatching/Problem.hpp"
#include "elliptic/hatching/Mechanics.hpp"
#include "elliptic/hatching/PhysicsScalarFunction.hpp"

namespace HatchingTestUtils
{

Plato::ScalarArray3D
RandomStep(Plato::Scalar aLowerBound, Plato::Scalar aUpperBound, Plato::ScalarArray3D aPattern)
{
    auto tSize0 = aPattern.extent(0);
    auto tSize1 = aPattern.extent(1);
    auto tSize2 = aPattern.extent(2);
    Plato::ScalarArray3D tStep = Plato::ScalarArray3D("Step", tSize0, tSize1, tSize2);
    auto tHostStep = Kokkos::create_mirror(tStep);

    unsigned int tRANDOM_SEED = 1;
    std::srand(tRANDOM_SEED);
    for(decltype(tSize0) iDim0=0; iDim0<tSize0; iDim0++)
    {
      for(decltype(tSize1) iDim1=0; iDim1<tSize1; iDim1++)
      {
        for(decltype(tSize2) iDim2=0; iDim2<tSize2; iDim2++)
        {
          const Plato::Scalar tRandNum = static_cast<Plato::Scalar>(std::rand()) / static_cast<Plato::Scalar>(RAND_MAX);
          tHostStep(iDim0, iDim1, iDim2) = aLowerBound + ( (aUpperBound - aLowerBound) * tRandNum);
        }
      }
    }
    Kokkos::deep_copy(tStep, tHostStep);
    return tStep;
}

void axpy(const Plato::Scalar & aAlpha, Plato::ScalarArray3D aInput, Plato::ScalarArray3D aOutput)
{
    auto tSize0 = aInput.extent(0);
    auto tSize1 = aInput.extent(1);
    auto tSize2 = aInput.extent(2);
    Kokkos::parallel_for("axpy", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {tSize0, tSize1, tSize2}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iDim0, const Plato::OrdinalType iDim1, const Plato::OrdinalType iDim2)
    {
        aOutput(iDim0, iDim1, iDim2) += aAlpha * aInput(iDim0, iDim1, iDim2);
    });
}

void
Flatten(Plato::ScalarArray3D aArray3D, Plato::ScalarVector & aVector)
{
    auto tSize0 = aArray3D.extent(0);
    auto tSize1 = aArray3D.extent(1);
    auto tSize2 = aArray3D.extent(2);
    auto tSize = tSize0*tSize1*tSize2;
    Kokkos::resize(aVector, tSize);
    Kokkos::parallel_for("flatten", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {tSize0, tSize1, tSize2}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iDim0, const Plato::OrdinalType iDim1, const Plato::OrdinalType iDim2)
    {
        aVector(iDim0*tSize1*tSize2+iDim1*tSize2+iDim2) = aArray3D(iDim0, iDim1, iDim2);
    });
}

void
Unflatten(Plato::ScalarArray3D aArray3D, Plato::ScalarVector & aVector)
{
    auto tSize0 = aArray3D.extent(0);
    auto tSize1 = aArray3D.extent(1);
    auto tSize2 = aArray3D.extent(2);
    auto tSize = tSize0*tSize1*tSize2;
    Kokkos::resize(aVector, tSize);
    Kokkos::parallel_for("flatten", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {tSize0, tSize1, tSize2}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iDim0, const Plato::OrdinalType iDim1, const Plato::OrdinalType iDim2)
    {
        aArray3D(iDim0, iDim1, iDim2) = aVector(iDim0*tSize1*tSize2+iDim1*tSize2+iDim2);
    });
}

template <class ElementType, class VectorFunctionT, class SolutionT, class ControlT>
Plato::Scalar testVectorFunction_Partial_z(VectorFunctionT& aVectorFunction, SolutionT aSolution, ControlT aControl, int aStepIndex)
{
    auto tState = aSolution.get("State");
    Plato::ScalarVector tGlobalState = Kokkos::subview(tState, aStepIndex, Kokkos::ALL());
    Plato::ScalarArray3D tPrevLocalState;
    if (aStepIndex > 0)
    {
        Plato::ScalarArray4D tLocalState;
        aSolution.get("Local State", tLocalState);
        tPrevLocalState = Kokkos::subview(tLocalState, aStepIndex-1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
    }
    else
    {
        // kokkos initializes new views to zero.
        tPrevLocalState = Plato::ScalarArray3D("initial local state", aVectorFunction.numCells(),
                                              ElementType::mNumGaussPoints, ElementType::mNumLocalStatesPerGP);
    }

    // compute initial R and dRdz
    auto tResidual = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    auto t_dRdz = aVectorFunction.gradient_z(tGlobalState, tPrevLocalState, aControl);

    Plato::ScalarVector tStep = Plato::ScalarVector("Step", aControl.extent(0));
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.00025, 0.0005, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);

    // compute F at z - step
    Plato::blas1::axpy(-1.0, tStep, aControl);
    auto tResidualNeg = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);

    // compute F at z + step
    Plato::blas1::axpy(2.0, tStep, aControl);
    auto tResidualPos = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    Plato::blas1::axpy(-1.0, tStep, aControl);

    // compute actual change in F over 2 * deltaZ
    Plato::blas1::axpy(-1.0, tResidualPos, tResidualNeg);
    auto tDeltaFD = Plato::blas1::norm(tResidualNeg);

    Plato::ScalarVector tDeltaR = Plato::ScalarVector("delta R", tResidual.extent(0));
    Plato::blas1::scale(2.0, tStep);
    Plato::VectorTimesMatrixPlusVector(tStep, t_dRdz, tDeltaR);
    auto tDeltaAD = Plato::blas1::norm(tDeltaR);

    // return error
    Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
    return std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
}

template <class ElementType, class ProblemT, class VectorT>
Plato::Scalar testProblem_Total_z(ProblemT& aProblem, VectorT aControl, std::string aCriterionName, Plato::Scalar aAlpha = 1.0e-1)
{
    // compute initial F and dFdz
    auto tSolution = aProblem.solution(aControl);
    auto t_value = aProblem.criterionValue(aControl, aCriterionName);
    auto t_dFdz = aProblem.criterionGradient(aControl, aCriterionName);

    auto tNorm = Plato::blas1::norm(t_dFdz);

    // compute F at z - deltaZ
    Plato::blas1::axpy(-aAlpha/tNorm, t_dFdz, aControl);
    auto tSolutionNeg = aProblem.solution(aControl);
    auto t_valueNeg = aProblem.criterionValue(aControl, aCriterionName);

    // compute F at z + deltaZ
    Plato::blas1::axpy(2.0*aAlpha/tNorm, t_dFdz, aControl);
    auto tSolutionPos = aProblem.solution(aControl);
    auto t_valuePos = aProblem.criterionValue(aControl, aCriterionName);
    Plato::blas1::axpy(-aAlpha/tNorm, t_dFdz, aControl);

    // compute actual change in F over 2 * deltaZ
    auto tDeltaFD = (t_valuePos - t_valueNeg);

    // compute predicted change in F over 2 * deltaZ
    auto tDeltaAD = Plato::blas1::dot(t_dFdz, t_dFdz);
    if (tDeltaAD != 0)
    {
        tDeltaAD *= 2.0*aAlpha/tNorm;
    }
    // return error
    Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
    return std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
}
template <class MeshT, class VectorT>
void perturbMesh(MeshT& aMesh, VectorT aPerturb)
{
    auto tCoords = aMesh->Coordinates();
    auto tNumDims = aMesh->NumDimensions();
    auto tNumDofs = tNumDims*aMesh->NumNodes();
    Plato::ScalarVector tCoordsCopy("coordinates", tNumDofs);
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType &aDofOrdinal)
    {
        tCoordsCopy(aDofOrdinal) = tCoords[aDofOrdinal] + aPerturb(aDofOrdinal);
    }, "tweak mesh");
    aMesh->SetCoordinates(tCoordsCopy);
}
template <class ElementType, class VectorFunctionT, class SolutionT, class ControlT>
Plato::Scalar testVectorFunction_Partial_u(VectorFunctionT& aVectorFunction, SolutionT aSolution, ControlT aControl, int aStepIndex)
{
    auto tState = aSolution.get("State");
    Plato::ScalarVector tGlobalState = Kokkos::subview(tState, aStepIndex, Kokkos::ALL());
    Plato::ScalarArray3D tPrevLocalState;
    if (aStepIndex > 0)
    {
        Plato::ScalarArray4D tLocalState;
        aSolution.get("Local State", tLocalState);
        tPrevLocalState = Kokkos::subview(tLocalState, aStepIndex-1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
    }
    else
    {
        // kokkos initializes new views to zero.
        tPrevLocalState = Plato::ScalarArray3D("initial local state",  aVectorFunction.numCells(),
            ElementType::mNumGaussPoints, ElementType::mNumLocalStatesPerGP);
    }

    // compute initial R and dRdu
    auto tResidual = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    auto t_dRdu = aVectorFunction.gradient_u(tGlobalState, tPrevLocalState, aControl);

    Plato::ScalarVector tStep = Plato::ScalarVector("Step", tGlobalState.extent(0));
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.00025, 0.0005, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);

    // compute F at z - step
    Plato::blas1::axpy(-1.0, tStep, tGlobalState);
    auto tResidualNeg = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);

    // compute F at z + step
    Plato::blas1::axpy(2.0, tStep, tGlobalState);
    auto tResidualPos = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    Plato::blas1::axpy(-1.0, tStep, tGlobalState);

    // compute actual change in F over 2 * deltaZ
    Plato::blas1::axpy(-1.0, tResidualNeg, tResidualPos);
    auto tDeltaFD = Plato::blas1::norm(tResidualPos);

    Plato::ScalarVector tDeltaR = Plato::ScalarVector("delta R", tResidual.extent(0));
    Plato::blas1::scale(2.0, tStep);
    Plato::MatrixTimesVectorPlusVector(t_dRdu, tStep, tDeltaR);
    auto tDeltaAD = Plato::blas1::norm(tDeltaR);

    Plato::blas1::axpy(-1.0, tResidualPos, tDeltaR);
    auto tErrorNorm = Plato::blas1::norm(tDeltaR);

    // return error
    Plato::Scalar tPer = (fabs(tDeltaFD) + fabs(tDeltaAD))/2.0;
    return tErrorNorm / (tPer != 0 ? tPer : 1.0);
}
template <class ElementType, class VectorFunctionT, class SolutionT, class ControlT>
Plato::Scalar testVectorFunction_Partial_u_T(VectorFunctionT& aVectorFunction, SolutionT aSolution, ControlT aControl, int aStepIndex)
{
    auto tState = aSolution.get("State");
    Plato::ScalarVector tGlobalState = Kokkos::subview(tState, aStepIndex, Kokkos::ALL());
    Plato::ScalarArray3D tPrevLocalState;
    if (aStepIndex > 0)
    {
        Plato::ScalarArray4D tLocalState;
        aSolution.get("Local State", tLocalState);
        tPrevLocalState = Kokkos::subview(tLocalState, aStepIndex-1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
    }
    else
    {
        // kokkos initializes new views to zero.
        tPrevLocalState = Plato::ScalarArray3D("initial local state",  aVectorFunction.numCells(),
            ElementType::mNumGaussPoints, ElementType::mNumLocalStatesPerGP);
    }

    // compute initial R and dRduT
    auto tResidual = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    auto t_dRduT = aVectorFunction.gradient_u_T(tGlobalState, tPrevLocalState, aControl);

    Plato::ScalarVector tStep = Plato::ScalarVector("Step", tGlobalState.extent(0));
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.00025, 0.0005, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);

    // compute F at z - step
    Plato::blas1::axpy(-1.0, tStep, tGlobalState);
    auto tResidualNeg = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);

    // compute F at z + step
    Plato::blas1::axpy(2.0, tStep, tGlobalState);
    auto tResidualPos = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    Plato::blas1::axpy(-1.0, tStep, tGlobalState);

    // compute actual change in F over 2 * deltaZ
    Plato::blas1::axpy(-1.0, tResidualNeg, tResidualPos);
    auto tDeltaFD = Plato::blas1::norm(tResidualPos);

    Plato::ScalarVector tDeltaR = Plato::ScalarVector("delta R", tResidual.extent(0));
    Plato::blas1::scale(2.0, tStep);
    Plato::VectorTimesMatrixPlusVector(tStep, t_dRduT, tDeltaR);
    auto tDeltaAD = Plato::blas1::norm(tDeltaR);

    Plato::blas1::axpy(-1.0, tResidualPos, tDeltaR);
    auto tErrorNorm = Plato::blas1::norm(tDeltaR);

    // return error
    Plato::Scalar tPer = (fabs(tDeltaFD) + fabs(tDeltaAD))/2.0;
    return tErrorNorm / (tPer != 0 ? tPer : 1.0);
}
template <class ElementType, class VectorFunctionT, class SolutionT, class ControlT>
Plato::Scalar testVectorFunction_Partial_cp_T(VectorFunctionT& aVectorFunction, SolutionT aSolution, ControlT aControl, int aStepIndex)
{
    auto tState = aSolution.get("State");
    Plato::ScalarVector tGlobalState = Kokkos::subview(tState, aStepIndex, Kokkos::ALL());
    Plato::ScalarArray3D tPrevLocalState;
    if (aStepIndex > 0)
    {
        Plato::ScalarArray4D tLocalStates;
        aSolution.get("Local State", tLocalStates);
        tPrevLocalState = Kokkos::subview(tLocalStates, aStepIndex-1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
    }
    else
    {
        // kokkos initializes new views to zero.
        tPrevLocalState = Plato::ScalarArray3D("initial local state", aVectorFunction.numCells(),
            ElementType::mNumGaussPoints, ElementType::mNumLocalStatesPerGP);
    }

    // compute initial R and dRdcpT
    auto tResidual = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    auto t_dRdcpT = aVectorFunction.gradient_cp_T(tGlobalState, tPrevLocalState, aControl);

    auto tStep = HatchingTestUtils::RandomStep(250000.0, 500000.0, tPrevLocalState);

    // compute F at z - step
    HatchingTestUtils::axpy(-1.0, tStep, tPrevLocalState);
    auto tResidualNeg = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);

    // compute F at z + step
    HatchingTestUtils::axpy(2.0, tStep, tPrevLocalState);
    auto tResidualPos = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    HatchingTestUtils::axpy(-1.0, tStep, tPrevLocalState);

    // compute actual change in F over 2 * deltaZ
    Plato::blas1::axpy(-1.0, tResidualNeg, tResidualPos);
    auto tDeltaFD = Plato::blas1::norm(tResidualPos);

    Plato::ScalarVector tDeltaR = Plato::ScalarVector("delta R", tResidual.extent(0));
    Plato::ScalarVector tFlatStep;
    HatchingTestUtils::Flatten(tStep, tFlatStep);
    Plato::blas1::scale(2.0, tFlatStep);
    Plato::VectorTimesMatrixPlusVector(tFlatStep, t_dRdcpT, tDeltaR);
    auto tDeltaAD = Plato::blas1::norm(tDeltaR);

    Plato::blas1::axpy(-1.0, tResidualPos, tDeltaR);
    auto tErrorNorm = Plato::blas1::norm(tDeltaR);

    // return error
    Plato::Scalar tPer = (fabs(tDeltaFD) + fabs(tDeltaAD))/2.0;
    return tErrorNorm / (tPer != 0 ? tPer : 1.0);
}
template <class ScalarFunctionT, class SolutionT, class ControlT>
Plato::Scalar testScalarFunction_Partial_z(ScalarFunctionT aScalarFunction, SolutionT aSolution, ControlT aControl)
{
    // compute initial F and dFdz
    Plato::ScalarArray4D tLocalState;
    aSolution.get("Local State", tLocalState);
    auto t_value0 = aScalarFunction.value(aSolution, tLocalState, aControl);
    auto t_dFdz = aScalarFunction.gradient_z(aSolution, tLocalState, aControl);

    Plato::Scalar tAlpha = 1.0e-4;
    auto tNorm = Plato::blas1::norm(t_dFdz);

    // compute F at z - deltaZ
    Plato::blas1::axpy(-tAlpha/tNorm, t_dFdz, aControl);
    auto t_valueNeg = aScalarFunction.value(aSolution, tLocalState, aControl);

    // compute F at z + deltaZ
    Plato::blas1::axpy(2.0*tAlpha/tNorm, t_dFdz, aControl);
    auto t_valuePos = aScalarFunction.value(aSolution, tLocalState, aControl);
    Plato::blas1::axpy(-tAlpha/tNorm, t_dFdz, aControl);

    // compute actual change in F over 2 * deltaZ
    auto tDeltaFD = (t_valuePos - t_valueNeg);

    // compute predicted change in F over 2 * deltaZ
    auto tDeltaAD = Plato::blas1::dot(t_dFdz, t_dFdz);
    if (tDeltaAD != 0)
    {
        tDeltaAD *= 2.0*tAlpha/tNorm;
    }
    // return error
    Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
    return std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
}

template <class ScalarFunctionT, class SolutionT, class ControlT>
Plato::Scalar testScalarFunction_Partial_c(ScalarFunctionT aScalarFunction, SolutionT aSolution, ControlT aControl, int aTimeStep)
{
    // compute initial F and dFdc
    Plato::ScalarArray4D tLocalStates;
    aSolution.get("Local State", tLocalStates);
    auto t_value0 = aScalarFunction.value(aSolution, tLocalStates, aControl);
    std::cout << "t_value0:" << t_value0 << std::endl;
    auto t_dFdc = aScalarFunction.gradient_c(aSolution, tLocalStates, aControl, aTimeStep);

    Plato::Scalar tAlpha = 1.0e-3;
    auto tNorm = Plato::blas1::norm(t_dFdc);
    std::cout << "tNorm:" << tNorm << std::endl;

    // compute F at c - deltac
    Plato::ScalarArray3D tLocalState = Kokkos::subview(tLocalStates, aTimeStep, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

    Plato::ScalarVector tStep("step", t_dFdc.extent(0));
    if (tNorm != 0)
    {
      Plato::blas1::axpy(-tAlpha/tNorm, t_dFdc, tStep);
    }
    else
    {
      Kokkos::deep_copy(tStep, tAlpha);
    }
    Plato::ScalarArray3D tStep3D("step 3d", tLocalState.extent(0), tLocalState.extent(1), tLocalState.extent(2));
    HatchingTestUtils::Unflatten(tStep3D, tStep);
    HatchingTestUtils::axpy(-1.0, tStep3D, tLocalState);
    
    auto t_valueNeg = aScalarFunction.value(aSolution, tLocalStates, aControl);
    std::cout << "t_valueNeg:" << t_valueNeg << std::endl;

    // compute F at c + deltac
    HatchingTestUtils::axpy(2.0, tStep3D, tLocalState);
    auto t_valuePos = aScalarFunction.value(aSolution, tLocalStates, aControl);
    std::cout << "t_valuePos:" << t_valuePos << std::endl;
    HatchingTestUtils::axpy(-1.0, tStep3D, tLocalState);

    // compute actual change in F over 2 * deltaZ
    auto tDeltaFD = (t_valuePos - t_valueNeg);
    std::cout << "tDeltaFD:" << tDeltaFD << std::endl;

    // compute predicted change in F over 2 * deltaZ
    auto tDeltaAD = 2.0*Plato::blas1::dot(t_dFdc, tStep);
    std::cout << "tDeltaAD:" << tDeltaAD << std::endl;

    // return error
    if (fabs(tDeltaFD) < 1e-12)
    {
      return fabs(tDeltaAD);
    }
    else
    {
      Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
      std::cout << "Error: " << std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0) << std::endl;
      return std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
    }
}

template <class ScalarFunctionT, class SolutionT, class ControlT>
Plato::Scalar
testScalarFunction_Partial_u(ScalarFunctionT aScalarFunction, SolutionT aSolution, ControlT aControl, int aTimeStep, Plato::Scalar aAlpha = 1.0e-4)
{
    // compute initial F and dFdu
    Plato::ScalarArray4D tLocalState;
    aSolution.get("Local State", tLocalState);
    auto t_value0 = aScalarFunction.value(aSolution, tLocalState, aControl);
    auto t_dFdu = aScalarFunction.gradient_u(aSolution, tLocalState, aControl, aTimeStep);

    auto tNorm = Plato::blas1::norm(t_dFdu);

    // compute F at u - deltau
    auto tState = aSolution.get("State");
    Plato::ScalarVector tGlobalState = Kokkos::subview(tState, aTimeStep, Kokkos::ALL());
    Plato::blas1::axpy(-aAlpha/tNorm, t_dFdu, tGlobalState);
    auto t_valueNeg = aScalarFunction.value(aSolution, tLocalState, aControl);

    // compute F at u + deltau
    Plato::blas1::axpy(2.0*aAlpha/tNorm, t_dFdu, tGlobalState);
    auto t_valuePos = aScalarFunction.value(aSolution, tLocalState, aControl);
    Plato::blas1::axpy(-aAlpha/tNorm, t_dFdu, tGlobalState);

    // compute actual change in F over 2 * deltaU
    auto tDeltaFD = (t_valuePos - t_valueNeg);


    // compute predicted change in F over 2 * deltaZ
    auto tDeltaAD = Plato::blas1::dot(t_dFdu, t_dFdu);
    if (tDeltaAD != 0.0)
    {
        tDeltaAD *= 2.0*aAlpha/tNorm;
    }

    // return error
    if (fabs(tDeltaFD) < 1.0e-16)
    {
      return fabs(tDeltaAD);
    }
    else
    {
      Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
      return std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
    }
}

} // end namespace HatchingTestUtils


TEUCHOS_UNIT_TEST( EllipticHatchingProblemTests, 3D )
{
  // create test mesh
  //
  constexpr int cMeshWidth=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth);

  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                   \n"
    "  <Parameter name='Physics' type='string' value='Mechanics'/>                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic Hatching'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                           \n"
    "  <ParameterList name='Elliptic Hatching'>                                             \n"
    "    <ParameterList name='Penalty Function'>                                            \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                              \n"
    "      <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
    "      <Parameter name='Minimum Value' type='double' value='1e-5'/>                     \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Spatial Model'>                                                 \n"
    "    <ParameterList name='Domains'>                                                     \n"
    "      <ParameterList name='Design Volume'>                                             \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                   \n"
    "        <Parameter name='Material Model' type='string' value='316 Stainless'/>         \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Criteria'>                                                      \n"
    "    <ParameterList name='Internal Energy'>                                             \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>                   \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/> \n"
    "      <ParameterList name='Penalty Function'>                                          \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                            \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                         \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>                    \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Material Models'>                                               \n"
    "    <ParameterList name='316 Stainless'>                                               \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                                  \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.00'/>                 \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.00e10'/>              \n"
    "        <Parameter  name='e11' type='double' value='0.0'/>                             \n"
    "        <Parameter  name='e22' type='double' value='0.0'/>                             \n"
    "        <Parameter  name='e33' type='double' value='1.0e-6'/>                          \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Sequence'>                                                      \n"
    "    <Parameter name='Sequence Type' type='string' value='Explicit'/>                   \n"
    "    <ParameterList name='Steps'>                                                       \n"
    "      <ParameterList name='Layer 1'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='0.50'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "      <ParameterList name='Layer 2'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='1.00'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                                \n"
    "    <ParameterList  name='X Fixed Displacement - bottom'>                              \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='0'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='Y Fixed Displacement - bottom'>                              \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='1'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='Z Fixed Displacement - bottom'>                              \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='2'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='X Fixed Displacement - top'>                                 \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='0'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z+'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='Y Fixed Displacement - top'>                                 \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='1'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z+'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='Z Fixed Displacement - top'>                                 \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='2'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z+'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "</ParameterList>                                                                       \n"
  );

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  using PhysicsType = Plato::Elliptic::Hatching::Mechanics<Plato::Tet4>;

  int tNumNodes = tMesh->NumNodes();
  Plato::ScalarVector tControl("control", tNumNodes);
  Plato::blas1::fill(1.0, tControl);

  Plato::SpatialModel tSpatialModel(tMesh, *tInputParams);
  Plato::Sequence<typename PhysicsType::ElementType> tSequence(tSpatialModel, *tInputParams);
  Plato::DataMap tDataMap;

  // create PDE constraint
  //
  std::string tMyConstraint = tInputParams->get<std::string>("PDE Constraint");
  Plato::Elliptic::Hatching::VectorFunction<PhysicsType>
    vectorFunction(tSpatialModel, tDataMap, *tInputParams, tMyConstraint);

  Plato::Solutions tSolution;
  {
    Plato::Elliptic::Hatching::Problem<PhysicsType> tProblem(tMesh, *tInputParams, tMachine);
    tSolution = tProblem.solution(tControl);
  }

  // compute and test constraint gradient_z
  //
  auto t_dRdz0_error = HatchingTestUtils::testVectorFunction_Partial_z<typename PhysicsType::ElementType>
                         (vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRdz0_error < 1.0e-6);

  auto t_dRdz1_error = HatchingTestUtils::testVectorFunction_Partial_z<typename PhysicsType::ElementType>
                         (vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRdz1_error < 1.0e-6);

  // compute and test constraint gradient_u
  //
  auto t_dRdu0_error = HatchingTestUtils::testVectorFunction_Partial_u<typename PhysicsType::ElementType>
                         (vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRdu0_error < 1.0e-6);

  auto t_dRdu1_error = HatchingTestUtils::testVectorFunction_Partial_u<typename PhysicsType::ElementType>
                         (vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRdu1_error < 1.0e-6);

  // compute and test constraint gradient_u_T
  //
  auto t_dRduT0_error = HatchingTestUtils::testVectorFunction_Partial_u_T<typename PhysicsType::ElementType>
                          (vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRduT0_error < 1.0e-6);

  auto t_dRduT1_error = HatchingTestUtils::testVectorFunction_Partial_u_T<typename PhysicsType::ElementType>
                          (vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRduT1_error < 1.0e-6);

  // compute and test constraint gradient_cp_T
  //
  auto t_dRdcpT0_error = HatchingTestUtils::testVectorFunction_Partial_cp_T<typename PhysicsType::ElementType>
                           (vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRdcpT0_error < 1.0e-6);

  auto t_dRdcpT1_error = HatchingTestUtils::testVectorFunction_Partial_cp_T<typename PhysicsType::ElementType>
                           (vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRdcpT1_error < 1.0e-6);


  // create objective
  //
  std::string tMyFunction("Internal Energy");
  using FunctionType = Plato::Elliptic::Hatching::PhysicsScalarFunction<PhysicsType>;
  FunctionType scalarFunction(tSpatialModel, tSequence, tDataMap, *tInputParams, tMyFunction);

  // compute and test criterion value
  //
  Plato::ScalarArray4D tLocalState;
  tSolution.get("Local State", tLocalState);
  auto t_value = scalarFunction.value(tSolution, tLocalState, tControl);
  TEST_FLOATING_EQUALITY(t_value, -0.00125000, 1e-7);

  // compute and test criterion gradient_z
  //
  auto t_dFdz_error = HatchingTestUtils::testScalarFunction_Partial_z(scalarFunction, tSolution, tControl);
  TEST_ASSERT(t_dFdz_error < 1.0e-10);

  // compute and test criterion gradient_x
  //
  {
    // set exact solution
    auto tState = tSolution.get("State");
    auto tGlobalState_Host = Kokkos::create_mirror(tState);
    std::vector<int> tDispIndices({5,14,23,32,41,50,59,68,77});
    for(int i=0; i<tDispIndices.size(); i++)
    {
      tGlobalState_Host(0, tDispIndices[i]) = 5.0e-7;
      tGlobalState_Host(1, tDispIndices[i]) =-2.5e-7;
    }
    Kokkos::deep_copy(tState, tGlobalState_Host);

    Plato::ScalarArray4D tLocalState;
    tSolution.get("Local State", tLocalState);
    auto tLocalState_Host = Kokkos::create_mirror(tLocalState);
    auto tCellMask0 = tSequence.getSteps()[0].getMask()->cellMask();
    auto tCellMask0_Host = Kokkos::create_mirror(tCellMask0);
    Kokkos::deep_copy(tCellMask0_Host, tCellMask0);

    auto tNumCells = tLocalState_Host.extent(1);
    for(int iCell=0; iCell<tNumCells; iCell++)
    {
      if( tCellMask0_Host(iCell) ) tLocalState_Host(0, iCell, /*iGP=*/ 0, 2) = 1.0e-6;
      tLocalState_Host(1, iCell, /*iGP=*/ 0, 2) = 5.0e-7;
    }
    Kokkos::deep_copy(tLocalState, tLocalState_Host);

    // compute initial F and dFdz
    auto t_value0 = scalarFunction.value(tSolution, tLocalState, tControl);
    auto t_dFdx = scalarFunction.gradient_x(tSolution, tLocalState, tControl);

    Plato::Scalar tAlpha = 1.0e-4;
    auto tNorm = Plato::blas1::norm(t_dFdx);

    Plato::ScalarVector tStep("step", t_dFdx.extent(0));
    Kokkos::deep_copy(tStep, t_dFdx);
    Plato::blas1::scale(-tAlpha/tNorm, tStep);

    // compute F at z - deltaZ
    HatchingTestUtils::perturbMesh(tMesh, tStep);
    FunctionType scalarFunctionNeg(tSpatialModel, tSequence, tDataMap, *tInputParams, tMyFunction);
    auto t_valueNeg = scalarFunctionNeg.value(tSolution, tLocalState, tControl);

    // compute F at z + deltaZ
    Plato::blas1::scale(-2.0, tStep);
    HatchingTestUtils::perturbMesh(tMesh, tStep);
    FunctionType scalarFunctionPos(tSpatialModel, tSequence, tDataMap, *tInputParams, tMyFunction);
    auto t_valuePos = scalarFunctionPos.value(tSolution, tLocalState, tControl);

    Plato::blas1::scale(-1.0/2.0, tStep);
    HatchingTestUtils::perturbMesh(tMesh, tStep);

    // compute actual change in F over 2 * deltaZ
    auto tDeltaFD = (t_valuePos - t_valueNeg);

    // compute predicted change in F over 2 * deltaZ
    Plato::blas1::scale(-2.0, tStep);
    auto tDeltaAD = Plato::blas1::dot(t_dFdx, tStep);

    // return error
    Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
    Plato::Scalar t_dFdx_error = std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
    TEST_ASSERT(t_dFdx_error < 1.0e-8);
  }

  // test gradient_z against semi-analytical values (generated w/ mathematica)
  //
  auto t_dFdz = scalarFunction.gradient_z(tSolution, tLocalState, tControl);
  auto t_dFdz_Host = Kokkos::create_mirror(t_dFdz);
  Kokkos::deep_copy(t_dFdz_Host, t_dFdz);

  std::vector<Plato::Scalar> t_dFdz_Gold = {
  -0.00003906250000000001, -0.00005208333333333334, -0.00001302083333333333,
  -0.00005208333333333334, -0.00007812500000000000, -0.00002604166666666666,
  -0.00001302083333333334, -0.00002604166666666667, -0.00001302083333333333,
  -0.00005208333333333334, -0.00007812500000000000, -0.00002604166666666666,
  -0.00007812500000000002, -0.0001562500000000000,  -0.00007812500000000000,
  -0.00002604166666666667, -0.00007812500000000002, -0.00005208333333333334,
  -0.00001302083333333334, -0.00002604166666666667, -0.00001302083333333333,
  -0.00002604166666666667, -0.00007812500000000002, -0.00005208333333333334,
  -0.00001302083333333334, -0.00005208333333333334, -0.00003906250000000000
  };
  for(Plato::OrdinalType tIndex = 0; tIndex < t_dFdz_Gold.size(); tIndex++)
  {
      TEST_FLOATING_EQUALITY(t_dFdz_Host(tIndex), t_dFdz_Gold[tIndex], 1e-6);
  }

  // compute and test criterion gradient_c
  //
  auto t_dFdc0_error = HatchingTestUtils::testScalarFunction_Partial_c(scalarFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dFdc0_error < 1.0e-8);

  auto t_dFdc1_error = HatchingTestUtils::testScalarFunction_Partial_c(scalarFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dFdc1_error < 1.0e-8);


  // compute and test criterion gradient_u
  //
  auto t_dFdu0_error = HatchingTestUtils::testScalarFunction_Partial_u(scalarFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dFdu0_error < 1.0e-8);

  auto t_dFdu1_error = HatchingTestUtils::testScalarFunction_Partial_u(scalarFunction, tSolution, tControl, /*timeStep=*/ 1, /*stepSize=*/ 1.0e-8);
  TEST_ASSERT(t_dFdu1_error < 1.0e-8);

  auto tCriterionName = "Internal Energy";
  Plato::ScalarVector t_dFdx;
  {
    Plato::Elliptic::Hatching::Problem<PhysicsType> tProblem(tMesh, *tInputParams, tMachine);
    tSolution = tProblem.solution(tControl);

    /*****************************************************
     Test Problem::criterionValue(aControl);
     *****************************************************/

    auto tCriterionValue = tProblem.criterionValue(tControl, "Internal Energy");
    Plato::Scalar tCriterionValue_gold = -0.00125;

    TEST_FLOATING_EQUALITY( tCriterionValue, tCriterionValue_gold, 1e-7);


    /*****************************************************
     Test Problem::criterionGradient(aControl);
     *****************************************************/

    auto t_dPdz_error = HatchingTestUtils::testProblem_Total_z<typename PhysicsType::ElementType>(tProblem, tControl, "Internal Energy", /*stepsize=*/ 1e-4);
    TEST_ASSERT(t_dPdz_error < 1.0e-6);


    /*****************************************************
     Call Problem::criterionGradientX(aControl);
     *****************************************************/

    // compute initial F and dFdx
    tSolution = tProblem.solution(tControl);
    t_dFdx = tProblem.criterionGradientX(tControl, tCriterionName);
  }

  Plato::Scalar tAlpha = 1.0e-4;
  auto tNorm = Plato::blas1::norm(t_dFdx);

  Plato::ScalarVector tStep("step", t_dFdx.extent(0));
  Kokkos::deep_copy(tStep, t_dFdx);
  Plato::blas1::scale(-tAlpha/tNorm, tStep);

  // compute F at x - deltax
  Plato::Scalar t_valueNeg(0);
  {
    HatchingTestUtils::perturbMesh(tMesh, tStep);
    Plato::Elliptic::Hatching::Problem<PhysicsType> tProblem2(tMesh, *tInputParams, tMachine);
    tSolution = tProblem2.solution(tControl);
    t_valueNeg = tProblem2.criterionValue(tControl, tCriterionName);
  }

  Plato::Scalar t_valueNegToo(0);
  {
    Plato::Elliptic::Hatching::Problem<PhysicsType> tProblem3(tMesh, *tInputParams, tMachine);
    tSolution = tProblem3.solution(tControl);
    t_valueNegToo = tProblem3.criterionValue(tControl, tCriterionName);
  }
  TEST_FLOATING_EQUALITY(t_valueNeg, t_valueNegToo, 1e-15);

  // compute F at x + deltax
  Plato::blas1::scale(-2.0, tStep);
  Plato::Scalar t_valuePos(0);
  {
    HatchingTestUtils::perturbMesh(tMesh, tStep);
    Plato::Elliptic::Hatching::Problem<PhysicsType> tProblem4(tMesh, *tInputParams, tMachine);
    tSolution = tProblem4.solution(tControl);
    t_valuePos = tProblem4.criterionValue(tControl, tCriterionName);
  }

  // compute actual change in F over 2 * deltax
  auto tDeltaFD = (t_valuePos - t_valueNeg);

  // compute predicted change in F over 2 * deltaZ
  auto tDeltaAD = Plato::blas1::dot(t_dFdx, tStep);

  // check error
  Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
  auto t_dPdx_error = std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
  TEST_ASSERT(t_dPdx_error < 1.0e-6);

  // change mesh back 
  Plato::blas1::scale(-1.0/2.0, tStep);
  HatchingTestUtils::perturbMesh(tMesh, tStep);
}



TEUCHOS_UNIT_TEST( EllipticHatchingProblemTests, 3D_full )
{
  // create test mesh
  //
  constexpr int cMeshWidth=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth);

  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                   \n"
    "  <Parameter name='Physics' type='string' value='Mechanics'/>                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic Hatching'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                           \n"
    "  <ParameterList name='Elliptic Hatching'>                                             \n"
    "    <ParameterList name='Penalty Function'>                                            \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                              \n"
    "      <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
    "      <Parameter name='Minimum Value' type='double' value='1e-5'/>                     \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Spatial Model'>                                                 \n"
    "    <ParameterList name='Domains'>                                                     \n"
    "      <ParameterList name='Design Volume'>                                             \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                   \n"
    "        <Parameter name='Material Model' type='string' value='316 Stainless'/>         \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Criteria'>                                                      \n"
    "    <ParameterList name='Internal Energy'>                                             \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>                   \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/> \n"
    "      <ParameterList name='Penalty Function'>                                          \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                            \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                         \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>                    \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Material Models'>                                               \n"
    "    <ParameterList name='316 Stainless'>                                               \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                                  \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.30'/>                 \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.00e10'/>              \n"
    "        <Parameter  name='e11' type='double' value='-1e-3'/>                           \n"
    "        <Parameter  name='e22' type='double' value='-1e-3'/>                           \n"
    "        <Parameter  name='e33' type='double' value='2.0e-3'/>                          \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Sequence'>                                                      \n"
    "    <Parameter name='Sequence Type' type='string' value='Explicit'/>                   \n"
    "    <ParameterList name='Steps'>                                                       \n"
    "      <ParameterList name='Layer 1'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='0.50'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "      <ParameterList name='Layer 2'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='1.00'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                                \n"
    "    <ParameterList  name='X Fixed Displacement - bottom'>                              \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='0'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='Y Fixed Displacement - bottom'>                              \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='1'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='Z Fixed Displacement - bottom'>                              \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='2'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "</ParameterList>                                                                       \n"
  );

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  using PhysicsType = Plato::Elliptic::Hatching::Mechanics<Plato::Tet4>;
  auto* tProblem = new Plato::Elliptic::Hatching::Problem<PhysicsType> (tMesh, *tInputParams, tMachine);

  TEST_ASSERT(tProblem != nullptr);

  int tNumNodes = tMesh->NumNodes();
  Plato::ScalarVector tControl("control", tNumNodes);
  Plato::blas1::fill(1.0, tControl);

  Plato::SpatialModel tSpatialModel(tMesh, *tInputParams);
  Plato::Sequence<typename PhysicsType::ElementType> tSequence(tSpatialModel, *tInputParams);
  Plato::DataMap tDataMap;

  auto tSolution = tProblem->solution(tControl);

#ifdef NOPE
  // create PDE constraint
  //
  std::string tMyConstraint = tInputParams->get<std::string>("PDE Constraint");
  Plato::Elliptic::Hatching::VectorFunction<PhysicsType>
    vectorFunction(tSpatialModel, tDataMap, *tInputParams, tMyConstraint);

  // compute and test constraint gradient_z
  //
  auto t_dRdz0_error = HatchingTestUtils::testVectorFunction_Partial_z<typename PhysicsType::ElementType>
                         (vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRdz0_error < 1.0e-6);

  auto t_dRdz1_error = HatchingTestUtils::testVectorFunction_Partial_z<typename PhysicsType::ElementType>
                         (vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRdz1_error < 1.0e-6);

  // compute and test constraint gradient_u
  //
  auto t_dRdu0_error = HatchingTestUtils::testVectorFunction_Partial_u<typename PhysicsType::ElementType>
                         (vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRdu0_error < 1.0e-6);

  auto t_dRdu1_error = HatchingTestUtils::testVectorFunction_Partial_u<typename PhysicsType::ElementType>
                         (vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRdu1_error < 1.0e-6);

  // compute and test constraint gradient_u_T
  //
  auto t_dRduT0_error = HatchingTestUtils::testVectorFunction_Partial_u_T<typename PhysicsType::ElementType>
                          (vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRduT0_error < 1.0e-6);

  auto t_dRduT1_error = HatchingTestUtils::testVectorFunction_Partial_u_T<typename PhysicsType::ElementType>
                          (vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRduT1_error < 1.0e-6);

  // compute and test constraint gradient_cp_T
  //
  auto t_dRdcpT0_error = HatchingTestUtils::testVectorFunction_Partial_cp_T<typename PhysicsType::ElementType>
                           (vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRdcpT0_error < 1.0e-6);

  auto t_dRdcpT1_error = HatchingTestUtils::testVectorFunction_Partial_cp_T<typename PhysicsType::ElementType>
                           (vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRdcpT1_error < 1.0e-6);


  // create objective
  //
  std::string tMyFunction("Internal Energy");
  using FunctionType = Plato::Elliptic::Hatching::PhysicsScalarFunction<PhysicsType>;
  FunctionType scalarFunction(tSpatialModel, tSequence, tDataMap, *tInputParams, tMyFunction);

  // compute and test criterion gradient_z
  //
  auto t_dFdz_error = HatchingTestUtils::testScalarFunction_Partial_z(scalarFunction, tSolution, tControl);
  TEST_ASSERT(t_dFdz_error < 1.0e-10);

  // compute and test criterion gradient_x
  //
  {
    // compute initial F and dFdx
    Plato::ScalarArray4D tLocalState;
    tSolution.get("Local State", tLocalState);
    auto t_value0 = scalarFunction.value(tSolution, tLocalState, tControl);
    auto t_dFdx = scalarFunction.gradient_x(tSolution, tLocalState, tControl);

    Plato::Scalar tAlpha = 1.0e-6;
    auto tNorm = Plato::blas1::norm(t_dFdx);

    Plato::ScalarVector tStep("step", t_dFdx.extent(0));
    Kokkos::deep_copy(tStep, t_dFdx);
    Plato::blas1::scale(-tAlpha/tNorm, tStep);

    // compute F at z - deltaZ
    HatchingTestUtils::perturbMesh(tMesh, tStep);
    FunctionType scalarFunctionNeg(tSpatialModel, tSequence, tDataMap, *tInputParams, tMyFunction);
    auto t_valueNeg = scalarFunctionNeg.value(tSolution, tLocalState, tControl);

    // compute F at z + deltaZ
    Plato::blas1::scale(-2.0, tStep);
    HatchingTestUtils::perturbMesh(tMesh, tStep);
    FunctionType scalarFunctionPos(tSpatialModel, tSequence, tDataMap, *tInputParams, tMyFunction);
    auto t_valuePos = scalarFunctionPos.value(tSolution, tLocalState, tControl);

    Plato::blas1::scale(-1.0/2.0, tStep);
    HatchingTestUtils::perturbMesh(tMesh, tStep);

    // compute actual change in F over 2 * deltaZ
    auto tDeltaFD = (t_valuePos - t_valueNeg);

    // compute predicted change in F over 2 * deltaZ
    Plato::blas1::scale(-2.0, tStep);
    auto tDeltaAD = Plato::blas1::dot(t_dFdx, tStep);

    // return error
    Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
    Plato::Scalar t_dFdx_error = std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
    TEST_ASSERT(t_dFdx_error < 1.0e-10);
  }


  // compute and test criterion gradient_c
  //
  auto t_dFdc0_error = HatchingTestUtils::testScalarFunction_Partial_c(scalarFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dFdc0_error < 1.0e-10);

  auto t_dFdc1_error = HatchingTestUtils::testScalarFunction_Partial_c(scalarFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dFdc1_error < 1.0e-10);


  // compute and test criterion gradient_u
  //
  auto t_dFdu0_error = HatchingTestUtils::testScalarFunction_Partial_u(scalarFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dFdu0_error < 1.0e-10);

  auto t_dFdu1_error = HatchingTestUtils::testScalarFunction_Partial_u(scalarFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dFdu1_error < 1.0e-10);

#endif

  /*****************************************************
   Test Problem::criterionGradient(aControl);
   *****************************************************/

  auto t_dPdz_error = HatchingTestUtils::testProblem_Total_z<typename PhysicsType::ElementType>(*tProblem, tControl, "Internal Energy", 1.0e-4);
  TEST_ASSERT(t_dPdz_error < 1.0e-6);


  /*****************************************************
   Call Problem::criterionGradientX(aControl);
   *****************************************************/
  auto tCriterionName = "Internal Energy";

  // compute initial F and dFdx
  tSolution = tProblem->solution(tControl);
  auto t_dFdx = tProblem->criterionGradientX(tControl, tCriterionName);

  Plato::Scalar tAlpha = 1.0e-4;
  auto tNorm = Plato::blas1::norm(t_dFdx);

  Plato::ScalarVector tStep("step", t_dFdx.extent(0));
  Kokkos::deep_copy(tStep, t_dFdx);
  Plato::blas1::scale(-tAlpha/tNorm, tStep);

  // compute F at x - deltax
  HatchingTestUtils::perturbMesh(tMesh, tStep);
  delete tProblem;
  tProblem = new Plato::Elliptic::Hatching::Problem<PhysicsType> (tMesh, *tInputParams, tMachine);
  tSolution = tProblem->solution(tControl);
  auto t_valueNeg = tProblem->criterionValue(tControl, tCriterionName);

  delete tProblem;
  tProblem = new Plato::Elliptic::Hatching::Problem<PhysicsType> (tMesh, *tInputParams, tMachine);
  tSolution = tProblem->solution(tControl);
  auto t_valueNegToo = tProblem->criterionValue(tControl, tCriterionName);
  TEST_FLOATING_EQUALITY(t_valueNeg, t_valueNegToo, 1e-15);

  // compute F at x + deltax
  Plato::blas1::scale(-2.0, tStep);
  HatchingTestUtils::perturbMesh(tMesh, tStep);
  delete tProblem;
  tProblem = new Plato::Elliptic::Hatching::Problem<PhysicsType> (tMesh, *tInputParams, tMachine);
  tSolution = tProblem->solution(tControl);
  auto t_valuePos = tProblem->criterionValue(tControl, tCriterionName);

  // compute actual change in F over 2 * deltax
  auto tDeltaFD = (t_valuePos - t_valueNeg);

  // compute predicted change in F over 2 * deltaZ
  auto tDeltaAD = Plato::blas1::dot(t_dFdx, tStep);

  // check error
  Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
  auto t_dPdx_error = std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
  TEST_ASSERT(t_dPdx_error < 1.0e-6);

  // change mesh back 
  Plato::blas1::scale(-1.0/2.0, tStep);
  HatchingTestUtils::perturbMesh(tMesh, tStep);

  delete tProblem;
}

TEUCHOS_UNIT_TEST( EllipticHatchingProblemTests, 3D_StateUpdate )
{
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                   \n"
    "  <Parameter name='Physics' type='string' value='Mechanics'/>                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic Hatching'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                           \n"
    "  <ParameterList name='Elliptic Hatching'>                                             \n"
    "    <ParameterList name='Penalty Function'>                                            \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                              \n"
    "      <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
    "      <Parameter name='Minimum Value' type='double' value='1e-5'/>                     \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Spatial Model'>                                                 \n"
    "    <ParameterList name='Domains'>                                                     \n"
    "      <ParameterList name='Design Volume'>                                             \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                   \n"
    "        <Parameter name='Material Model' type='string' value='316 Stainless'/>         \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Sequence'>                                                      \n"
    "    <Parameter name='Sequence Type' type='string' value='Explicit'/>                   \n"
    "    <ParameterList name='Steps'>                                                       \n"
    "      <ParameterList name='Layer 2'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='1.00'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "</ParameterList>                                                                       \n"
  );

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth, "omfg.exo");

  Plato::SpatialModel tSpatialModel(tMesh, *tInputParams);

  /*****************************************************
   Test Elliptic::StateUpdate(aMesh);
   *****************************************************/

  using PhysicsType = Plato::Elliptic::Hatching::Mechanics<Plato::Tet4>;
  auto* tStateUpdate = new Plato::StateUpdate<PhysicsType> (tSpatialModel);


  /*****************************************************
   Call StateUpdate::operator()
   *****************************************************/

   Plato::DataMap tDataMap;

   auto tNumEl = tMesh->NumElements();
   constexpr auto cNumGP = PhysicsType::ElementType::mNumGaussPoints;
   constexpr auto cNumVT = PhysicsType::ElementType::mNumVoigtTerms;
   constexpr auto cNumDm = PhysicsType::ElementType::mNumSpatialDims;

   Plato::Scalar tTestVal = 1.0;
   // create 'strain increment' view
   Plato::ScalarArray3D tStrainIncrement("strain increment", tNumEl, cNumGP, cNumVT);
   Kokkos::deep_copy(tStrainIncrement, tTestVal);

   // add 'strain increment' view to datamap
   Plato::toMap(tDataMap, tStrainIncrement, "strain increment");

   // define current and updated state view
   Plato::ScalarArray3D tLocalState("current state", tNumEl, cNumGP, cNumVT);
   Plato::ScalarArray3D tUpdatedLocalState("current state", tNumEl, cNumGP, cNumVT);

   // compute updated local state
   tStateUpdate->operator()(tDataMap, tLocalState, tUpdatedLocalState);

   // check values
   auto tUpdatedLocalState_Host = Kokkos::create_mirror_view(tUpdatedLocalState);
   Kokkos::deep_copy(tUpdatedLocalState_Host, tUpdatedLocalState);

  for(int iElem=0; iElem<tNumEl; iElem++){
    for(int iGp=0; iGp<cNumGP; iGp++){
      for(int iTerm=0; iTerm<cNumVT; iTerm++){
        TEST_FLOATING_EQUALITY(tUpdatedLocalState_Host(iElem, iGp, iTerm), tTestVal, 1e-14);
      }
    }
  }

  /*****************************************************
   Call StateUpdate::gradient_*()
   *****************************************************/

  // create a displacement field, u_x = x, u_y = 0, u_z = 0
  auto tNumNodes = tMesh->NumNodes();
  auto tCoords = tMesh->Coordinates();
  Plato::ScalarVector tU("displacement", tNumNodes * cNumDm);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(int aNodeOrdinal)
  {
    tU(aNodeOrdinal * cNumDm + 0) = tCoords[aNodeOrdinal * cNumDm + 0];
    tU(aNodeOrdinal * cNumDm + 1) = 0.0;
    tU(aNodeOrdinal * cNumDm + 2) = 0.0;
  }, "initial data");

  auto t_dHdx = tStateUpdate->gradient_x(tU, tUpdatedLocalState, tLocalState);

  auto t_dHdx_entries = t_dHdx->entries();
  auto t_dHdx_entriesHost = Kokkos::create_mirror_view( t_dHdx_entries );
  Kokkos::deep_copy(t_dHdx_entriesHost, t_dHdx_entries);

  std::vector<Plato::Scalar> gold_t_dHdx_entries = {
    0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   -2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  int t_dHdx_entriesSize = gold_t_dHdx_entries.size();
  for(int i=0; i<t_dHdx_entriesSize; i++){
    TEST_FLOATING_EQUALITY(t_dHdx_entriesHost(i), gold_t_dHdx_entries[i], 1.0e-14);
  }


  auto t_dHdu = tStateUpdate->gradient_u_T(tU, tUpdatedLocalState, tLocalState);

  auto t_dHdu_entries = t_dHdu->entries();
  auto t_dHdu_entriesHost = Kokkos::create_mirror_view( t_dHdu_entries );
  Kokkos::deep_copy(t_dHdu_entriesHost, t_dHdu_entries);

  std::vector<Plato::Scalar> gold_t_dHdu_entries = {
    0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
    0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
    0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0,
    0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0,
    0, 0, 0, 0,-2, 2, 0, 2, 0,-2, 0, 0, 0, 0,-2, 2, 0, 0,
    2, 0, 0, 0,-2, 0, 0, 0, 0,-2, 0, 2, 0, 0,-2, 0, 2, 0,
    0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
    0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0
  };

  int t_dHdu_entriesSize = gold_t_dHdu_entries.size();
  for(int i=0; i<t_dHdu_entriesSize; i++){
    TEST_FLOATING_EQUALITY(t_dHdu_entriesHost(i), gold_t_dHdu_entries[i], 1.0e-14);
  }

  delete tStateUpdate;
}


TEUCHOS_UNIT_TEST( EllipticHatchingProblemTests, 3D_StateUpdate_2layer )
{
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                   \n"
    "  <Parameter name='Physics' type='string' value='Mechanics'/>                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic Hatching'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                           \n"
    "  <ParameterList name='Elliptic Hatching'>                                             \n"
    "    <ParameterList name='Penalty Function'>                                            \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                              \n"
    "      <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
    "      <Parameter name='Minimum Value' type='double' value='1e-5'/>                     \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Spatial Model'>                                                 \n"
    "    <ParameterList name='Domains'>                                                     \n"
    "      <ParameterList name='Design Volume'>                                             \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                   \n"
    "        <Parameter name='Material Model' type='string' value='316 Stainless'/>         \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Material Models'>                                               \n"
    "    <ParameterList name='316 Stainless'>                                               \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                                  \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.00'/>                 \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.00e10'/>              \n"
    "        <Parameter  name='e11' type='double' value='0.0'/>                             \n"
    "        <Parameter  name='e22' type='double' value='0.0'/>                             \n"
    "        <Parameter  name='e33' type='double' value='1.0e-6'/>                          \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Sequence'>                                                      \n"
    "    <Parameter name='Sequence Type' type='string' value='Explicit'/>                   \n"
    "    <ParameterList name='Steps'>                                                       \n"
    "      <ParameterList name='Layer 1'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='0.50'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "      <ParameterList name='Layer 2'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='1.00'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "</ParameterList>                                                                       \n"
  );

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", cMeshWidth);

  using PhysicsType = Plato::Elliptic::Hatching::Mechanics<Plato::Tet4>;

  Plato::SpatialModel tSpatialModel(tMesh, *tInputParams);
  Plato::Sequence<typename PhysicsType::ElementType> tSequence(tSpatialModel, *tInputParams);

  /*****************************************************
   Test Elliptic::StateUpdate(aMesh);
   *****************************************************/

  auto* tStateUpdate = new Plato::StateUpdate<PhysicsType> (tSpatialModel);

  int tNumNodes = tMesh->NumNodes();
  Plato::ScalarVector tControl("control", tNumNodes);
  Plato::blas1::fill(1.0, tControl);

  auto tNumEl = tMesh->NumElements();
  constexpr auto cNumGP = PhysicsType::ElementType::mNumGaussPoints;
  constexpr auto cNumVT = PhysicsType::ElementType::mNumVoigtTerms;
  constexpr auto cNumDm = PhysicsType::ElementType::mNumSpatialDims;
  constexpr auto cNumDPN = PhysicsType::ElementType::mNumDofsPerNode;

  // create solution 
  Plato::DataMap tDataMap;
  Plato::ScalarMultiVector tGlobalStates("global state", /*numsteps=*/ 2, tMesh->NumNodes() * cNumDPN);
  Plato::ScalarArray4D tLocalStates("local state", /*numsteps=*/ 2, tNumEl, cNumGP, cNumVT);
  {
    auto tGlobalStates_Host = Kokkos::create_mirror(tGlobalStates);
    std::vector<int> tDispIndices({5,11,20,41,50,53,62,65,74});
    for(int i=0; i<tDispIndices.size(); i++)
    {
      tGlobalStates_Host(0, tDispIndices[i]) = 5.0e-7;
      tGlobalStates_Host(1, tDispIndices[i]) =-2.5e-7;
    }
    Kokkos::deep_copy(tGlobalStates, tGlobalStates_Host);

    auto tLocalStates_Host = Kokkos::create_mirror(tLocalStates);
    auto tCellMask0 = tSequence.getSteps()[0].getMask()->cellMask();
    auto tCellMask0_Host = Kokkos::create_mirror(tCellMask0);
    Kokkos::deep_copy(tCellMask0_Host, tCellMask0);

    for(int iElem=0; iElem<tNumEl; iElem++)
    {
      if( tCellMask0_Host(iElem) ) tLocalStates_Host(0, iElem, /*iGP=*/ 0, 2) = 1.0e-6;
      tLocalStates_Host(1, iElem, /*iGP=*/ 0, 2) = 5.0e-7;
    }
    Kokkos::deep_copy(tLocalStates, tLocalStates_Host);

    // create 'strain increment' view
    Plato::ScalarArray3D tStrainIncrement("strain increment", tMesh->NumElements(), cNumGP, cNumVT);

    // add 'strain increment' view to datamap
    Plato::toMap(tDataMap, tStrainIncrement, "strain increment");
  }

  /*****************************************************
   Test StateUpdate::gradient_x()
   *****************************************************/

  using VectorFunctionType = Plato::Elliptic::Hatching::VectorFunction<PhysicsType>;
  auto tResidualFunction = std::make_shared<VectorFunctionType>(tSpatialModel, tDataMap, *tInputParams, tInputParams->get<std::string>("PDE Constraint"));

  int tStepIndex = 1;

  // compute updated local state, eta_0, at x_0
  Plato::ScalarVector tGlobalState = Kokkos::subview(tGlobalStates, tStepIndex, Kokkos::ALL());
  Plato::ScalarArray3D tLocalState = Kokkos::subview(tLocalStates, tStepIndex-1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
  auto tResidual = tResidualFunction->value(tGlobalState, tLocalState, tControl);
  Plato::ScalarArray3D tUpdatedLocalState_0("eta_0", tLocalState.extent(0), tLocalState.extent(1), tLocalState.extent(2));
  tStateUpdate->operator()(tDataMap, tLocalState, tUpdatedLocalState_0);

  // compute gradient_x at x_0
  auto t_dHdx = tStateUpdate->gradient_x(tGlobalState, tUpdatedLocalState_0, tLocalState);

  Plato::ScalarVector tStep = Plato::ScalarVector("Step", tMesh->NumNodes() * cNumDm);
  auto tHostStep = Kokkos::create_mirror(tStep);
  Plato::blas1::random(0.0, 0.00001, tHostStep);
  Kokkos::deep_copy(tStep, tHostStep);

  // perturb mesh with deltaX (now at x_1)
  HatchingTestUtils::perturbMesh(tMesh, tStep);
  tResidualFunction = nullptr;
  tResidualFunction = std::make_shared<VectorFunctionType>(tSpatialModel, tDataMap, *tInputParams, tInputParams->get<std::string>("PDE Constraint"));

  // compute updated local state, eta_1, at x_1
  Plato::ScalarArray3D tUpdatedLocalState_1("eta_1", tLocalState.extent(0), tLocalState.extent(1), tLocalState.extent(2));
  tResidual = tResidualFunction->value(tGlobalState, tLocalState, tControl);
  tStateUpdate->operator()(tDataMap, tLocalState, tUpdatedLocalState_1);

  // compute deltaFD (eta_1 - eta_0)
  Plato::ScalarVector tDiffFD("difference", tUpdatedLocalState_0.extent(0)*tUpdatedLocalState_0.extent(1)*tUpdatedLocalState_0.extent(2));
  HatchingTestUtils::Flatten(tUpdatedLocalState_1, tDiffFD);
  Plato::ScalarVector tState_0("difference", tUpdatedLocalState_0.extent(0)*tUpdatedLocalState_0.extent(1)*tUpdatedLocalState_0.extent(2));
  HatchingTestUtils::Flatten(tUpdatedLocalState_0, tState_0);
  Plato::blas1::axpy(-1.0, tState_0, tDiffFD);
  auto tNormFD = Plato::blas1::norm(tDiffFD);

  // compute deltaAD (gradient_x . deltaX)
  Plato::ScalarVector tDiffAD("difference", tUpdatedLocalState_0.extent(0)*tUpdatedLocalState_0.extent(1)*tUpdatedLocalState_0.extent(2));
  Plato::VectorTimesMatrixPlusVector(tStep, t_dHdx, tDiffAD);
  auto tNormAD = Plato::blas1::norm(tDiffAD);

  // perturb mesh back to x_0 (just in case more test are added later)
  Plato::blas1::scale( -1.0, tStep);
  HatchingTestUtils::perturbMesh(tMesh, tStep);

  Plato::Scalar tPer = fabs(tNormFD) + fabs(tNormAD);
  Plato::Scalar t_dHdx_error = std::fabs(tNormFD - tNormAD) / (tPer != 0 ? tPer : 1.0);
  TEST_ASSERT(t_dHdx_error < 5.0e-6);

  delete tStateUpdate;
}
