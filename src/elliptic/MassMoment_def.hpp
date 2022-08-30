#pragma once

#include "elliptic/MassMoment.hpp"

#include "FadTypes.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "Plato_TopOptFunctors.hpp"

namespace Plato
{

namespace Elliptic
{

    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain 
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aInputParams input parameters database
     **********************************************************************************/
    template<typename EvaluationType>
    MassMoment<EvaluationType>::MassMoment(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aInputParams
    ) :
       FunctionBaseType(aSpatialDomain, aDataMap, aInputParams, "MassMoment"),
       mCellMaterialDensity(1.0),
       mCalculationType("")
    /**************************************************************************/
    {
      auto tMaterialModelInputs = aInputParams.get<Teuchos::ParameterList>("Material Model");
      mCellMaterialDensity = tMaterialModelInputs.get<Plato::Scalar>("Density", 1.0);
    }

    /******************************************************************************//**
     * \brief Unit testing constructor
     * \param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    template<typename EvaluationType>
    MassMoment<EvaluationType>::MassMoment(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap& aDataMap
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, "MassMoment"),
        mCellMaterialDensity(1.0),
        mCalculationType(""){}
    /**************************************************************************/

    /******************************************************************************//**
     * \brief set material density
     * \param [in] aMaterialDensity material density
     **********************************************************************************/
    template<typename EvaluationType>
    void MassMoment<EvaluationType>::setMaterialDensity(const Plato::Scalar aMaterialDensity)
    /**************************************************************************/
    {
      mCellMaterialDensity = aMaterialDensity;
    }

    /******************************************************************************//**
     * \brief set calculation type
     * \param [in] aCalculationType calculation type string
     **********************************************************************************/
    template<typename EvaluationType>
    void MassMoment<EvaluationType>::setCalculationType(const std::string & aCalculationType)
    /**************************************************************************/
    {
      mCalculationType = aCalculationType;
    }

    /******************************************************************************//**
     * \brief Evaluate mass moment function
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    template<typename EvaluationType>
    void
    MassMoment<EvaluationType>::evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
      if (mCalculationType == "Mass")
        computeStructuralMass(aControl, aConfig, aResult, aTimeStep);
      else if (mCalculationType == "FirstX")
        computeFirstMoment(aControl, aConfig, aResult, 0, aTimeStep);
      else if (mCalculationType == "FirstY")
        computeFirstMoment(aControl, aConfig, aResult, 1, aTimeStep);
      else if (mCalculationType == "FirstZ")
        computeFirstMoment(aControl, aConfig, aResult, 2, aTimeStep);
      else if (mCalculationType == "SecondXX")
        computeSecondMoment(aControl, aConfig, aResult, 0, 0, aTimeStep);
      else if (mCalculationType == "SecondYY")
        computeSecondMoment(aControl, aConfig, aResult, 1, 1, aTimeStep);
      else if (mCalculationType == "SecondZZ")
        computeSecondMoment(aControl, aConfig, aResult, 2, 2, aTimeStep);
      else if (mCalculationType == "SecondXY")
        computeSecondMoment(aControl, aConfig, aResult, 0, 1, aTimeStep);
      else if (mCalculationType == "SecondXZ")
        computeSecondMoment(aControl, aConfig, aResult, 0, 2, aTimeStep);
      else if (mCalculationType == "SecondYZ")
        computeSecondMoment(aControl, aConfig, aResult, 1, 2, aTimeStep);
      else {
        ANALYZE_THROWERR("Specified mass moment calculation type not implemented.")
      }
    }

    /******************************************************************************//**
     * \brief Compute structural mass
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    template<typename EvaluationType>
    void
    MassMoment<EvaluationType>::computeStructuralMass(
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      auto tCellMaterialDensity = mCellMaterialDensity;

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        auto tCubPoint  = tCubPoints(iGpOrdinal);
        auto tCubWeight = tCubWeights(iGpOrdinal);

        auto tJacobian = ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal);

        ResultScalarType tVolume = Plato::determinant(tJacobian);

        tVolume *= tCubWeight;

        auto tBasisValues = ElementType::basisValues(tCubPoint);
        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(iCellOrdinal, tBasisValues, aControl);

        Kokkos::atomic_add(&aResult(iCellOrdinal), tCellMass * tCellMaterialDensity * tVolume);

      });
    }

    /******************************************************************************//**
     * \brief Compute first mass moment
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [in] aComponent vector component (e.g. x = 0 , y = 1, z = 2)
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    template<typename EvaluationType>
    void
    MassMoment<EvaluationType>::computeFirstMoment(
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::OrdinalType  aComponent,
              Plato::Scalar       aTimeStep
    ) const 
    /**************************************************************************/
    {
      assert(aComponent < mNumSpatialDims);

      auto tNumCells = mSpatialDomain.numCells();

      auto tCellMaterialDensity = mCellMaterialDensity;

      auto tCubPoints  = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints  = tCubWeights.size();

      Plato::ScalarArray3DT<ConfigScalarType> tMappedPoints("mapped quad points", tNumCells, tNumPoints, mNumSpatialDims);
      mapQuadraturePoints(aConfig, tMappedPoints);

      Kokkos::parallel_for("first moment calculation", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        auto tCubPoint  = tCubPoints(iGpOrdinal);
        auto tCubWeight = tCubWeights(iGpOrdinal);

        auto tJacobian = ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal);

        ResultScalarType tVolume = Plato::determinant(tJacobian);

        tVolume *= tCubWeight;

        auto tBasisValues = ElementType::basisValues(tCubPoint);
        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(iCellOrdinal, tBasisValues, aControl);

        ConfigScalarType tMomentArm = tMappedPoints(iCellOrdinal, iGpOrdinal, aComponent);

        Kokkos::atomic_add(&aResult(iCellOrdinal), tCellMass * tCellMaterialDensity * tVolume *tMomentArm);
      });
    }


    /******************************************************************************//**
     * \brief Compute second mass moment
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [in] aComponent1 vector component (e.g. x = 0 , y = 1, z = 2)
     * \param [in] aComponent2 vector component (e.g. x = 0 , y = 1, z = 2)
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    template<typename EvaluationType>
    void
    MassMoment<EvaluationType>::computeSecondMoment(
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::OrdinalType aComponent1,
              Plato::OrdinalType aComponent2,
              Plato::Scalar      aTimeStep
    ) const 
    /**************************************************************************/
    {
      assert(aComponent1 < mNumSpatialDims);
      assert(aComponent2 < mNumSpatialDims);

      auto tNumCells = mSpatialDomain.numCells();

      auto tCellMaterialDensity = mCellMaterialDensity;

      auto tCubPoints  = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints  = tCubWeights.size();

      Plato::ScalarArray3DT<ConfigScalarType> tMappedPoints("mapped quad points", tNumCells, tNumPoints, mNumSpatialDims);
      mapQuadraturePoints(aConfig, tMappedPoints);

      Kokkos::parallel_for("second moment calculation", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        auto tCubPoint  = tCubPoints(iGpOrdinal);
        auto tCubWeight = tCubWeights(iGpOrdinal);

        auto tJacobian = ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal);

        ResultScalarType tVolume = Plato::determinant(tJacobian);

        tVolume *= tCubWeight;

        auto tBasisValues = ElementType::basisValues(tCubPoint);
        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(iCellOrdinal, tBasisValues, aControl);

        ConfigScalarType tMomentArm1 = tMappedPoints(iCellOrdinal, iGpOrdinal, aComponent1);
        ConfigScalarType tMomentArm2 = tMappedPoints(iCellOrdinal, iGpOrdinal, aComponent2);
        ConfigScalarType tSecondMoment  = tMomentArm1 * tMomentArm2;

        Kokkos::atomic_add(&aResult(iCellOrdinal), tCellMass * tCellMaterialDensity * tVolume * tSecondMoment);
      });
    }

    /******************************************************************************//**
     * \brief Map quadrature points to physical domain
     * \param [in] aRefPoint incoming quadrature points
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aMappedPoints points mapped to physical domain
    **********************************************************************************/
    template<typename EvaluationType>
    void
    MassMoment<EvaluationType>::mapQuadraturePoints(
        const Plato::ScalarArray3DT <ConfigScalarType> & aConfig,
              Plato::ScalarArray3DT <ConfigScalarType> & aMappedPoints
    ) const
    /******************************************************************************/
    {
        auto tNumCells = mSpatialDomain.numCells();

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::deep_copy(aMappedPoints, static_cast<ConfigScalarType>(0.0));

        Kokkos::parallel_for("map points", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint    = tCubPoints(iGpOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);

            for (Plato::OrdinalType iDim=0; iDim<mNumSpatialDims; iDim++)
            {
                for (Plato::OrdinalType iNodeOrdinal=0; iNodeOrdinal<mNumNodesPerCell; iNodeOrdinal++)
                {
                    aMappedPoints(iCellOrdinal, iGpOrdinal, iDim) += tBasisValues(iNodeOrdinal) * aConfig(iCellOrdinal, iNodeOrdinal, iDim);
                }
            }
        });
    }
} // namespace Elliptic

} // namespace Plato
