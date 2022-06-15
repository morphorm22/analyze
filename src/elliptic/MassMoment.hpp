#pragma once

#include "PlatoStaticsTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "WorksetBase.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "ImplicitFunctors.hpp"
#include "PlatoMathHelpers.hpp"
#include "AnalyzeMacros.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Mass moment class
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class MassMoment :
  public EvaluationType::ElementType,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;

    using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;
    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mCellMaterialDensity;

    /*!< calculation type = Mass, CGx, CGy, CGz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz */
    std::string mCalculationType;

  public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain 
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aInputParams input parameters database
     **********************************************************************************/
    MassMoment(
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
    MassMoment(
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
    void setMaterialDensity(const Plato::Scalar aMaterialDensity)
    /**************************************************************************/
    {
      mCellMaterialDensity = aMaterialDensity;
    }

    /******************************************************************************//**
     * \brief set calculation type
     * \param [in] aCalculationType calculation type string
     **********************************************************************************/
    void setCalculationType(const std::string & aCalculationType)
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
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
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
    void
    computeStructuralMass(
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      auto tCellMaterialDensity = mCellMaterialDensity;

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
    void
    computeFirstMoment(
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::OrdinalType  aComponent,
              Plato::Scalar       aTimeStep = 0.0
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
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
    void
    computeSecondMoment(
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::OrdinalType aComponent1,
              Plato::OrdinalType aComponent2,
              Plato::Scalar      aTimeStep = 0.0
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
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
    void
    mapQuadraturePoints(
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
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
};
// class MassMoment

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
//TODO extern template class Plato::Elliptic::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
//TODO extern template class Plato::Elliptic::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
//TODO extern template class Plato::Elliptic::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
//TODO extern template class Plato::Elliptic::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATOANALYZE_2D
//TODO extern template class Plato::Elliptic::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
//TODO extern template class Plato::Elliptic::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
//TODO extern template class Plato::Elliptic::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
//TODO extern template class Plato::Elliptic::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATOANALYZE_3D
//TODO extern template class Plato::Elliptic::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
//TODO extern template class Plato::Elliptic::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
//TODO extern template class Plato::Elliptic::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
//TODO extern template class Plato::Elliptic::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif
