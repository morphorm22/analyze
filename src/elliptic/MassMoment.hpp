#pragma once

#include "Simplex.hpp"
//#include "SimplexFadTypes.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "SimplexMechanics.hpp"
#include "WorksetBase.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
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
class MassMoment : public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
                   public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int mSpaceDim = EvaluationType::SpatialDim; /*!< space dimension */
    static constexpr Plato::OrdinalType mNumVoigtTerms = Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms; /*!< number of Voigt terms */
    static constexpr Plato::OrdinalType mNumNodesPerCell = Plato::SimplexMechanics<mSpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell/element */
    
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain; /*!< domain object */
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap; /*!< data map object */

    using StateScalarType   = typename EvaluationType::StateScalarType;  /*!< state variables automatic differentiation type */
    using ControlScalarType = typename EvaluationType::ControlScalarType; /*!< control variables automatic differentiation type */
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultScalarType  = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

    Plato::Scalar mCellMaterialDensity; /*!< material density */

    std::string mCalculationType; /*!< calculation type = Mass, CGx, CGy, CGz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz */

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
       Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, "MassMoment"),
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
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
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

      Plato::ComputeCellVolume<mSpaceDim> tComputeCellVolume;

      auto tCellMaterialDensity = mCellMaterialDensity;

      Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
      auto tCubWeight = tCubatureRule.getCubWeight();
      auto tBasisFunc = tCubatureRule.getBasisFunctions();

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
      {
        ConfigScalarType tCellVolume;
        tComputeCellVolume(aCellOrdinal, aConfig, tCellVolume);
        tCellVolume *= tCubWeight;

        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(aCellOrdinal, tBasisFunc, aControl);

        aResult(aCellOrdinal) = ( tCellMass * tCellMaterialDensity * tCellVolume );

      }, "mass calculation");
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
      assert(aComponent < mSpaceDim);

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeCellVolume<mSpaceDim> tComputeCellVolume;

      auto tCellMaterialDensity = mCellMaterialDensity;

      Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
      auto tCubPoint  = tCubatureRule.getCubPointsCoords();
      auto tCubWeight = tCubatureRule.getCubWeight();
      auto tBasisFunc = tCubatureRule.getBasisFunctions();
      auto tNumPoints = tCubatureRule.getNumCubPoints();

      Plato::ScalarMultiVectorT<ConfigScalarType> tMappedPoints("mapped quad points", tNumCells, mSpaceDim);
      mapQuadraturePoint(tCubPoint, aConfig, tMappedPoints);

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & tCellOrdinal)
      {
        ConfigScalarType tCellVolume;
        tComputeCellVolume(tCellOrdinal, aConfig, tCellVolume);
        tCellVolume *= tCubWeight;

        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(tCellOrdinal, tBasisFunc, aControl);

        ConfigScalarType tMomentArm = tMappedPoints(tCellOrdinal, aComponent);

        aResult(tCellOrdinal) = ( tCellMass * tCellMaterialDensity * tCellVolume * tMomentArm );

      }, "first moment calculation");
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
      assert(aComponent1 < mSpaceDim);
      assert(aComponent2 < mSpaceDim);

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeCellVolume<mSpaceDim> tComputeCellVolume;

      auto tCellMaterialDensity = mCellMaterialDensity;

      Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
      auto tCubPoint  = tCubatureRule.getCubPointsCoords();
      auto tCubWeight = tCubatureRule.getCubWeight();
      auto tBasisFunc = tCubatureRule.getBasisFunctions();
      auto tNumPoints = tCubatureRule.getNumCubPoints();

      Plato::ScalarMultiVectorT<ConfigScalarType> tMappedPoints("mapped quad points", tNumCells, mSpaceDim);
      mapQuadraturePoint(tCubPoint, aConfig, tMappedPoints);

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & tCellOrdinal)
      {
        ConfigScalarType tCellVolume;
        tComputeCellVolume(tCellOrdinal, aConfig, tCellVolume);
        tCellVolume *= tCubWeight;

        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(tCellOrdinal, tBasisFunc, aControl);

        ConfigScalarType tMomentArm1 = tMappedPoints(tCellOrdinal, aComponent1);
        ConfigScalarType tMomentArm2 = tMappedPoints(tCellOrdinal, aComponent2);
        ConfigScalarType tSecondMoment  = tMomentArm1 * tMomentArm2;

        aResult(tCellOrdinal) = ( tCellMass * tCellMaterialDensity * tCellVolume * tSecondMoment );

      }, "second moment calculation");
    }

    /******************************************************************************//**
     * \brief Map quadrature points to physical domain
     * \param [in] aRefPoint incoming quadrature points
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aMappedPoints points mapped to physical domain
    **********************************************************************************/
    void
    mapQuadraturePoint(
        const Plato::ScalarVector                          & aRefPoint,
        const Plato::ScalarArray3DT     <ConfigScalarType> & aConfig,
              Plato::ScalarMultiVectorT <ConfigScalarType> & aMappedPoints
    ) const
    /******************************************************************************/
    {
      Plato::OrdinalType tNumCells  = mSpatialDomain.numCells();

      Kokkos::deep_copy(aMappedPoints, static_cast<ConfigScalarType>(0.0));

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(Plato::OrdinalType tCellOrdinal) {
        Plato::OrdinalType tNodeOrdinal;
        Plato::Scalar tFinalNodeValue = 1.0;
        for (tNodeOrdinal = 0; tNodeOrdinal < mSpaceDim; ++tNodeOrdinal)
        {
          Plato::Scalar tNodeValue = aRefPoint(tNodeOrdinal);
          tFinalNodeValue -= tNodeValue;
          for (Plato::OrdinalType tDim = 0; tDim < mSpaceDim; ++tDim)
          {
            aMappedPoints(tCellOrdinal,tDim) += tNodeValue * aConfig(tCellOrdinal,tNodeOrdinal,tDim);
          }
        }
        tNodeOrdinal = mSpaceDim;
        for (Plato::OrdinalType tDim = 0; tDim < mSpaceDim; ++tDim)
        {
          aMappedPoints(tCellOrdinal,tDim) += tFinalNodeValue * aConfig(tCellOrdinal,tNodeOrdinal,tDim);
        }
      }, "map single quadrature point to physical domain");
    }
};
// class MassMoment

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::Elliptic::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::Elliptic::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::Elliptic::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::Elliptic::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Elliptic::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::Elliptic::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::Elliptic::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::Elliptic::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Elliptic::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::Elliptic::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::Elliptic::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::Elliptic::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif
