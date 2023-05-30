#pragma once

#include "BLAS1.hpp"
#include "FadTypes.hpp"
#include "Assembly.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoUtilities.hpp"
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
     * \param [in] aFuncName function name
     **********************************************************************************/
    template<typename EvaluationType>
    CriterionMassMoment<EvaluationType>::
    CriterionMassMoment(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aInputParams,
              std::string              aFuncName
    ) :
       FunctionBaseType(aSpatialDomain, aDataMap, aInputParams, aFuncName)
    /**************************************************************************/
    {
      this->initialize(aSpatialDomain,aInputParams);
    }

    /******************************************************************************//**
     * \brief Unit testing constructor
     * \param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    template<typename EvaluationType>
    CriterionMassMoment<EvaluationType>::
    CriterionMassMoment(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap& aDataMap
    ) :
      Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, "CriterionMassMoment")
    {}
    /**************************************************************************/

    /**************************************************************************/    
    template<typename EvaluationType>
    void CriterionMassMoment<EvaluationType>::
    initialize(
      const Plato::SpatialDomain   & aSpatialDomain,
            Teuchos::ParameterList & aInputParams
    )
    {
      this->parseMaterialDensity(aSpatialDomain,aInputParams);
      this->parseNormalizeCriterion(aSpatialDomain,aInputParams);
      this->computeTotalStructuralMass();
    }
    /**************************************************************************/

    template<typename EvaluationType>
    void CriterionMassMoment<EvaluationType>::
    parseMaterialDensity(
      const Plato::SpatialDomain   & aSpatialDomain,
            Teuchos::ParameterList & aInputParams
    )
    {
      auto tMaterialName = aSpatialDomain.getMaterialName();
      auto tMaterialModels = aInputParams.get<Teuchos::ParameterList>("Material Models");
      const Teuchos::ParameterList &tMaterialInputs = tMaterialModels.sublist(tMaterialName);
      if(tMaterialInputs.isSublist(tMaterialName))
      {
        auto tMsg = std::string("Parameter list for material with name '") 
          + tMaterialName + "' is not defined";
        ANALYZE_THROWERR(tMsg)
      }
      Teuchos::ParameterList::ConstIterator tMaterialItr = tMaterialInputs.begin();
      const std::string &tMaterialModelType = tMaterialInputs.name(tMaterialItr);
      const Teuchos::ParameterEntry &tMaterialEntry = tMaterialInputs.entry(tMaterialItr);
      if(!tMaterialEntry.isList())
      {
        auto tMsg = std::string("Parameter entry in Material Models block is invalid. Parameter ") 
          + tMaterialModelType + "' is not a parameter list.";
        ANALYZE_THROWERR(tMsg)
      }
      const Teuchos::ParameterList& tMaterialModelInputs = tMaterialInputs.sublist(tMaterialModelType);
      if( !tMaterialModelInputs.isParameter("Mass Density") )
      {
        auto tMsg = std::string("Parameter 'Mass Density' is not defined for material with name '")
          + tMaterialName + "' is not defined. Total structural mass cannot be computed.";
        ANALYZE_THROWERR(tMsg)
      }
      mMassDensity = tMaterialModelInputs.get<Plato::Scalar>("Mass Density");
      if(mMassDensity <= 0.)
      {
        auto tMsg = std::string("Unphysical 'Mass Density' parameter specified, 'Mass Density' is set to '") 
          + std::to_string(mMassDensity) + ".";
        ANALYZE_THROWERR(tMsg)
      }      
    }

    template<typename EvaluationType>
    void CriterionMassMoment<EvaluationType>::
    parseNormalizeCriterion(
      const Plato::SpatialDomain   & aSpatialDomain,
            Teuchos::ParameterList & aInputParams
    )
    {
      const std::string tFunctionName = this->getName();
      Teuchos::ParameterList &tProblemParams = aInputParams.sublist("Criteria").sublist(tFunctionName);
      mNormalizeCriterion = tProblemParams.get<bool>("Normalize Criterion",false);
    }

    /******************************************************************************//**
     * \brief Compute total structural mass, used for criterion normalization purposes
    **********************************************************************************/
    template<typename EvaluationType>
    void
    CriterionMassMoment<EvaluationType>::
    computeTotalStructuralMass()
    {
        if( !mNormalizeCriterion )
        { return; }

        auto tNumCells = mSpatialDomain.numCells();
        Plato::NodeCoordinate<mNumSpatialDims, mNumNodesPerCell> tCoordinates(mSpatialDomain.Mesh);
        Plato::ScalarArray3D tConfig("configuration", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>(tNumCells, tCoordinates, tConfig);

        Plato::ScalarVector tTotalStructuralMass("total mass", tNumCells);
        Plato::ScalarMultiVector tDensities("densities", tNumCells, mNumNodesPerCell);
        Kokkos::deep_copy(tDensities, 1.0);

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        auto tMassDensity = mMassDensity;
        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint  = tCubPoints(iGpOrdinal);
            auto tCubWeight = tCubWeights(iGpOrdinal);

            auto tJacobian = ElementType::jacobian(tCubPoint, tConfig, iCellOrdinal);

            auto tVolume = Plato::determinant(tJacobian);

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(iCellOrdinal, tBasisValues, tDensities);
            auto tLocalCellMass = tCellMass * tMassDensity * tVolume * tCubWeight;
            Kokkos::atomic_add(&tTotalStructuralMass(iCellOrdinal), tLocalCellMass);
        });
        Plato::blas1::local_sum(tTotalStructuralMass, mTotalStructuralMass);
    }

    /******************************************************************************//**
     * \brief set material density
     * \param [in] aMaterialDensity material density
     **********************************************************************************/
    template<typename EvaluationType>
    void CriterionMassMoment<EvaluationType>::
    setMaterialDensity(const Plato::Scalar aMaterialDensity)
    /**************************************************************************/
    {
      mMassDensity = aMaterialDensity;
    }

    /******************************************************************************//**
     * \brief set calculation type
     * \param [in] aCalculationType calculation type string
     **********************************************************************************/
    template<typename EvaluationType>
    void CriterionMassMoment<EvaluationType>::
    setCalculationType(const std::string & aCalculationType)
    /**************************************************************************/
    {
      mCalculationType = Plato::tolower(aCalculationType);
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
    CriterionMassMoment<EvaluationType>::
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
      if (mCalculationType == "mass")
        computeStructuralMass(aControl, aConfig, aResult, aTimeStep);
      else if (mCalculationType == "firstx")
        computeFirstMoment(aControl, aConfig, aResult, 0, aTimeStep);
      else if (mCalculationType == "firsty")
        computeFirstMoment(aControl, aConfig, aResult, 1, aTimeStep);
      else if (mCalculationType == "firstz")
        computeFirstMoment(aControl, aConfig, aResult, 2, aTimeStep);
      else if (mCalculationType == "secondxx")
        computeSecondMoment(aControl, aConfig, aResult, 0, 0, aTimeStep);
      else if (mCalculationType == "secondyy")
        computeSecondMoment(aControl, aConfig, aResult, 1, 1, aTimeStep);
      else if (mCalculationType == "secondzz")
        computeSecondMoment(aControl, aConfig, aResult, 2, 2, aTimeStep);
      else if (mCalculationType == "secondxy")
        computeSecondMoment(aControl, aConfig, aResult, 0, 1, aTimeStep);
      else if (mCalculationType == "secondxz")
        computeSecondMoment(aControl, aConfig, aResult, 0, 2, aTimeStep);
      else if (mCalculationType == "secondyz")
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
    CriterionMassMoment<EvaluationType>::
    computeStructuralMass(
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      auto tMassDensity = mMassDensity;
      auto tTotalStructuralMass = mTotalStructuralMass;

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();
      Kokkos::parallel_for("structural mass", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        auto tCubPoint  = tCubPoints(iGpOrdinal);
        auto tCubWeight = tCubWeights(iGpOrdinal);

        auto tJacobian = ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal);

        ResultScalarType tVolume = Plato::determinant(tJacobian);

        tVolume *= tCubWeight;

        auto tBasisValues = ElementType::basisValues(tCubPoint);
        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(iCellOrdinal, tBasisValues, aControl);

        ResultScalarType tNormalizedCellMass = (tCellMass / tTotalStructuralMass) * tMassDensity * tVolume;
        Kokkos::atomic_add(&aResult(iCellOrdinal), tNormalizedCellMass);

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
    CriterionMassMoment<EvaluationType>::
    computeFirstMoment(
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

      auto tMassDensity = mMassDensity;

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

        Kokkos::atomic_add(&aResult(iCellOrdinal), tCellMass * tMassDensity * tVolume *tMomentArm);
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
    CriterionMassMoment<EvaluationType>::
    computeSecondMoment(
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

      auto tMassDensity = mMassDensity;

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

        Kokkos::atomic_add(&aResult(iCellOrdinal), tCellMass * tMassDensity * tVolume * tSecondMoment);
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
    CriterionMassMoment<EvaluationType>::
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
