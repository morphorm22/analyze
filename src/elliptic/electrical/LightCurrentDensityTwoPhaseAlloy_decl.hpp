/*
 * LightCurrentDensityTwoPhaseAlloy_decl.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once


#include "MaterialModel.hpp"
#include "Plato_TopOptFunctors.hpp"

#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/electrical/CurrentDensityEvaluator.hpp"
#include "elliptic/electrical/LightGeneratedCurrentDensityConstant.hpp"

namespace Plato
{
    
template<typename EvaluationType>
class LightCurrentDensityTwoPhaseAlloy : 
  public Plato::CurrentDensityEvaluator<EvaluationType>
{
private:
    // set local element type
    using ElementType = typename EvaluationType::ElementType;
    static constexpr int mNumGaussPoints  = ElementType::mNumGaussPoints;
    static constexpr int mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr int mNumDofsPerNode  = ElementType::mNumDofsPerNode;
    static constexpr int mNumNodesPerCell = ElementType::mNumNodesPerCell;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    std::string mMaterialName = "";
    std::string mCurrentDensityName  = "";
    std::string mCurrentDensityModel = "Constant";

    Plato::Scalar mPenaltyExponent = 3.0; /*!< penalty exponent for material penalty model */
    Plato::Scalar mMinErsatzMaterialValue = 0.0; /*!< minimum value for the ersatz material density */
    std::vector<Plato::Scalar> mOutofPlaneThickness; /*!< list of out-of-plane material thickness */

    std::shared_ptr<Plato::LightGeneratedCurrentDensityConstant<EvaluationType>>
      mLightGeneratedCurrentDensity;

public:
  LightCurrentDensityTwoPhaseAlloy(
    const std::string            & aMaterialName,
    const std::string            & aCurrentDensityName,
          Teuchos::ParameterList & aParamList
  );

  ~LightCurrentDensityTwoPhaseAlloy();

  void 
  evaluate(
      const Plato::SpatialDomain                         & aSpatialDomain,
      const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
      const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
      const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
      const Plato::Scalar                                & aScale
  ) const;

  template<typename CurrentDensityType>
  void evaluate(
      const Plato::SpatialDomain                          & aSpatialDomain,
      const Plato::ScalarMultiVectorT<CurrentDensityType> & aCurrentDensity,
      const Plato::ScalarMultiVectorT<StateScalarType>    & aState,
      const Plato::ScalarMultiVectorT<ControlScalarType>  & aControl,
      const Plato::ScalarArray3DT<ConfigScalarType>       & aConfig,
      const Plato::ScalarMultiVectorT<ResultScalarType>   & aResult,
      const Plato::Scalar                                 & aScale
  ) const
  {
    // integration rule
    auto tCubPoints  = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    // out-of-plane thicknesses
    Plato::Scalar tThicknessOne = mOutofPlaneThickness.front();
    Plato::Scalar tThicknessTwo = mOutofPlaneThickness.back();
    // evaluate light-generated current density
    Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
    Kokkos::parallel_for("light-generated current density", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, mNumGaussPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
        auto tCubPoint = tCubPoints(iGpOrdinal);
        auto tDetJ = Plato::determinant(ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal));
        auto tBasisValues = ElementType::basisValues(tCubPoint);
        // material interpolation
        ControlScalarType tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal,aControl,tBasisValues);
        ControlScalarType tMaterialFraction = static_cast<Plato::Scalar>(1.0) - tDensity;
        ControlScalarType tMaterialPenalty = pow(tMaterialFraction, mPenaltyExponent);
        // out-of-plane thickness interpolation
        ControlScalarType tThicknessPenalty = pow(tDensity, mPenaltyExponent);
        ControlScalarType tThicknessInterpolation = tThicknessTwo + 
          ( ( tThicknessOne - tThicknessTwo) * tThicknessPenalty );
        auto tWeight = aScale * tCubWeights(iGpOrdinal) * tDetJ;
        for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < mNumNodesPerCell; tFieldOrdinal++)
        {
          ResultScalarType tCellResult = ( tBasisValues(tFieldOrdinal) * 
            (tMaterialPenalty * aCurrentDensity(iCellOrdinal,iGpOrdinal)) * tWeight ) / tThicknessInterpolation; 
          Kokkos::atomic_add( &aResult(iCellOrdinal,tFieldOrdinal*mNumDofsPerNode),tCellResult );
        }
    });
  }

  void 
  evaluate(
      const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
      const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
      const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
      const Plato::Scalar                                & aScale
  ) const;

  template<typename CurrentDensityType>
  void evaluate(
      const Plato::ScalarMultiVectorT<CurrentDensityType> & aCurrentDensity,
      const Plato::ScalarMultiVectorT<StateScalarType>    & aState,
      const Plato::ScalarMultiVectorT<ControlScalarType>  & aControl,
      const Plato::ScalarArray3DT<ConfigScalarType>       & aConfig,
      const Plato::ScalarMultiVectorT<ResultScalarType>   & aResult,
      const Plato::Scalar                                 & aScale
  ) const
  {
    // integration rule
    auto tCubPoints  = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    // out-of-plane thicknesses
    Plato::Scalar tThicknessOne = mOutofPlaneThickness.front();
    Plato::Scalar tThicknessTwo = mOutofPlaneThickness.back();
    // evaluate light-generated current density
    Plato::OrdinalType tNumCells = aResult.extent(0);
    Kokkos::parallel_for("light-generated current density", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, mNumGaussPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      auto tCubPoint = tCubPoints(iGpOrdinal);
      auto tBasisValues = ElementType::basisValues(tCubPoint);
      // material interpolation
      ControlScalarType tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal,aControl,tBasisValues);
      ControlScalarType tMaterialFraction = static_cast<Plato::Scalar>(1.0) - tDensity;
      ControlScalarType tMaterialPenalty = pow(tMaterialFraction, mPenaltyExponent);
      // out-of-plane thickness interpolation
      ControlScalarType tThicknessPenalty = pow(tDensity, mPenaltyExponent);
      ControlScalarType tThicknessInterpolation = tThicknessTwo + 
        ( ( tThicknessOne - tThicknessTwo) * tThicknessPenalty );
      // compute penalized current density at gauss points
      ResultScalarType tCellResult = 
      aResult(iCellOrdinal,iGpOrdinal) = ( aScale * tMaterialPenalty * 
        aCurrentDensity(iCellOrdinal,iGpOrdinal) ) / tThicknessInterpolation;
    });
  }

private:
    void 
    initialize(
      Teuchos::ParameterList &aParamList
    );

    void 
    setOutofPlaneThickness(
        Plato::MaterialModel<EvaluationType> & aMaterialModel
    );

    void buildCurrentDensityModel(
      Teuchos::ParameterList &aParamList
    );
};

} 
// namespace  Plato
