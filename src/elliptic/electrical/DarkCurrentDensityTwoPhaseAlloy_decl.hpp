/*
 * DarkCurrentDensityTwoPhaseAlloy_decl.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once


#include "MaterialModel.hpp"
#include "Plato_TopOptFunctors.hpp"

#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/electrical/CurrentDensitySourceEvaluator.hpp"
#include "elliptic/electrical/DarkCurrentDensityQuadratic.hpp"

namespace Plato
{

/// @class DarkCurrentDensityTwoPhaseAlloy
/// @brief evaluator of dark current density models
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class DarkCurrentDensityTwoPhaseAlloy : 
  public Plato::CurrentDensitySourceEvaluator<EvaluationType>
{
private:
    /// @brief topological element type
    using ElementType = typename EvaluationType::ElementType;
    /// @brief number of Gauss integration points
    static constexpr int mNumGaussPoints  = ElementType::mNumGaussPoints;
    /// @brief number of spatial dimensions
    static constexpr int mNumSpatialDims  = ElementType::mNumSpatialDims;
    /// @brief number of degrees of freedom per node
    static constexpr int mNumDofsPerNode  = ElementType::mNumDofsPerNode;
    /// @brief number of nodes per cell
    static constexpr int mNumNodesPerCell = ElementType::mNumNodesPerCell;
    /// @brief scalar types associated with the evaluation type
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    /// @brief name of input material parameter list
    std::string mMaterialName = "";
    /// @brief name of input current density parameter list
    std::string mCurrentDensityName  = "";
    /// @brief input current density model
    std::string mCurrentDensityModel = "Quadratic";
    /// @brief penalty exponent for material penalty model
    Plato::Scalar mPenaltyExponent = 3.0;
    /// @brief minimum value for the ersatz material density
    Plato::Scalar mMinErsatzMaterialValue = 0.0; 
    /// @brief list of out-of-plane material thickness
    std::vector<Plato::Scalar> mOutofPlaneThickness;
    /// @brief shared pointer to current density model
    std::shared_ptr<Plato::DarkCurrentDensityQuadratic<EvaluationType>> mDarkCurrentDensity;

public:
  /// @brief class constructor
  /// @param aMaterialName       name of input material parameter list
  /// @param aCurrentDensityName name of input current density parameter list
  /// @param aParamList          input problem parameters
  DarkCurrentDensityTwoPhaseAlloy(
    const std::string            & aMaterialName,
    const std::string            & aCurrentDensityName,
          Teuchos::ParameterList & aParamList
  );

  /// @brief class destructor
  ~DarkCurrentDensityTwoPhaseAlloy();

  /// @fn evaluate
  /// @brief evaluate dark current density
  /// @param [in]     aSpatialDomain contains meshed model information
  /// @param [in]     aState         2D state workset
  /// @param [in]     aControl       2D control workset
  /// @param [in]     aConfig        3D configuration workset
  /// @param [in,out] aResult        2D result workset
  /// @param [in]     aScale         scalar
  void
  evaluate(
      const Plato::SpatialDomain                         & aSpatialDomain,
      const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
      const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
      const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
      const Plato::Scalar                                & aScale
  ) const;

  /// @fn evaluate
  /// @brief evaluate dark current density
  /// @param [in]     aSpatialDomain  contains meshed model information
  /// @param [in]     aCurrentDensity 2D current density workset
  /// @param [in]     aState          2D state workset
  /// @param [in]     aControl        2D control workset
  /// @param [in]     aConfig         3D configuration workset
  /// @param [in,out] aResult         2D result workset
  /// @param [in]     aScale          scalar
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
    // evaluate dark current density
    Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
    Kokkos::parallel_for("evaluate dark current density", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, mNumGaussPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
        // get basis functions and weights for this integration point
        auto tCubPoint = tCubPoints(iGpOrdinal);
        auto tDetJ = Plato::determinant(ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal));
        auto tBasisValues = ElementType::basisValues(tCubPoint);
        // out-of-plane thickness interpolation
        ControlScalarType tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal,aControl,tBasisValues);
        ControlScalarType tThicknessPenalty = pow(tDensity, mPenaltyExponent);
        ControlScalarType tThicknessInterpolation = tThicknessTwo + 
          ( ( tThicknessOne - tThicknessTwo) * tThicknessPenalty );
        auto tWeight = aScale * tCubWeights(iGpOrdinal) * tDetJ;
        for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < mNumNodesPerCell; tFieldOrdinal++)
        {
          ResultScalarType tCellResult = ( tBasisValues(tFieldOrdinal) * 
            aCurrentDensity(iCellOrdinal,iGpOrdinal) * tWeight ) / tThicknessInterpolation;
          Kokkos::atomic_add( &aResult(iCellOrdinal,tFieldOrdinal), tCellResult );
        }
    });
  }

  /// @fn evaluate
  /// @brief evaluate dark current density
  /// @param [in]     aState   2D state workset
  /// @param [in]     aControl 2D control workset
  /// @param [in]     aConfig  3D configuration workset
  /// @param [in,out] aResult  2D result workset
  /// @param [in]     aScale   scalar
  void 
  evaluate(
      const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
      const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
      const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
      const Plato::Scalar                                & aScale
  ) 
  const;

  /// @fn evaluate
  /// @brief evaluate dark current density
  /// @param [in]     aCurrentDensity 2D current density workset
  /// @param [in]     aState          2D state workset
  /// @param [in]     aControl        2D control workset
  /// @param [in]     aConfig         3D configuration workset
  /// @param [in,out] aResult         2D result workset
  /// @param [in]     aScale          scalar
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
    // evaluate dark current density
    Plato::OrdinalType tNumCells = aResult.extent(0);
    Kokkos::parallel_for("evaluate dark current density", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, mNumGaussPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // get basis functions and weights for this integration point
      auto tCubPoint = tCubPoints(iGpOrdinal);
      auto tBasisValues = ElementType::basisValues(tCubPoint);
      // out-of-plane thickness interpolation
      ControlScalarType tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal,aControl,tBasisValues);
      ControlScalarType tThicknessPenalty = pow(tDensity, mPenaltyExponent);
      ControlScalarType tThicknessInterpolation = tThicknessTwo + 
        ( ( tThicknessOne - tThicknessTwo) * tThicknessPenalty );
      aResult(iCellOrdinal,iGpOrdinal) = 
        ( aScale * aCurrentDensity(iCellOrdinal,iGpOrdinal) ) / tThicknessInterpolation;
    });
  }

private:
    /// @fn initialize
    /// @brief initialize dark current density evaluator class
    /// @param [in] aParamList input problem parameters
    void 
    initialize(
        Teuchos::ParameterList &aParamList
    );
    
    /// @fn setOutofPlaneThickness
    /// @brief set out-of-plane material thickness for two-phase alloy material
    /// @param aMaterialModel material constitutive model evaluator
    void 
    setOutofPlaneThickness(
        Plato::MaterialModel<EvaluationType> & aMaterialModel
    );

    /// @fn buildCurrentDensityModel
    /// @brief build light-generated current density model
    /// @param [in] aParamList input problem parameters
    void buildCurrentDensityModel(
      Teuchos::ParameterList &aParamList
    );
};

}
// namespace Plato