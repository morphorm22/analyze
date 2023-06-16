/*
 * ThermoelastostaticTotalLagrangianTests.cpp
 *
 *  Created on: June 14, 2023
 */

// trilinos includes
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

// unit test includes
#include "util/PlatoTestHelpers.hpp"

// analyze includes
#include "Tri3.hpp"
#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "SpatialModel.hpp"
#include "GradientMatrix.hpp"
#include "PlatoMathTypes.hpp"
#include "base/WorksetBase.hpp"
#include "base/ResidualBase.hpp"
#include "InterpolateFromNodal.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "ThermalConductivityMaterial.hpp"

#include "elliptic/mechanical/nonlinear/StateGradient.hpp"
#include "elliptic/mechanical/nonlinear/DeformationGradient.hpp"
#include "elliptic/mechanical/nonlinear/FactoryStressEvaluator.hpp"

namespace Plato
{

/// @class ThermoElasticElement
/// @brief base class for thermo-elastic element
/// @tparam TopoElementTypeT topological element type
/// @tparam NumControls      number of control degree of freedom per node
template<typename TopoElementTypeT, Plato::OrdinalType NumControls = 1>
class ThermoElasticElement : public TopoElementTypeT, public ElementBase<TopoElementTypeT>
{
public:
  /// @brief number of nodes per cell
  using TopoElementTypeT::mNumNodesPerCell;
  /// @brief number of spatial dimensions
  using TopoElementTypeT::mNumSpatialDims;
  /// @brief topological element type
  using TopoElementType = TopoElementTypeT;
  /// @brief number of stress-strain components
  static constexpr Plato::OrdinalType mNumVoigtTerms = (mNumSpatialDims == 3) ? 6 :
                                                       ((mNumSpatialDims == 2) ? 3 :
                                                       (((mNumSpatialDims == 1) ? 1 : 0)));
  /// @brief number of displacement degrees of freedom per node
  static constexpr Plato::OrdinalType mNumDofsPerNode = mNumSpatialDims;
  /// @brief number of displacement degrees of freedom per cell
  static constexpr Plato::OrdinalType mNumDofsPerCell = mNumDofsPerNode * mNumNodesPerCell;
  /// @brief number of control degrees of freedom per node
  static constexpr Plato::OrdinalType mNumControl = NumControls;
  /// @brief number of temperature degrees of freedom per node
  static constexpr Plato::OrdinalType mNumNodeStatePerNode = 1;
  /// @brief number of local state degrees of freedom per cell
  static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
};

template<typename EvaluationType>
class ThermalDeformationGradient
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief evaluation scalar types for function range and domain
  using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;

public:
  /// @brief reference temperature 
  Plato::Scalar mReferenceTemperature;
  /// @brief coefficient of thermal expansion
  Plato::Scalar mThermalExpansivity;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName name of input material parameter list
  /// @param [in] aParamList    input problem parameters
  ThermalDeformationGradient(
    const std::string            & aMaterialName,
          Teuchos::ParameterList & aParamList
  )
  {
    Plato::ThermalConductionModelFactory<EvaluationType> tMaterialFactory(aParamList);
    auto tMaterialModel = tMaterialFactory.create(aMaterialName);
    if( !tMaterialModel->scalarConstantExists("Thermal Expansivity") ){
      auto tMsg = std::string("Material parameter ('Thermal Expansivity') is not defined, thermal deformation ") 
        + "gradient cannot be computed";
      ANALYZE_THROWERR(tMsg)
    }
    mThermalExpansivity = tMaterialModel->getScalarConstant("Thermal Expansivity");

    if( !tMaterialModel->scalarConstantExists("Reference Temperature") ){
      auto tMsg = std::string("Material parameter ('Reference Temperature') is not defined, thermal deformation ") 
        + "gradient cannot be computed";
      ANALYZE_THROWERR(tMsg)
    }
    mReferenceTemperature = tMaterialModel->getScalarConstant("Reference Temperature");
  }

  /// @fn operator()()
  /// @brief compute thermal deformation gradient
  /// @param [in]     aTemp    temperature
  /// @param [in,out] aDefGrad deformation gradient
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const NodeStateScalarType                                                & aTemp,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,NodeStateScalarType> & aDefGrad
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      aDefGrad(tDimI,tDimI) += 1.0 + ( mThermalExpansivity * (aTemp - mReferenceTemperature) );
    }
  }
};

/// @brief compute thermo-elastic deformation gradient via a multiplicative decomposition of the form:
/// \f[ 
///   F_{ij}=F^{\theta}_{ik}F^{u}_{kj},
/// \f]
/// where \f$F_{ij}\f$ is the deformation gradient, \f$F^{\theta}_{ij}\f$ is the thermal deformation gradient,
/// and \f$F^{u}_{ij}\f$ is the mechanical deformation gradient
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class ThermoElasticDeformationGradient
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types associated with input evaluation template type
  using StateScalarType     = typename EvaluationType::StateScalarType;
  using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
  using ResultScalarType    = typename EvaluationType::ResultScalarType;
  using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
  using StrainScalarType    = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;

public:
  /// @fn operator()()
  /// @brief compute thermo-elastic deformation gradient
  /// @param [in]     aThermalGrad    thermal deformation gradient
  /// @param [in]     aMechanicalGrad mechanical deformation gradient
  /// @param [in,out] aDefGrad thermo-elastic deformation gradient
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,NodeStateScalarType> & aThermalGrad,
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>    & aMechanicalGrad,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType>    & aDefGrad
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        for(Plato::OrdinalType tDimK = 0; tDimK < mNumSpatialDims; tDimK++){
          aDefGrad(tDimI,tDimJ) += aThermalGrad(tDimI,tDimK) * aMechanicalGrad(tDimK,tDimJ);
        }
      }
    }
  }
};

/// @class KineticPullBackOperation
/// @brief apply pull back operation to second order kinetic tensor:
/// \f[ 
///   S_{ij}=F^{-1}_{ik}\sigma_{kl}F^{-1}_{jl} J or \mathbf{S}=\mathbf{F}^{-1}\mathbf{\sigma}\mathbf{F}^{-T} J, 
/// \f]
/// where \f$J=\det(F)\f$, \f$S\f$ and \f$\sigma\f$ are generic second order kinetic tensors (e.g., stress), 
/// \f$F\f$ is a generic deformation gradient 
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class KineticPullBackOperation
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;

public:
  /// @brief class constructor
  KineticPullBackOperation(){}
  /// @brief class destructor
  ~KineticPullBackOperation(){}

  /// @fn operator()
  /// @brief apply pull back operation to second order kinetic tensor
  /// @tparam DefGradScalarType scalar type for deformation gradient
  /// @tparam KineticScalarType scalar type for second order kinetic tensor
  /// @tparam OutputScalarType  scalar type for output second order kinetic tensor
  /// @param [in]     aDefGrad          deformation gradient
  /// @param [in]     aInKineticTensor  input second order kinetic tensor
  /// @param [in,out] aOutKineticTensor output second order kinetic tensor
  template<typename DefGradScalarType, 
           typename KineticScalarType, 
           typename OutputScalarType>
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,DefGradScalarType> & aDefGrad,
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,KineticScalarType> & aInKineticTensor,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,OutputScalarType>  & aOutKineticTensor
  ) const
  {
    // compute deformation gradient inverse
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,DefGradScalarType> aInvDefGrad = Plato::invert(aDefGrad);
    // compute determinant of deformation gradient 
    DefGradScalarType tDeterminantDefGrad = Plato::determinant(aDefGrad);
    // apply pull-back operation to kinetic (stress) tensor
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        for(Plato::OrdinalType tDimK = 0; tDimK < mNumSpatialDims; tDimK++){
          for(Plato::OrdinalType tDimL = 0; tDimL < mNumSpatialDims; tDimL++){
            aOutKineticTensor(tDimI,tDimJ) += tDeterminantDefGrad * 
              aInvDefGrad(tDimI,tDimK) * aInKineticTensor(tDimK,tDimL) * aInvDefGrad(tDimJ,tDimL);
          }
        }
      }
    }
  }
};

/// @class NominalStress
/// @brief compute nominal (first piola-kirchhoff) stress tensor:
/// \f[ 
///   P_{ij}=S_{ik}F_{jk} or \mathbf{P}=\mathbf{S}\mathbf{F}^{T}, 
/// \f]
/// where \f$\mathbf{P}\f$ is the first piola-kirchhoff stress, \f$\mathbf{S}\f$ is the second 
/// piola-kirchhoff stress, and \f$\mathbf{F}\f$ is the deformation gradient 
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class NominalStress
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;

public:
  /// @brief class constructor
  NominalStress(){}
  /// @brief class destructor
  ~NominalStress(){}

  /// @fn operator()
  /// @brief compute nominal stress
  /// @tparam DefGradScalarType scalar type for deformation gradient
  /// @tparam KineticScalarType scalar type for second piola-kirchhoff stress
  /// @tparam OutputScalarType  scalar type for first piola-kirchhoff stress (nominal stress)
  /// @param [in]     aDefGrad deformation gradient
  /// @param [in]     a2PKS    second piola-kirchhoff stress
  /// @param [in,out] a1PKS    first piola-kirchhoff stress
  template<typename DefGradScalarType, 
           typename KineticScalarType, 
           typename OutputScalarType>
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,DefGradScalarType> & aDefGrad,
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,KineticScalarType> & a2PKS,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,OutputScalarType>  & a1PKS
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        for(Plato::OrdinalType tDimK = 0; tDimK < mNumSpatialDims; tDimK++){
          a1PKS(tDimI,tDimJ) += a2PKS(tDimI,tDimK) * aDefGrad(tDimJ,tDimK);
        }
      }
    }
  }
};

namespace Elliptic
{

template<Plato::OrdinalType NumSpatialDims, typename ScalarType>
KOKKOS_INLINE_FUNCTION
void 
get_cell_2PKS(
  const Plato::OrdinalType                                      & aCellOrdinal,
  const Plato::OrdinalType                                      & aIntegrationPointOrdinal,
  const Plato::ScalarArray4DT<ScalarType>                       & aInSecondPiolaKirchhoffStress,
        Plato::Matrix<NumSpatialDims,NumSpatialDims,ScalarType> & aOutSecondPiolaKirchhoffStress
)
{
  for(Plato::OrdinalType tDimI = 0; tDimI < NumSpatialDims; tDimI++){
    for(Plato::OrdinalType tDimJ = 0; tDimJ < NumSpatialDims; tDimJ++){
      aOutSecondPiolaKirchhoffStress(tDimI,tDimJ) = 
        aInSecondPiolaKirchhoffStress(aCellOrdinal,aIntegrationPointOrdinal,tDimI,tDimJ);
    }
  }
}

template<typename EvaluationType>
class ResidualThermoElastoStaticTotalLagrangian : public Plato::ResidualBase
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief number of displacement degrees of freedom per node
  static constexpr auto mNumDofsPerNode = ElementType::mNumDofsPerNode;
  /// @brief number of displacement degrees of freedom per cell
  static constexpr auto mNumDofsPerCell = ElementType::mNumDofsPerCell;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;
  /// @brief number of integration points per cell
  static constexpr auto mNumGaussPoints = ElementType::mNumGaussPoints;
  /// @brief number of temperature degrees of freedom per node
  static constexpr auto mNumNodeStatePerNode = ElementType::mNumNodeStatePerNode;
  /// @brief local typename for base class
  using FunctionBaseType = Plato::ResidualBase;
  /// @brief contains mesh and model information
  using FunctionBaseType::mSpatialDomain;
  /// @brief output database
  using FunctionBaseType::mDataMap;
  /// @brief contains degrees of freedom names 
  using FunctionBaseType::mDofNames;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType     = typename EvaluationType::StateScalarType;
  using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
  using ResultScalarType    = typename EvaluationType::ResultScalarType;
  using ControlScalarType   = typename EvaluationType::ControlScalarType;
  using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
  using StrainScalarType    = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief stress evaluator
  std::shared_ptr<Plato::StressEvaluator<EvaluationType>> mStressEvaluator;
  /// @brief natural boundary conditions evaluator
  std::shared_ptr<Plato::NaturalBCs<ElementType>> mNaturalBCs;
  /// @brief body loads evaluator
  std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;
  /// @brief output plot table, contains requested output quantities of interests
  std::vector<std::string> mPlotTable;
  /// @brief input problem parameters
  const Teuchos::ParameterList & mParamList;

public:
  ResidualThermoElastoStaticTotalLagrangian(
    const Plato::SpatialDomain   & aSpatialDomain,
    const Teuchos::ParameterList & aParamList,
          Plato::DataMap         & aDataMap
  ) : 
    FunctionBaseType(aSpatialDomain, aDataMap),
    mStressEvaluator(nullptr),
    mNaturalBCs     (nullptr),
    mBodyLoads      (nullptr),
    mParamList(aParamList)
  {}

  ~ResidualThermoElastoStaticTotalLagrangian(){}

  Plato::Solutions
  getSolutionStateOutputData(
    const Plato::Solutions & aSolutions
  ) const
  {
    return aSolutions;
  }

  void
  evaluate(
    Plato::WorkSets & aWorkSets,
    Plato::Scalar     aCycle = 0.0
  ) const
  {
    // unpack worksets
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    Plato::ScalarMultiVectorT<ControlScalarType> tControlWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ControlScalarType>>(aWorkSets.get("controls"));
    Plato::ScalarMultiVectorT<StateScalarType> tDispWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
    Plato::ScalarMultiVectorT<NodeStateScalarType> tTempWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<NodeStateScalarType>>(aWorkSets.get("node states"));
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    // evaluate mechanical stresses
    Plato::OrdinalType tNumCells = mSpatialDomain.numCells();
    Plato::OrdinalType tNumGaussPoints = ElementType::mNumGaussPoints;
    Plato::ScalarArray4DT<ResultScalarType> t2PKS(
      "2nd Piola-Kirchhoff Stress",tNumCells,tNumGaussPoints,mNumSpatialDims,mNumSpatialDims
    );
    mStressEvaluator->evaluate(aWorkSets,t2PKS,aCycle);
    // create local functors
    Plato::StateGradient<EvaluationType> tComputeDispGradient;
    Plato::NominalStress<EvaluationType> tComputeNominalStress;
    Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
    Plato::KineticPullBackOperation<EvaluationType> tApplyKineticPullBackOperation;
    Plato::DeformationGradient<EvaluationType> tComputeMechanicalDeformationGradient;
    Plato::ThermoElasticDeformationGradient<EvaluationType> tComputeThermoElasticDeformationGradient;
    Plato::ThermalDeformationGradient<EvaluationType> tComputeThermalDeformationGradient(
      mSpatialDomain.getMaterialName(),mParamList
    );
    Plato::InterpolateFromNodal<ElementType,mNumNodeStatePerNode> tInterpolateFromNodal;
    // get integration rule data
    auto tCubPoints  = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    // evaluate internal forces
    Kokkos::parallel_for("compute internal forces", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,mNumGaussPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient of interpolation functions
      ConfigScalarType tVolume(0.0);
      auto tCubPoint = tCubPoints(iGpOrdinal);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient(ConfigScalarType(0.));
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute displacement gradient
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> tDispGradient(StrainScalarType(0.));
      tComputeDispGradient(iCellOrdinal,tDispWS,tGradient,tDispGradient);
      // compute mechanical deformation gradient 
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> tMechanicalDefGradient(StrainScalarType(0.));
      tComputeMechanicalDeformationGradient(tDispGradient,tMechanicalDefGradient);
      // interpolate temperature field from nodes to integration point
      auto tBasisValues = ElementType::basisValues(tCubPoint);
      NodeStateScalarType tTemperature = tInterpolateFromNodal(iCellOrdinal,tBasisValues,tTempWS);
      // compute thermal deformation gradient 
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,NodeStateScalarType> tThermalDefGradient(StrainScalarType(0.));
      tComputeThermalDeformationGradient(tTemperature,tThermalDefGradient);
      // compute multiplicative decomposition of the thermo-elastic deformation gradient 
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tThermoElasticDefGradient(ResultScalarType(0.));
      tComputeThermoElasticDeformationGradient(tThermalDefGradient,tMechanicalDefGradient,tThermoElasticDefGradient);
      // pull back second Piola-Kirchhoff stress from deformed to undeformed configuration
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tDefConfig2PKS(ResultScalarType(0.));
      Plato::Elliptic::get_cell_2PKS<mNumSpatialDims>(iCellOrdinal,iGpOrdinal,t2PKS,tDefConfig2PKS);
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tUnDefConfig2PKS(ResultScalarType(0.));
      tApplyKineticPullBackOperation(tThermalDefGradient,tDefConfig2PKS,tUnDefConfig2PKS);
      // compute nominal stress
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ResultScalarType> tNominalStress(ResultScalarType(0.));
      tComputeNominalStress(tThermoElasticDefGradient,tUnDefConfig2PKS,tNominalStress);
      // apply integration point weight to element volume
      tVolume *= tCubWeights(iGpOrdinal);
      // apply divergence operator to nominal stress tensor
      for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++){
        for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
          Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumSpatialDims + tDimI;
          for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
            ResultScalarType tVal = tNominalStress(tDimI,tDimJ) * tGradient(tNodeIndex,tDimJ) * tVolume;
            Kokkos::atomic_add( &tResultWS(iCellOrdinal,tLocalOrdinal),tVal );
          }
        }
      }
    });
    // evaluate body forces
    if( mBodyLoads != nullptr )
    {
      mBodyLoads->get( mSpatialDomain,tDispWS,tControlWS,tConfigWS,tResultWS,-1.0 );
    }
  }

  void
  evaluateBoundary(
    const Plato::SpatialModel & aSpatialModel,
          Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0
  ) const
  {
    // unpack worksets
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    Plato::ScalarMultiVectorT<ControlScalarType> tControlWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ControlScalarType>>(aWorkSets.get("controls"));
    Plato::ScalarMultiVectorT<StateScalarType> tDispWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("state"));
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    // evaluate boundary forces
    if( mNaturalBCs != nullptr )
    {
      mNaturalBCs->get(aSpatialModel, tDispWS, tControlWS, tConfigWS, tResultWS, -1.0 );
    }
  }

private:
  void 
  initialize(
    Teuchos::ParameterList & aParamList
  )
  {
    // create material model and get stiffness
    //
    Plato::FactoryStressEvaluator<EvaluationType> tStressEvaluatorFactory(mSpatialDomain.getMaterialName());
    mStressEvaluator = tStressEvaluatorFactory.create(aParamList,mSpatialDomain,mDataMap);
    // parse body loads
    // 
    if(aParamList.isSublist("Body Loads"))
    {
      mBodyLoads = 
        std::make_shared<Plato::BodyLoads<EvaluationType, ElementType>>(aParamList.sublist("Body Loads"));
    }
    // parse boundary Conditions
    // 
    if(aParamList.isSublist("Natural Boundary Conditions"))
    {
      mNaturalBCs = 
        std::make_shared<Plato::NaturalBCs<ElementType>>(aParamList.sublist("Natural Boundary Conditions"));
    }
    // parse plot table
    //
    auto tResidualParams = aParamList.sublist("Output");
    if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
    {
      mPlotTable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
    }
  }

};

} // namespace Elliptic

} // namespace Plato

namespace ThermoelastostaticTotalLagrangianTests
{

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, tComputeThermalDefGrad )
{
  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                             \n"
    "<ParameterList name='Material Models'>                                         \n"
      "<ParameterList name='Unobtainium'>                                           \n"
        "<ParameterList name='Thermal Conduction'>                                  \n"
          "<Parameter  name='Thermal Expansivity'   type='double' value='0.5'/>     \n"
          "<Parameter  name='Reference Temperature' type='double' value='1.0'/>     \n"
          "<Parameter  name='Thermal Conductivity'  type='double' value='2.0'/>     \n"
        "</ParameterList>                                                           \n"
      "</ParameterList>                                                             \n"
    "</ParameterList>                                                               \n"
  "</ParameterList>                                                                 \n"
  ); 
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using NodeStateT  = typename Residual::NodeStateScalarType;
  // create temperature workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumNodes = tMesh->NumNodes();
  Plato::ScalarVector tTemp("Temps", tNumNodes);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill node state",
    Kokkos::RangePolicy<>(0, tNumNodes), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<NodeStateT> tTempWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tTemp, tTempWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute thermal deformation gradient
  Plato::ThermalDeformationGradient<Residual> tComputeThermalDeformationGradient("Unobtainium",*tParamList);
  Plato::InterpolateFromNodal<ElementType,ElementType::mNumNodeStatePerNode> tInterpolateFromNodal;
  Kokkos::parallel_for("compute thermal deformation gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    // interpolate temperature from nodes to integration points
    NodeStateT tTemperature = tInterpolateFromNodal(iCellOrdinal,tBasisValues,tTempWS);
    // compute thermal deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,NodeStateT> tTempDefGrad;
    tComputeThermalDeformationGradient(tTemperature,tTempDefGrad);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tTempDefGrad(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.5166666666666666,0.,0.,0.5166666666666666}, {0.5166666666666666,0.,0.,0.5166666666666666} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, tComputeThermoElasticDefGrad )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using VecStateT   = typename Residual::StateScalarType;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using NodeStateT  = typename Residual::NodeStateScalarType;
  using StrainT     = typename Plato::fad_type_t<ElementType,VecStateT,ConfigT>;
  // create thermal gradient workset
  std::vector<std::vector<Plato::Scalar>> tTempDefGrad = 
    { {0.5166666666666666,0.,0.,0.5166666666666666}, {0.5166666666666666,0.,0.,0.5166666666666666} };
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  Plato::ScalarArray3DT<NodeStateT> tTempDefGradWS("thermal deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostTempDefGradWS = Kokkos::create_mirror(tTempDefGradWS);
  Kokkos::deep_copy(tHostTempDefGradWS, tTempDefGradWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostTempDefGradWS(tCell,tDimI,tDimJ) = tTempDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tTempDefGradWS,tHostTempDefGradWS);
  // create mechanical gradient workset
  std::vector<std::vector<Plato::Scalar>> tMechDefGrad = 
    { {1.4,0.2,0.4,1.2}, {1.4,0.2,0.4,1.2} };
  Plato::ScalarArray3DT<StrainT> tMechDefGradWS("mechanical deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostMechDefGradWS = Kokkos::create_mirror(tMechDefGradWS);
  Kokkos::deep_copy(tHostMechDefGradWS,tMechDefGradWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostMechDefGradWS(tCell,tDimI,tDimJ) = tMechDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tMechDefGradWS,tHostMechDefGradWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // compute thermo-elastic deformation gradient
  auto tNumPoints = ElementType::mNumGaussPoints;
  Plato::ThermoElasticDeformationGradient<Residual> tComputeThermoElasticDeformationGradient;
  Kokkos::parallel_for("compute thermo-elastic deformation gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
      tCellMechDefGrad(StrainT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellMechDefGrad(tDimI,tDimJ) = tMechDefGradWS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,NodeStateT> 
      tCellTempDefGrad(NodeStateT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellTempDefGrad(tDimI,tDimJ) = tTempDefGradWS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    // compute thermo-elastic deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> 
      tTMechDefGrad(ResultT(0.));
    tComputeThermoElasticDeformationGradient(tCellTempDefGrad,tCellMechDefGrad,tTMechDefGrad);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tTMechDefGrad(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.72333333,0.10333333,0.20666667,0.62}, {0.72333333,0.10333333,0.20666667,0.62} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, KineticPullBackOperation_1 )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using VecStateT   = typename Residual::StateScalarType;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using StrainT     = typename Plato::fad_type_t<ElementType,VecStateT,ConfigT>;
  // create mechanical deformation gradient workset
  std::vector<std::vector<Plato::Scalar>> tDataDefGrad = 
    { {1.4,0.2,0.4,1.2}, {1.4,0.2,0.4,1.2} };
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  Plato::ScalarArray3DT<StrainT> tDefGradient("mechanical deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostDefGradWS = Kokkos::create_mirror(tDefGradient);
  Kokkos::deep_copy(tHostDefGradWS, tDefGradient);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostDefGradWS(tCell,tDimI,tDimJ) = tDataDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tDefGradient,tHostDefGradWS);
  // create second piola-kirchhoff stress data
  std::vector<std::vector<Plato::Scalar>> tData2PKS = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };  
  Plato::ScalarArray3DT<ResultT> t2PKS_WS("second piola-kirchhoff stress",tNumCells,tSpaceDim,tSpaceDim);
  auto tHost2PKS_WS = Kokkos::create_mirror(t2PKS_WS);
  Kokkos::deep_copy(tHost2PKS_WS, t2PKS_WS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHost2PKS_WS(tCell,tDimI,tDimJ) = tData2PKS[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(t2PKS_WS,tHost2PKS_WS);
  // create results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // pull back second piola-kirchhoff stress from deformed to undeformed configuration
  auto tNumPoints = ElementType::mNumGaussPoints;
  Plato::KineticPullBackOperation<Residual> tApplyKineticPullBackOperation;
  Kokkos::parallel_for("pull back stress operation", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tDefConfig2PKS(ResultT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tDefConfig2PKS(tDimI,tDimJ) = t2PKS_WS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tCellDefGrad(StrainT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellDefGrad(tDimI,tDimJ) = tDefGradient(iCellOrdinal,tDimI,tDimJ);
      }
    }
    // compute thermo-elastic deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tUndefConfig2PKS(ResultT(0.));
    tApplyKineticPullBackOperation(tCellDefGrad,tDefConfig2PKS,tUndefConfig2PKS);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tUndefConfig2PKS(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.9328371,-0.1743207,-0.1743207,0.9782719}, {0.9328371,-0.1743207,-0.1743207,0.9782719} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, KineticPullBackOperation_2 )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using VecStateT   = typename Residual::StateScalarType;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using NodeStateT  = typename Residual::NodeStateScalarType;
  using StrainT     = typename Plato::fad_type_t<ElementType,VecStateT,ConfigT>;
  // create thermal gradient workset
  std::vector<std::vector<Plato::Scalar>> tDataTempDefGrad = 
    { {0.5166666666666666,0.,0.,0.5166666666666666}, {0.5166666666666666,0.,0.,0.5166666666666666} };
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  Plato::ScalarArray3DT<NodeStateT> tTempDefGradWS("thermal deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostTempDefGradWS = Kokkos::create_mirror(tTempDefGradWS);
  Kokkos::deep_copy(tHostTempDefGradWS, tTempDefGradWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostTempDefGradWS(tCell,tDimI,tDimJ) = tDataTempDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tTempDefGradWS,tHostTempDefGradWS);
  // create second piola-kirchhoff stress data
  std::vector<std::vector<Plato::Scalar>> tData2PKS = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };  
  Plato::ScalarArray3DT<ResultT> t2PKS_WS("second piola-kirchhoff stress",tNumCells,tSpaceDim,tSpaceDim);
  auto tHost2PKS_WS = Kokkos::create_mirror(t2PKS_WS);
  Kokkos::deep_copy(tHost2PKS_WS, t2PKS_WS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHost2PKS_WS(tCell,tDimI,tDimJ) = tData2PKS[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(t2PKS_WS,tHost2PKS_WS);
  // create results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // pull back second piola-kirchhoff stress from deformed to undeformed configuration
  auto tNumPoints = ElementType::mNumGaussPoints;
  Plato::KineticPullBackOperation<Residual> tApplyKineticPullBackOperation;
  Kokkos::parallel_for("pull back stress operation", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tDefConfig2PKS(ResultT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tDefConfig2PKS(tDimI,tDimJ) = t2PKS_WS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,NodeStateT> 
      tCellTempDefGrad(NodeStateT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellTempDefGrad(tDimI,tDimJ) = tTempDefGradWS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    // compute thermo-elastic deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tUndefConfig2PKS(ResultT(0.));
    tApplyKineticPullBackOperation(tCellTempDefGrad,tDefConfig2PKS,tUndefConfig2PKS);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tUndefConfig2PKS(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, NominalStress )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using VecStateT   = typename Residual::StateScalarType;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using StrainT     = typename Plato::fad_type_t<ElementType,VecStateT,ConfigT>;
  // create mechanical deformation gradient workset
  std::vector<std::vector<Plato::Scalar>> tDataDefGrad = 
    { {1.4,0.2,0.4,1.2}, {1.4,0.2,0.4,1.2} };
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  Plato::ScalarArray3DT<StrainT> tDefGradient("mechanical deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostDefGradWS = Kokkos::create_mirror(tDefGradient);
  Kokkos::deep_copy(tHostDefGradWS, tDefGradient);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostDefGradWS(tCell,tDimI,tDimJ) = tDataDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tDefGradient,tHostDefGradWS);
  // create second piola-kirchhoff stress data
  std::vector<std::vector<Plato::Scalar>> tData2PKS = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };  
  Plato::ScalarArray3DT<ResultT> t2PKS_WS("second piola-kirchhoff stress",tNumCells,tSpaceDim,tSpaceDim);
  auto tHost2PKS_WS = Kokkos::create_mirror(t2PKS_WS);
  Kokkos::deep_copy(tHost2PKS_WS, t2PKS_WS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHost2PKS_WS(tCell,tDimI,tDimJ) = tData2PKS[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(t2PKS_WS,tHost2PKS_WS);
  // create results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // pull back second piola-kirchhoff stress from deformed to undeformed configuration
  auto tNumPoints = ElementType::mNumGaussPoints;
  Plato::NominalStress<Residual> tComputeNominalStress;
  Kokkos::parallel_for("pull back stress operation", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tCell2PKS(ResultT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCell2PKS(tDimI,tDimJ) = t2PKS_WS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tCellDefGrad(StrainT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellDefGrad(tDimI,tDimJ) = tDefGradient(iCellOrdinal,tDimI,tDimJ);
      }
    }
    // compute thermo-elastic deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tNominalStress(ResultT(0.));
    tComputeNominalStress(tCellDefGrad,tCell2PKS,tNominalStress);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tNominalStress(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.6049342,0.7802452,0.5679006,1.1555556}, {1.6049342,0.7802452,0.5679006,1.1555556} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, get_cell_2PKS )
{
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  // create second piola-kirchhoff stress data
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tNumCells = 2;
  std::vector<std::vector<Plato::Scalar>> tData2PKS = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };
  Plato::ScalarArray4D t2PKS_WS("second piola-kirchhoff stress",tNumCells,/*num_intg_pts=*/1,tSpaceDim,tSpaceDim);
  auto tHost2PKS_WS = Kokkos::create_mirror(t2PKS_WS);
  Kokkos::deep_copy(tHost2PKS_WS, t2PKS_WS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHost2PKS_WS(tCell,0,tDimI,tDimJ) = tData2PKS[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(t2PKS_WS,tHost2PKS_WS);
  // create results workset
  Plato::ScalarArray3D tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get cell 2PKS
  auto tNumPoints = ElementType::mNumGaussPoints;
  Kokkos::parallel_for("get cell 2PKS", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {

    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims> tCell2PKS(0.);
    Plato::Elliptic::get_cell_2PKS(iCellOrdinal,iGpOrdinal,t2PKS_WS,tCell2PKS);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tCell2PKS(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

}