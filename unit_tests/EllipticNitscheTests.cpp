/*
 * EllipticNitscheTests.cpp
 *
 *  Created on: July 6, 2023
 */

/// @include trilinos includes
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

/// @include analyze includes
#include "Tri3.hpp"
#include "FadTypes.hpp"
#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "SurfaceArea.hpp"
#include "SpatialModel.hpp"
#include "AnalyzeMacros.hpp"
#include "MaterialModel.hpp"
#include "GradientMatrix.hpp"
#include "WeightedNormalVector.hpp"
#include "InterpolateFromNodal.hpp"

#include "base/SupportedParamOptions.hpp"

#include "base/WorksetBase.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/base/WorksetBuilder.hpp"
#include "bcs/dirichlet/NitscheBase.hpp"
#include "elliptic/thermal/Thermal.hpp"
#include "elliptic/mechanical/StressEvaluator.hpp"
#include "elliptic/mechanical/linear/Mechanics.hpp"
#include "elliptic/mechanical/SupportedParamOptions.hpp"
#include "elliptic/thermal/FactoryThermalConductionMaterial.hpp"

#include "elliptic/mechanical/nonlinear/StateGradient.hpp"
#include "elliptic/mechanical/nonlinear/NominalStressTensor.hpp"
#include "elliptic/mechanical/nonlinear/DeformationGradient.hpp"
#include "elliptic/mechanical/nonlinear/KineticPullBackOperation.hpp"
#include "elliptic/mechanical/nonlinear/GreenLagrangeStrainTensor.hpp"
#include "elliptic/mechanical/nonlinear/KirchhoffSecondPiolaStress.hpp"
#include "elliptic/mechanical/nonlinear/NeoHookeanSecondPiolaStress.hpp"
#include "elliptic/mechanical/nonlinear/FactoryNonlinearElasticMaterial.hpp"

#include "elliptic/thermomechanics/nonlinear/ThermoMechanics.hpp"
#include "elliptic/thermomechanics/nonlinear/UtilitiesThermoMechanics.hpp"
#include "elliptic/thermomechanics/nonlinear/ThermalDeformationGradient.hpp"
#include "elliptic/thermomechanics/nonlinear/ThermoElasticDeformationGradient.hpp"

/// @include analyze unit test includes
#include "util/PlatoTestHelpers.hpp"

namespace Plato
{

template<typename EvaluationType>
class MaterialIsotropicElastic : public MaterialModel<EvaluationType>
{
public:
  MaterialIsotropicElastic(){}

  MaterialIsotropicElastic(
    const Teuchos::ParameterList& aParamList
  )
  {
    this->parse(aParamList);
    this->computeLameConstants();
  }

  ~MaterialIsotropicElastic(){}

  Plato::Scalar 
  mu() 
  const
  { return this->getScalarConstant("mu"); }
  
  void 
  mu(
    const Plato::Scalar & aValue
  )
  { this->setScalarConstant("mu",aValue); }

  Plato::Scalar 
  lambda() 
  const
  { return this->getScalarConstant("lambda"); }
  
  void 
  lambda(
    const Plato::Scalar & aValue
  )
  { this->setScalarConstant("lambda",aValue); }

private:
  void 
  parse(
    const Teuchos::ParameterList& aParamList
  )
  {
    this->parseScalarConstant("Youngs Modulus", aParamList);
    this->parseScalarConstant("Poissons Ratio", aParamList);
  }

  void 
  computeLameConstants()
  {
    auto tYoungsModulus = this->getScalarConstant("youngs modulus");
    if(tYoungsModulus <= std::numeric_limits<Plato::Scalar>::epsilon())
    {
      ANALYZE_THROWERR(std::string("ERROR: The Young's Modulus is less than the machine epsilon. ")
        + "The input material properties were not parsed properly.");
    }

    auto tPoissonsRatio = this->getScalarConstant("poissons ratio");
    if(tPoissonsRatio <= std::numeric_limits<Plato::Scalar>::epsilon())
    {
      ANALYZE_THROWERR(std::string("ERROR: The Poisson's Ratio is less than the machine epsilon. ")
        + "The input material properties were not parsed properly.");
    }
    auto tMu = tYoungsModulus / (2.0 * (1.0 + tPoissonsRatio) );
    this->setScalarConstant("mu",tMu);
    auto tLambda = (tYoungsModulus * tPoissonsRatio) / ( (1.0 + tPoissonsRatio) * (1.0 - 2.0 * tPoissonsRatio) );
    this->setScalarConstant("lambda",tLambda);
  }

};

template<typename EvaluationType>
class FactoryElasticMaterial
{
private:
  const Teuchos::ParameterList& mParamList;

public:
  FactoryElasticMaterial(
    const Teuchos::ParameterList& aParamList
  ) :
    mParamList(aParamList){}

  std::shared_ptr<Plato::MaterialModel<EvaluationType>>
  create(
    std::string aModelName
  ) const
  {
    if (!mParamList.isSublist("Material Models"))
    {
      ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
    }
    else
    {
      auto tModelsParamList = mParamList.get<Teuchos::ParameterList>("Material Models");
      if (!tModelsParamList.isSublist(aModelName))
      {
        std::stringstream tSS;
        tSS << "ERROR: Requested a material model ('" << aModelName << "') that isn't defined";
        ANALYZE_THROWERR(tSS.str());
      }
      auto tModelParamList = tModelsParamList.sublist(aModelName);
      if(tModelParamList.isSublist("Isotropic Linear Elastic"))
      {
        return ( std::make_shared<Plato::MaterialIsotropicElastic<EvaluationType>>( 
          tModelParamList.sublist("Isotropic Linear Elastic") ) 
        );
      }
      else
      {
        auto tErrMsg = this->getErrorMsg();
        ANALYZE_THROWERR(tErrMsg);
      }
    }
  }

private:
  /*!< map from input force type string to supported enum */
  std::vector<std::string> mSupportedMaterials =
    {"isotropic linear elastic"};

  std::string
  getErrorMsg()
  const
  {
    std::string tMsg = std::string("ERROR: Requested material constitutive model is not supported. ")
      + "Supported material constitutive models for mechanical analyses are: ";
    for(const auto& tElement : mSupportedMaterials)
    {
      tMsg = tMsg + "'" + tElement + "', ";
    }
    auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
    return tSubMsg;
  }
};

template<typename EvaluationType>
class FactoryMechanicalMaterials
{
public:
  std::shared_ptr<Plato::MaterialModel<EvaluationType>>
  create(
    const std::string            & aMaterialName,
          Teuchos::ParameterList & aParamList
  )
  {
    Plato::PhysicsEnum tS2E;
    auto tResponse = aParamList.get<std::string>("Response","Linear");
    auto tResponseEnum = tS2E.response(tResponse);
    switch (tResponseEnum)
    {
    case Plato::response_t::LINEAR:
    {
      Plato::FactoryElasticMaterial<EvaluationType> tFactory(aParamList);
      return ( tFactory.create(aMaterialName) );
      break;
    }
    case Plato::response_t::NONLINEAR:
    {
      Plato::FactoryNonlinearElasticMaterial<EvaluationType> tFactory(aParamList);
      return ( tFactory.create(aMaterialName) );
      break;
    }
    default:
      ANALYZE_THROWERR(std::string("ERROR: Response '") + tResponse 
        + "' does not support weak enforcement of Dirichlet boundary conditions")
      break;
    }
  }
};

template<typename EvaluationType>
class ComputeStrainTensor
{
private:
  // set local element type
  using ElementType = typename EvaluationType::ElementType;
  // set local static parameters
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell; /*!< number of nodes per element */

public:
  template
  <typename StrainScalarType,
   typename GradientScalarType>
  KOKKOS_INLINE_FUNCTION void
  operator()(
    const Plato::OrdinalType                                                  & aCellOrdinal,
    const Plato::Matrix<mNumNodesPerCell,mNumSpatialDims, GradientScalarType> & aGradient,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims, StrainScalarType>    & aVirtualStrains
  ) const
  {
    constexpr Plato::Scalar tDeltaState = 1.0;
    constexpr Plato::Scalar tOneOverTwo = 0.5;
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
    {
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
      {
        for(Plato::OrdinalType tNodeI = 0; tNodeI < mNumNodesPerCell; tNodeI++)
        {
          aVirtualStrains(tDimI,tDimJ) += tOneOverTwo *
              ( ( tDeltaState * aGradient(tNodeI, tDimI) )
              + ( tDeltaState * aGradient(tNodeI, tDimJ) ) );
        }
      }
    }
  }

  template
  <typename StrainScalarType,
   typename DispScalarType,
   typename GradientScalarType>
  KOKKOS_INLINE_FUNCTION void
  operator()(
    const Plato::OrdinalType                                                  & aCellOrdinal,
    const Plato::ScalarMultiVectorT<DispScalarType>                           & aState,
    const Plato::Matrix<mNumNodesPerCell,mNumSpatialDims, GradientScalarType> & aGradient,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims, StrainScalarType>    & aStrains
  ) const
  {
    constexpr Plato::Scalar tOneOverTwo = 0.5;
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
    {
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
      {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
          auto tLocalOrdinalI = tNodeIndex * mNumSpatialDims + tDimI;
          auto tLocalOrdinalJ = tNodeIndex * mNumSpatialDims + tDimJ;
          aStrains(tDimI,tDimJ) += tOneOverTwo *
            ( aState(aCellOrdinal, tLocalOrdinalJ) * aGradient(tNodeIndex, tDimI)
            + aState(aCellOrdinal, tLocalOrdinalI) * aGradient(tNodeIndex, tDimJ));
        }
      }
    }
  }
};

template< typename EvaluationType>
class ComputeIsotropicElasticStressTensor
{
private:
  // set local element type
  using ElementType = typename EvaluationType::ElementType;
  // set local static parameters
  static constexpr auto mNumSpatialDims   = ElementType::mNumSpatialDims;
  static constexpr auto mNumNodesPerCell  = ElementType::mNumNodesPerCell; /*!< number of nodes per element */
  using StateScalarType  = typename EvaluationType::StateScalarType;  
  using ConfigScalarType = typename EvaluationType::ConfigScalarType; 
  using ResultScalarType = typename EvaluationType::ResultScalarType; 
  using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>; 
  // set local member data
  Plato::Scalar mMu = 1.0;
  Plato::Scalar mLambda = 1.0;
public:
  ComputeIsotropicElasticStressTensor(
    Plato::MaterialModel<EvaluationType> & aMaterialModel
  )
  {
    mMu = aMaterialModel.getScalarConstant("mu");
    if(mMu <= std::numeric_limits<Plato::Scalar>::epsilon())
    {
      ANALYZE_THROWERR(std::string("Error: Lame constant 'mu' is less than the machine epsilon. ")
          + "The input material properties were not parsed properly.");
    }
    mLambda = aMaterialModel.getScalarConstant("lambda");
    if(mLambda <= std::numeric_limits<Plato::Scalar>::epsilon())
    {
      ANALYZE_THROWERR(std::string("Error: Lame constant 'lambda' is less than the machine epsilon. ")
          + "The input material properties were not parsed properly.");
    }
  }

  template<typename ResultScalarType,
           typename StrainScalarType>
  KOKKOS_INLINE_FUNCTION void
  operator()
  (const Plato::Matrix<mNumSpatialDims, mNumSpatialDims, StrainScalarType> & aStrainTensor,
         Plato::Matrix<mNumSpatialDims, mNumSpatialDims, ResultScalarType> & aStressTensor) const
  {
    // compute first strain invariant
    StrainScalarType tFirstStrainInvariant(0.0);
    for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++){
      tFirstStrainInvariant += aStrainTensor(tDim,tDim);
    }
    // add contribution from first stress invariant to the stress tensor
    for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++){
      aStressTensor(tDim,tDim) += mLambda * tFirstStrainInvariant;
    }
    // add shear stress contribution to the stress tensor
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        aStressTensor(tDimI,tDimJ) += 2.0 * mMu * aStrainTensor(tDimI,tDimJ);
      }
    }
  }
};



template<typename EvaluationType>
class ComputeSideCellVolumes
{
private:
  // set local element types
  using ElementType = typename EvaluationType::ElementType;
  // set local fad type definitions
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  // set local member data
  std::string mSideSetName;

public:
  ComputeSideCellVolumes(
    const std::string& aEntitySetName
  ) :
    mSideSetName(aEntitySetName)
  {}

  void 
  operator()(
    const Plato::SpatialModel                    & aSpatialModel,
    const Plato::WorkSets                        & aWorkSets,
          Plato::ScalarVectorT<ConfigScalarType> & aSideCellVolumes
  )
  {
    // get side set connectivity information
    auto tSideCellOrdinals = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
    // get input workset
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    // get body quadrature points and weights
    auto tCubPoints  = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints  = tCubWeights.size();
    // compute volume of each cell in the entity set
    Kokkos::parallel_for("compute volume", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumSideCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal, const Plato::OrdinalType aGpOrdinal)
    {
      auto tCubPoint  = tCubPoints(aGpOrdinal);
      auto tCubWeight = tCubWeights(aGpOrdinal);
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      auto tJacobian = ElementType::jacobian(tCubPoint, tConfigWS, tCellOrdinal);
      ConfigScalarType tVolume = Plato::determinant(tJacobian);
      tVolume *= tCubWeight;
      Kokkos::atomic_add(&aSideCellVolumes(aSideOrdinal), tVolume);
    });
  }
};

template<typename EvaluationType>
class ComputeSideCellFaceAreas
{
private:
  // set local element types
  using BodyElementType = typename EvaluationType::ElementType;
  using FaceElementType = typename BodyElementType::Face;
  // set local constexpr members
  static constexpr auto mNumNodesPerFace = BodyElementType::mNumNodesPerFace;
  // set local fad type definitions
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  // set local member data
  std::string mSideSetName;

public:
  ComputeSideCellFaceAreas(
    const std::string& aEntitySetName
  ) :
    mSideSetName(aEntitySetName)
  {}

  void 
  operator()(
    const Plato::SpatialModel                    & aSpatialModel,
    const Plato::WorkSets                        & aWorkSets,
          Plato::ScalarVectorT<ConfigScalarType> & aSideCellFaceAreas)
  {
    // get side set connectivity information
    auto tSideCellOrdinals = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tLocalNodeOrds    = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
    // get input workset
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    // create surface area functor
    Plato::SurfaceArea<BodyElementType> tComputeFaceArea;
    // get face quadrature points and weights
    auto tFaceCubPoints    = FaceElementType::getCubPoints();
    auto tFaceCubWeights   = FaceElementType::getCubWeights();
    auto tFaceNumCubPoints = tFaceCubWeights.size();
    // compute characteristic length of each cell in the entity set
    Kokkos::parallel_for("compute volume", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumSideCells, tFaceNumCubPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal, const Plato::OrdinalType aGpOrdinal)
    {
      // get quadrature point and weight
      auto tFaceCubPoint   = tFaceCubPoints(aGpOrdinal);
      auto tFaceCubWeight  = tFaceCubWeights(aGpOrdinal);
      auto tFaceBasisGrads = FaceElementType::basisGrads(tFaceCubPoint);
      // get local node ordinal on boundary entity
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrdinals;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrdinals(tIndex) = tLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute entity area
      ConfigScalarType tFaceArea(0.0);
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      tComputeFaceArea(tCellOrdinal,tFaceLocalNodeOrdinals,tFaceBasisGrads,tConfigWS,tFaceArea);
      tFaceArea *= tFaceCubWeight;
      // add characteristic length contribution from this quadrature point, i.e. Gauss point
      Kokkos::atomic_add(&aSideCellFaceAreas(aSideOrdinal), tFaceArea);
    });
  }
};



template<typename EvaluationType>
class ComputeCharacteristicLength
{
private:
  // set local fad type definitions
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  // set local member data
  std::string mSideSetName;

public:
  ComputeCharacteristicLength(
    const std::string& aEntitySetName
  ) :
    mSideSetName(aEntitySetName)
  {}

  void operator()(
    const Plato::SpatialModel                    & aSpatialModel,
    const Plato::WorkSets                        & aWorkSets,
          Plato::ScalarVectorT<ConfigScalarType> & aCharLength
  )
  {
    // get side set connectivity information
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tLocalNodeOrds     = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
    // compute volumes of cells in side
    Plato::ScalarVectorT<ConfigScalarType> tSideCellVolumes("volume",tNumSideCells);
    Plato::ComputeSideCellVolumes<EvaluationType> tComputeSideCellVolumes(mSideSetName);
    tComputeSideCellVolumes(aSpatialModel,aWorkSets,tSideCellVolumes);
    // compute face areas of cells in side
    Plato::ScalarVectorT<ConfigScalarType> tSideCellFaceAreas("area",tNumSideCells);
    Plato::ComputeSideCellFaceAreas<EvaluationType> tComputeSideCellFaceAreas(mSideSetName);
    tComputeSideCellFaceAreas(aSpatialModel,aWorkSets,tSideCellFaceAreas);
    // compute characteristic length of each cell in the entity set
    Kokkos::parallel_for("compute characteristic length", Kokkos::RangePolicy<>(0, tNumSideCells),
    KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal)
    {
      aCharLength(aSideOrdinal) = tSideCellVolumes(aSideOrdinal) / tSideCellFaceAreas(aSideOrdinal);
    });
  }
};


/// @brief parent class for nitsche integral evaluators
class NitscheEvaluator
{
protected:
  /// @brief side set name
  std::string mSideSetName;
  /// @brief name assigned to the material model applied on this side set
  std::string mMaterialName;

public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  NitscheEvaluator(
    Teuchos::ParameterList & aParamList
  )
  {
    this->initialize(aParamList);
  }

  /// @brief class destructor
  virtual 
  ~NitscheEvaluator()
  {}

  /// @fn sideset
  /// @brief return side set name
  /// @return string
  std::string 
  sideset() 
  const 
  { return mSideSetName; }

  /// @fn material
  /// @brief return name assigned to material constitutive model
  /// @return string
  std::string 
  material() 
  const 
  { return mMaterialName; }

  virtual
  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
        Plato::Scalar           aCycle = 0.0,
        Plato::Scalar           aScale = 1.0
  ) = 0;

private:
  /// @fn initialize
  /// @brief initialize member data
  /// @param [in] aParamList input problem parameters
  void 
  initialize(
    Teuchos::ParameterList & aParamList
  )
  {
    if( !aParamList.isParameter("Sides") ){
      ANALYZE_THROWERR( std::string("ERROR: Input argument ('Sides') is not defined, ") + 
        "side set for Nitsche's method cannot be determined" )
    }
    mSideSetName = aParamList.get<std::string>("Sides");
    
    if( !aParamList.isParameter("Material Model") ){
      ANALYZE_THROWERR( std::string("ERROR: Input argument ('Material Model') is not defined, ") + 
        "material constitutive model for Nitsche's method cannot be determined" )
    }
    mMaterialName = aParamList.get<std::string>("Material Model");
  }

};

/// @class BoundaryFluxEvaluator
/// @brief base class for stress evaluators
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class BoundaryFluxEvaluator
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;

protected:
  /// @brief name assigned to side set
  std::string mSideSetName;
  /// @brief name assigned to the material model applied on this side set
  std::string mMaterialName;

public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  explicit
  BoundaryFluxEvaluator(
    Teuchos::ParameterList& aNitscheParams
  )
  {
    this->initialize(aNitscheParams);
  }

  /// @brief class destructor
  virtual ~BoundaryFluxEvaluator(){}

  /// @fn evaluate
  /// @brief evaluate flux on boundary
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in]     aWorkSets     domain and range workset database
  /// @param [in,out] aResult       4D scalar container
  /// @param [in]     aCycle        scalar
  virtual 
  void 
  evaluate(
    const Plato::SpatialModel                     & aSpatialModel,
    const Plato::WorkSets                         & aWorkSets,
          Plato::ScalarArray4DT<ResultScalarType> & aResult,
          Plato::Scalar                             aCycle
  ) const = 0;

private:
  /// @fn initialize
  /// @brief initialize member data
  /// @param [in] aParamList input problem parameters
  void 
  initialize(
    Teuchos::ParameterList & aParamList
  )
  {
    if( !aParamList.isParameter("Sides") ){
      ANALYZE_THROWERR( std::string("ERROR: Input argument ('Sides') is not defined, ") + 
        "side set for Nitsche's method cannot be determined" )
    }
    mSideSetName = aParamList.get<std::string>("Sides");
    
    if( !aParamList.isParameter("Material Model") ){
      ANALYZE_THROWERR( std::string("ERROR: Input argument ('Material Model') is not defined, ") + 
        "material constitutive model for Nitsche's method cannot be determined" )
    }
    mMaterialName = aParamList.get<std::string>("Material Model");
  }

};

template<typename EvaluationType>
class BoundarEvaluatoryTrialIsotropicElasticStress : public Plato::BoundaryFluxEvaluator<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;
  /// @brief local typename for base class
  using BaseClassType = Plato::BoundaryFluxEvaluator<EvaluationType>;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to the material constitutive model
  using BaseClassType::mMaterialName;

public:
  BoundarEvaluatoryTrialIsotropicElasticStress(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    Plato::FactoryElasticMaterial<EvaluationType> tFactory(aParamList);
    mMaterialModel = tFactory.create(mMaterialName);
  }

  void 
  evaluate(
    const Plato::SpatialModel                     & aSpatialModel,
    const Plato::WorkSets                         & aWorkSets,
          Plato::ScalarArray4DT<ResultScalarType> & aResult,
          Plato::Scalar                             aCycle
  ) const
  {
    // unpack worksets
    //
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    // create local functors
    //
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::ComputeStrainTensor<EvaluationType> tComputeStrainTensor;
    Plato::ComputeIsotropicElasticStressTensor<EvaluationType> tComputeStressTensor(mMaterialModel.operator*());
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // evaluate integral
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Kokkos::parallel_for("boundary trial stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      // quadrature data to evaluate integral on the body surface of interest
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      // compute strains and stresses at quadrature point
      //
      ConfigScalarType tVolume(0.0);
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims, ConfigScalarType> tGradient;
      tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims, StrainScalarType>  tStrainTensor(0.0);
      tComputeStrainTensor(tCellOrdinal,tStateWS, tGradient, tStrainTensor);
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ResultScalarType>  tStressTensor(0.0);
      tComputeStressTensor(tStrainTensor,tStressTensor);
      // copy stress tensor to output workset
      //
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
          aResult(tCellOrdinal,aPointOrdinal,tDimI,tDimJ) = tStressTensor(tDimI,tDimJ);
        }
      }
    });
  }
};

template<typename EvaluationType>
class BoundarEvaluatoryTestIsotropicElasticStress : public Plato::BoundaryFluxEvaluator<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;
  /// @brief local typename for base class
  using BaseClassType = Plato::BoundaryFluxEvaluator<EvaluationType>;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to the material constitutive model
  using BaseClassType::mMaterialName;

public:
  BoundarEvaluatoryTestIsotropicElasticStress(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    Plato::FactoryElasticMaterial<EvaluationType> tFactory(aParamList);
    mMaterialModel = tFactory.create(mMaterialName);
  }

  void 
  evaluate(
    const Plato::SpatialModel                     & aSpatialModel,
    const Plato::WorkSets                         & aWorkSets,
          Plato::ScalarArray4DT<ResultScalarType> & aResult,
          Plato::Scalar                             aCycle
  ) const
  {
    // unpack worksets
    //
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    // create local functors
    //
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::ComputeStrainTensor<EvaluationType> tComputeStrainTensor;
    Plato::ComputeIsotropicElasticStressTensor<EvaluationType> tComputeStressTensor(mMaterialModel.operator*());
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // evaluate integral
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Kokkos::parallel_for("boundary test stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      // quadrature data to evaluate integral on the body surface of interest
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      // compute trial stress tensor
      //
      ConfigScalarType tVolume(0.0);
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims, ConfigScalarType> tGradient;
      tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ConfigScalarType>  tVirtualStrainTensor(0.0);
      tComputeStrainTensor(tCellOrdinal,tGradient,tVirtualStrainTensor);
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ConfigScalarType>  tVirtualStressTensor(0.0);
      tComputeStressTensor(tVirtualStrainTensor,tVirtualStressTensor);
      // copy stress tensor to output workset
      //
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
          aResult(tCellOrdinal,aPointOrdinal,tDimI,tDimJ) = tVirtualStressTensor(tDimI,tDimJ);
        }
      }
    });
  }
};

template<typename EvaluationType>
class BoundaryEvaluatoryTrialKirchhoffStress : public Plato::BoundaryFluxEvaluator<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;
  /// @brief local typename for base class
  using BaseClassType = Plato::BoundaryFluxEvaluator<EvaluationType>;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to the material constitutive model
  using BaseClassType::mMaterialName;

public:
  BoundaryEvaluatoryTrialKirchhoffStress(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    Plato::FactoryNonlinearElasticMaterial<EvaluationType> tFactory(aParamList);
    mMaterialModel = tFactory.create(mMaterialName);
  }

  void 
  evaluate(
    const Plato::SpatialModel                     & aSpatialModel,
    const Plato::WorkSets                         & aWorkSets,
          Plato::ScalarArray4DT<ResultScalarType> & aResult,
          Plato::Scalar                             aCycle
  ) const
  {
    // unpack worksets
    //
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    // create local functors
    //
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
    Plato::GreenLagrangeStrainTensor<EvaluationType> tGreenLagrangeStrainTensor;
    Plato::KirchhoffSecondPiolaStress<EvaluationType> tComputeSecondPiolaKirchhoffStress(*mMaterialModel);
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // evaluate integral
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Kokkos::parallel_for("boundary test stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      // quadrature data to evaluate integral on the body surface of interest
      //
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      // compute gradient of interpolation functions
      //
      ConfigScalarType tVolume(0.0);
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
      tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
      // compute state gradient
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tStateGradient(StrainScalarType(0.));
      tComputeStateGradient(tCellOrdinal,tStateWS,tGradient,tStateGradient);
      // compute deformation gradient 
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tDefGradient(StrainScalarType(0.));
      tComputeDeformationGradient(tStateGradient,tDefGradient);
      // compute green-lagrange strain tensor
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tStrainTensor(StrainScalarType(0.));
      tGreenLagrangeStrainTensor(tStateGradient,tStrainTensor);
      // compute second piola-kirchhoff stress
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> 
        tStressTensor2PK(ResultScalarType(0.));
      tComputeSecondPiolaKirchhoffStress(tStrainTensor,tStressTensor2PK);
      // copy stress tensor to output workset
      //
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
          aResult(tCellOrdinal,aPointOrdinal,tDimI,tDimJ) = tStressTensor2PK(tDimI,tDimJ);
        }
      }
    });
  }
};

template<typename EvaluationType>
class BoundaryEvaluatorTestKirchhoffStress : public Plato::BoundaryFluxEvaluator<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;
  /// @brief local typename for base class
  using BaseClassType = Plato::BoundaryFluxEvaluator<EvaluationType>;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to the material constitutive model
  using BaseClassType::mMaterialName;

public:
  BoundaryEvaluatorTestKirchhoffStress(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    Plato::FactoryNonlinearElasticMaterial<EvaluationType> tFactory(aParamList);
    mMaterialModel = tFactory.create(mMaterialName);
  }

  void 
  evaluate(
    const Plato::SpatialModel                     & aSpatialModel,
    const Plato::WorkSets                         & aWorkSets,
          Plato::ScalarArray4DT<ResultScalarType> & aResult,
          Plato::Scalar                             aCycle
  ) const
  {
    // unpack worksets
    //
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    // create local functors
    //
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
    Plato::GreenLagrangeStrainTensor<EvaluationType> tGreenLagrangeStrainTensor;
    Plato::KirchhoffSecondPiolaStress<EvaluationType> tComputeSecondPiolaKirchhoffStress(*mMaterialModel);
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // evaluate integral
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Kokkos::parallel_for("boundary test stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      // quadrature data to evaluate integral on the body surface of interest
      //
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      // compute gradient of interpolation functions
      //
      ConfigScalarType tVolume(0.0);
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
      tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
      // compute state gradient
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tStateGradient(StrainScalarType(0.));
      tComputeStateGradient(tCellOrdinal,tGradient,tStateGradient);
      // compute deformation gradient 
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tDefGradient(StrainScalarType(0.));
      tComputeDeformationGradient(tStateGradient,tDefGradient);
      // compute green-lagrange strain tensor
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tStrainTensor(StrainScalarType(0.));
      tGreenLagrangeStrainTensor(tStateGradient,tStrainTensor);
      // compute second piola-kirchhoff stress
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> 
        tStressTensor2PK(ResultScalarType(0.));
      tComputeSecondPiolaKirchhoffStress(tStrainTensor,tStressTensor2PK);
      // copy stress tensor to output workset
      //
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
          aResult(tCellOrdinal,aPointOrdinal,tDimI,tDimJ) = tStressTensor2PK(tDimI,tDimJ);
        }
      }
    });
  }
};


template<typename EvaluationType>
class BoundaryEvaluatorTrialNeoHookeanStress : public Plato::BoundaryFluxEvaluator<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;
  /// @brief local typename for base class
  using BaseClassType = Plato::BoundaryFluxEvaluator<EvaluationType>;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to the material constitutive model
  using BaseClassType::mMaterialName;

public:
  BoundaryEvaluatorTrialNeoHookeanStress(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    Plato::FactoryNonlinearElasticMaterial<EvaluationType> tFactory(aParamList);
    mMaterialModel = tFactory.create(mMaterialName);
  }

  void 
  evaluate(
    const Plato::SpatialModel                     & aSpatialModel,
    const Plato::WorkSets                         & aWorkSets,
          Plato::ScalarArray4DT<ResultScalarType> & aResult,
          Plato::Scalar                             aCycle
  ) const
  {
    // unpack worksets
    //
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    // create local functors
    //
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
    Plato::NeoHookeanSecondPiolaStress<EvaluationType> tComputeSecondPiolaKirchhoffStress(*mMaterialModel);
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // evaluate integral
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Kokkos::parallel_for("boundary test stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      // quadrature data to evaluate integral on the body surface of interest
      //
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      // compute gradient of interpolation functions
      //
      ConfigScalarType tVolume(0.0);
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
      tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
      // compute state gradient
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tStateGradient(StrainScalarType(0.));
      tComputeStateGradient(tCellOrdinal,tStateWS,tGradient,tStateGradient);
      // compute deformation gradient 
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tDefGradient(StrainScalarType(0.));
      tComputeDeformationGradient(tStateGradient,tDefGradient);
      // compute second piola-kirchhoff stress
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> 
        tStressTensor2PK(ResultScalarType(0.));
      tComputeSecondPiolaKirchhoffStress(tDefGradient,tStressTensor2PK);
      // copy stress tensor to output workset
      //
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
          aResult(tCellOrdinal,aPointOrdinal,tDimI,tDimJ) = tStressTensor2PK(tDimI,tDimJ);
        }
      }
    });
  }
};


template<typename EvaluationType>
class BoundaryEvaluatorTestNeoHookeanStress : public Plato::BoundaryFluxEvaluator<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;
  /// @brief local typename for base class
  using BaseClassType = Plato::BoundaryFluxEvaluator<EvaluationType>;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to the material constitutive model
  using BaseClassType::mMaterialName;

public:
  BoundaryEvaluatorTestNeoHookeanStress(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    Plato::FactoryNonlinearElasticMaterial<EvaluationType> tFactory(aParamList);
    mMaterialModel = tFactory.create(mMaterialName);
  }

  void 
  evaluate(
    const Plato::SpatialModel                     & aSpatialModel,
    const Plato::WorkSets                         & aWorkSets,
          Plato::ScalarArray4DT<ResultScalarType> & aResult,
          Plato::Scalar                             aCycle
  ) const
  {
    // unpack worksets
    //
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    // create local functors
    //
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
    Plato::NeoHookeanSecondPiolaStress<EvaluationType> tComputeSecondPiolaKirchhoffStress(*mMaterialModel);
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // evaluate integral
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Kokkos::parallel_for("boundary test stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      // quadrature data to evaluate integral on the body surface of interest
      //
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      // compute gradient of interpolation functions
      //
      ConfigScalarType tVolume(0.0);
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
      tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
      // compute state gradient
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tStateGradient(StrainScalarType(0.));
      tComputeStateGradient(tCellOrdinal,tGradient,tStateGradient);
      // compute deformation gradient 
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tDefGradient(StrainScalarType(0.));
      tComputeDeformationGradient(tStateGradient,tDefGradient);
      // compute second piola-kirchhoff stress
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> 
        tStressTensor2PK(ResultScalarType(0.));
      tComputeSecondPiolaKirchhoffStress(tDefGradient,tStressTensor2PK);
      // copy stress tensor to output workset
      //
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
          aResult(tCellOrdinal,aPointOrdinal,tDimI,tDimJ) = tStressTensor2PK(tDimI,tDimJ);
        }
      }
    });
  }
};

namespace Elliptic
{
  

template<typename EvaluationType>
class NitscheTrialElasticStressEvaluator : public Plato::NitscheEvaluator
{
private:
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of degrees of freedom per parent body element vertex/node
  static constexpr auto mNumDofsPerNode = BodyElementBase::mNumDofsPerNode;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of nodes per parent body element surface
  static constexpr auto mNumNodesPerFace = BodyElementBase::mNumNodesPerFace;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to the material constitutive model
  using BaseClassType::mMaterialName;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<BodyElementBase,StateScalarType,ConfigScalarType>;
  /// @brief evaluates boundary trial and test stress tensors
  std::shared_ptr<Plato::BoundarEvaluatoryTrialIsotropicElasticStress<EvaluationType>> mBoundaryStressEvaluator;

public:
  NitscheTrialElasticStressEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // create boundary stress evaluator
    //
    mBoundaryStressEvaluator = 
      std::make_shared<Plato::BoundarEvaluatoryTrialIsotropicElasticStress<EvaluationType>>(aParamList,aNitscheParams);
  }
  ~NitscheTrialElasticStressEvaluator()
  {}

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  )
  {
    // unpack worksets
    //
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    // create local functors
    //
    Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // evaluate boundary trial stress tensors
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Plato::ScalarArray4DT<ResultScalarType> tStressTensors(
      "stress tensors",tNumCellsOnSideSet,mNumGaussPointsPerFace,mNumSpatialDims,mNumSpatialDims
    );
    mBoundaryStressEvaluator->evaluate(aSpatialModel,aWorkSets,tStressTensors,aCycle);
    // evaluate integral
    //
    Kokkos::parallel_for("nitsche stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      auto tCubPointOnParentFaceElem = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsOnParentFaceElem = FaceElementBase::basisGrads(tCubPointOnParentFaceElem);
      // quadrature data to evaluate integral on the body surface of interest
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      auto tCubWeightOnParentBodyElemSurface = tCubWeightsOnParentBodyElemSurface(aPointOrdinal);
      auto tBasisValuesOnParentBodyElemSurface = BodyElementBase::basisValues(tCubPointOnParentBodyElemSurface);
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute surface area weighted normal vector
      //
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Array<mNumSpatialDims, ConfigScalarType> tWeightedNormalVector;
      tComputeWeightedNormalVector(
        tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tWeightedNormalVector
      );
      // evaluate: int_{\Gamma_D} \delta{u}\cdot(\sigma\cdot{n}) d\Gamma_D
      //
      for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++){
        for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++){
          auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
          ResultScalarType tStressTimesSurfaceWeightedNormal(0.0);
          for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++){
            tStressTimesSurfaceWeightedNormal += 
              tStressTensors(tCellOrdinal,aPointOrdinal,tDimI,tDimJ) * tWeightedNormalVector[tDimJ];
          }
          ResultScalarType tValue = -aScale * tBasisValuesOnParentBodyElemSurface(tNode) 
            * tCubWeightOnParentBodyElemSurface * tStressTimesSurfaceWeightedNormal;
          Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
        }
      }
    });
  }
};

template<typename EvaluationType>
class NitscheTestElasticStressEvaluator : public Plato::NitscheEvaluator
{
private:
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of degrees of freedom per parent body element vertex/node
  static constexpr auto mNumDofsPerNode = BodyElementBase::mNumDofsPerNode;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of nodes per parent body element surface
  static constexpr auto mNumNodesPerFace = BodyElementBase::mNumNodesPerFace;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to the material constitutive model
  using BaseClassType::mMaterialName;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<BodyElementBase,StateScalarType,ConfigScalarType>;
  /// @brief evaluates boundary trial and test stress tensors
  std::shared_ptr<Plato::BoundarEvaluatoryTestIsotropicElasticStress<EvaluationType>> mBoundaryStressEvaluator;

public:
  NitscheTestElasticStressEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // create boundary stress evaluator
    //
    mBoundaryStressEvaluator = 
      std::make_shared<Plato::BoundarEvaluatoryTestIsotropicElasticStress<EvaluationType>>(aParamList,aNitscheParams);
  }
  ~NitscheTestElasticStressEvaluator()
  {}

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
        Plato::Scalar           aCycle = 0.0,
        Plato::Scalar           aScale = 1.0
  )
  {
    // unpack worksets
    //
    Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    Plato::ScalarMultiVector tDirichletWS = 
      Plato::unpack<Plato::ScalarMultiVector>(aWorkSets.get("dirichlet"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    // create local functors
    //
    Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // evaluate boundary test stress tensors
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Plato::ScalarArray4DT<ResultScalarType> tStressTensors(
      "stress tensors",tNumCellsOnSideSet,mNumGaussPointsPerFace,mNumSpatialDims,mNumSpatialDims
    );
    mBoundaryStressEvaluator->evaluate(aSpatialModel,aWorkSets,tStressTensors,aCycle);
    // evaluate integral
    //
    Kokkos::parallel_for("nitsche stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      auto tCubPointOnParentFaceElem = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsOnParentFaceElem = FaceElementBase::basisGrads(tCubPointOnParentFaceElem);
      // quadrature data to evaluate integral on the body surface of interest
      //
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      auto tCubWeightOnParentBodyElemSurface = tCubWeightsOnParentBodyElemSurface(aPointOrdinal);
      auto tBasisValuesOnParentBodyElemSurface = BodyElementBase::basisValues(tCubPointOnParentBodyElemSurface);
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute surface area weighted normal vector
      //
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Array<mNumSpatialDims, ConfigScalarType> tWeightedNormalVector;
      tComputeWeightedNormalVector(
        tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tWeightedNormalVector
      );
      // project state from nodes to quadrature/cubature point
      //
      Plato::Array<mNumSpatialDims,StateScalarType> tProjectedStates;
      for(Plato::OrdinalType tDof = 0; tDof < mNumDofsPerNode; tDof++){
        tProjectedStates(tDof) = 0.0;
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++){
          Plato::OrdinalType tCellDofIndex = (mNumDofsPerNode * tNodeIndex) + tDof;
          tProjectedStates(tDof) += tStateWS(tCellOrdinal, tCellDofIndex) * 
            tBasisValuesOnParentBodyElemSurface(tNodeIndex);
        }
      }
      // evaluate: int_{\Gamma_D} \delta(\sigma\cdot{n})\cdot(u - u_D) d\Gamma_D
      //
      for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++)
      {
        for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
        {
          auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
          ResultScalarType tVirtualStressTimesWeightedNormal(0.);
          for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++)
          {
            tVirtualStressTimesWeightedNormal += 
              tStressTensors(tCellOrdinal,aPointOrdinal,tDimI,tDimJ) * tWeightedNormalVector[tDimJ];
          }
          ResultScalarType tValue = aScale * tCubWeightOnParentBodyElemSurface * tVirtualStressTimesWeightedNormal
            * (tProjectedStates[tDimI] - tDirichletWS(tCellOrdinal,tLocalDofOrdinal));
          Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
        }
      }
    });
  }
};



template<typename EvaluationType>
class FactoryNitscheHyperElasticStressEvaluator
{
public:
  std::shared_ptr<Plato::BoundaryFluxEvaluator<EvaluationType>> 
  createTrialEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  )
  {
    if (!aParamList.isSublist("Material Models"))
    {
      ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
    }
    else
    {
      auto tMaterialName = this->getMaterialName(aNitscheParams);
      auto tMaterialModelParamList = aParamList.get<Teuchos::ParameterList>("Material Models");
      if (!tMaterialModelParamList.isSublist(tMaterialName))
      {
          auto tMsg = std::string("Requested a material model with name ('") + tMaterialName 
            + "') that is not defined in the input deck";
          ANALYZE_THROWERR(tMsg);
      }
      auto tMaterialParamList = tMaterialModelParamList.sublist(tMaterialName);
      if(tMaterialParamList.isSublist("Hyperelastic Kirchhoff")){
        auto tStressEvaluator = 
          std::make_shared<Plato::BoundaryEvaluatoryTrialKirchhoffStress<EvaluationType>>(
            aParamList,aNitscheParams
          );
        return tStressEvaluator;
      }
      else
      if(tMaterialParamList.isSublist("Hyperelastic Neo-Hookean")){
        auto tStressEvaluator = 
          std::make_shared<Plato::BoundaryEvaluatorTrialNeoHookeanStress<EvaluationType>>(
            aParamList,aNitscheParams
          );
        return tStressEvaluator;
      }
      else{
        Plato::Elliptic::mechanical::MaterialEnum tS2E;
        tS2E.get("Not Supported"); // throws
        return nullptr;
      }
    }
  }

  std::shared_ptr<Plato::BoundaryFluxEvaluator<EvaluationType>> 
  createTestEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  )
  {
    if (!aParamList.isSublist("Material Models"))
    {
      ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
    }
    else
    {
      auto tMaterialName = this->getMaterialName(aNitscheParams);
      auto tMaterialModelParamList = aParamList.get<Teuchos::ParameterList>("Material Models");
      if (!tMaterialModelParamList.isSublist(tMaterialName))
      {
          auto tMsg = std::string("Requested a material model with name ('") + tMaterialName 
            + "') that is not defined in the input deck";
          ANALYZE_THROWERR(tMsg);
      }
      auto tMaterialParamList = tMaterialModelParamList.sublist(tMaterialName);
      if(tMaterialParamList.isSublist("Hyperelastic Kirchhoff")){
        auto tStressEvaluator = 
          std::make_shared<Plato::BoundaryEvaluatorTestKirchhoffStress<EvaluationType>>(
            aParamList,aNitscheParams
          );
        return tStressEvaluator;
      }
      else
      if(tMaterialParamList.isSublist("Hyperelastic Neo-Hookean")){
        auto tStressEvaluator = 
          std::make_shared<Plato::BoundaryEvaluatorTestNeoHookeanStress<EvaluationType>>(
            aParamList,aNitscheParams
          );
        return tStressEvaluator;
      }
      else{
        Plato::Elliptic::mechanical::MaterialEnum tS2E;
        tS2E.get("Not Supported"); // throws
        return nullptr;
      }
    }
  }

  /// @fn initialize
  /// @brief return material name
  /// @param [in] aParamList input problem parameters
  /// @return string
  std::string 
  getMaterialName(
    const Teuchos::ParameterList & aParamList
  )
  {
    if( !aParamList.isParameter("Material Model") ){
      ANALYZE_THROWERR( std::string("ERROR: Input argument ('Material Model') is not defined, ") + 
        "material constitutive model for Nitsche's method cannot be determined" )
    }
    auto tMaterialName = aParamList.get<std::string>("Material Model");
    return (tMaterialName);
  }
};


template<typename EvaluationType>
class NitscheTrialHyperElasticStressEvaluator : public Plato::NitscheEvaluator
{
private:
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of degrees of freedom per parent body element vertex/node
  static constexpr auto mNumDofsPerNode = BodyElementBase::mNumDofsPerNode;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of nodes per parent body element surface
  static constexpr auto mNumNodesPerFace = BodyElementBase::mNumNodesPerFace;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to the material constitutive model
  using BaseClassType::mMaterialName;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<BodyElementBase,StateScalarType,ConfigScalarType>;
  /// @brief evaluates boundary trial and test stress tensors
  std::shared_ptr<Plato::BoundaryFluxEvaluator<EvaluationType>> mBoundaryStressEvaluator;

public:
  NitscheTrialHyperElasticStressEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // create boundary stress evaluator
    //
    Plato::Elliptic::FactoryNitscheHyperElasticStressEvaluator<EvaluationType> tFactory;
    mBoundaryStressEvaluator = tFactory.createTrialEvaluator(aParamList,aNitscheParams);
  }
  ~NitscheTrialHyperElasticStressEvaluator()
  {}

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  )
  {
    // unpack worksets
    //
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    // create local functors
    //
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
    Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // evaluate boundary trial stress tensors
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Plato::ScalarArray4DT<ResultScalarType> tStressTensors(
      "stress tensors",tNumCellsOnSideSet,mNumGaussPointsPerFace,mNumSpatialDims,mNumSpatialDims
    );
    mBoundaryStressEvaluator->evaluate(aSpatialModel,aWorkSets,tStressTensors,aCycle);
    // evaluate integral
    //
    Kokkos::parallel_for("nitsche stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      auto tCubPointOnParentFaceElem = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsOnParentFaceElem = FaceElementBase::basisGrads(tCubPointOnParentFaceElem);
      // quadrature data to evaluate integral on the body surface of interest
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      auto tCubWeightOnParentBodyElemSurface = tCubWeightsOnParentBodyElemSurface(aPointOrdinal);
      auto tBasisValuesOnParentBodyElemSurface = BodyElementBase::basisValues(tCubPointOnParentBodyElemSurface);
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute surface area weighted normal vector
      //
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Array<mNumSpatialDims, ConfigScalarType> tWeightedNormalVector;
      tComputeWeightedNormalVector(
        tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tWeightedNormalVector
      );
      // compute gradient of interpolation functions
      //
      ConfigScalarType tVolume(0.0);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
      tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
      // compute state gradient
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tStateGradient(StrainScalarType(0.));
      tComputeStateGradient(tCellOrdinal,tStateWS,tGradient,tStateGradient);
      // compute deformation gradient 
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tDefGradient(StrainScalarType(0.));
      tComputeDeformationGradient(tStateGradient,tDefGradient);
      // compute nominal stress
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ResultScalarType> 
        tNominalStress(ResultScalarType(0.));
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
          for(Plato::OrdinalType tDimK = 0; tDimK < mNumSpatialDims; tDimK++){
            tNominalStress(tDimI,tDimJ) += 
              tStressTensors(tCellOrdinal,aPointOrdinal,tDimI,tDimK) * tDefGradient(tDimJ,tDimK);
          }
        }
      }
      // evaluate: int_{\Gamma_D} \delta{u}\cdot(\mathbf{P}\cdot{n}) d\Gamma_D
      //
      for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++){
        for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++){
          auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
          ResultScalarType tStressTimesSurfaceWeightedNormal(0.0);
          for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++){
            tStressTimesSurfaceWeightedNormal += tNominalStress(tDimI,tDimJ) * tWeightedNormalVector[tDimJ];
          }
          ResultScalarType tValue = -aScale * tBasisValuesOnParentBodyElemSurface(tNode) 
            * tCubWeightOnParentBodyElemSurface * tStressTimesSurfaceWeightedNormal;
          Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
        }
      }
    });
  }
};

template<typename EvaluationType>
class NitscheTestHyperElasticStressEvaluator : public Plato::NitscheEvaluator
{
private:
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of degrees of freedom per parent body element vertex/node
  static constexpr auto mNumDofsPerNode = BodyElementBase::mNumDofsPerNode;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of nodes per parent body element surface
  static constexpr auto mNumNodesPerFace = BodyElementBase::mNumNodesPerFace;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to the material constitutive model
  using BaseClassType::mMaterialName;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<BodyElementBase,StateScalarType,ConfigScalarType>;
  /// @brief evaluates boundary trial and test stress tensors
  std::shared_ptr<Plato::BoundaryFluxEvaluator<EvaluationType>> mBoundaryStressEvaluator;

public:
  NitscheTestHyperElasticStressEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // create boundary stress evaluator
    //
    Plato::Elliptic::FactoryNitscheHyperElasticStressEvaluator<EvaluationType> tFactory;
    mBoundaryStressEvaluator = tFactory.createTestEvaluator(aParamList,aNitscheParams);
  }
  ~NitscheTestHyperElasticStressEvaluator()
  {}

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  )
  {
    // unpack worksets
    //
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    // create local functors
    //
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
    Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // evaluate boundary trial stress tensors
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Plato::ScalarArray4DT<ResultScalarType> tStressTensors(
      "stress tensors",tNumCellsOnSideSet,mNumGaussPointsPerFace,mNumSpatialDims,mNumSpatialDims
    );
    mBoundaryStressEvaluator->evaluate(aSpatialModel,aWorkSets,tStressTensors,aCycle);
    // evaluate integral
    //
    Kokkos::parallel_for("nitsche stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      auto tCubPointOnParentFaceElem = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsOnParentFaceElem = FaceElementBase::basisGrads(tCubPointOnParentFaceElem);
      // quadrature data to evaluate integral on the body surface of interest
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      auto tCubWeightOnParentBodyElemSurface = tCubWeightsOnParentBodyElemSurface(aPointOrdinal);
      auto tBasisValuesOnParentBodyElemSurface = BodyElementBase::basisValues(tCubPointOnParentBodyElemSurface);
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute surface area weighted normal vector
      //
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Array<mNumSpatialDims, ConfigScalarType> tWeightedNormalVector;
      tComputeWeightedNormalVector(
        tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tWeightedNormalVector
      );
      // compute gradient of interpolation functions
      //
      ConfigScalarType tVolume(0.0);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
      tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
      // compute state gradient
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tStateGradient(StrainScalarType(0.));
      tComputeStateGradient(tCellOrdinal,tGradient,tStateGradient);
      // compute deformation gradient 
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tDefGradient(StrainScalarType(0.));
      tComputeDeformationGradient(tStateGradient,tDefGradient);
      // compute nominal stress
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ResultScalarType> 
        tNominalStress(ResultScalarType(0.));
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
          for(Plato::OrdinalType tDimK = 0; tDimK < mNumSpatialDims; tDimK++){
            tNominalStress(tDimI,tDimJ) += 
              tStressTensors(tCellOrdinal,aPointOrdinal,tDimI,tDimK) * tDefGradient(tDimJ,tDimK);
          }
        }
      }
      // evaluate: int_{\Gamma_D} \delta{u}\cdot(\mathbf{P}\cdot{n}) d\Gamma_D
      //
      for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++){
        for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++){
          auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
          ResultScalarType tStressTimesSurfaceWeightedNormal(0.0);
          for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++){
            tStressTimesSurfaceWeightedNormal += tNominalStress(tDimI,tDimJ) * tWeightedNormalVector[tDimJ];
          }
          ResultScalarType tValue = -aScale * tBasisValuesOnParentBodyElemSurface(tNode) 
            * tCubWeightOnParentBodyElemSurface * tStressTimesSurfaceWeightedNormal;
          Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
        }
      }
    });
  }
};



template<typename EvaluationType>
class NitscheTrialThermalHyperElasticStressEvaluator : public Plato::NitscheEvaluator
{
private:
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of degrees of freedom per parent body element vertex/node
  static constexpr auto mNumDofsPerNode = BodyElementBase::mNumDofsPerNode;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of nodes per parent body element surface
  static constexpr auto mNumNodesPerFace = BodyElementBase::mNumNodesPerFace;
  /// @brief number of temperature degrees of freedom per node
  static constexpr auto mNumNodeStatePerNode = BodyElementBase::mNumNodeStatePerNode;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to the material constitutive model
  using BaseClassType::mMaterialName;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType     = typename EvaluationType::StateScalarType;
  using ResultScalarType    = typename EvaluationType::ResultScalarType;
  using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
  using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
  using StrainScalarType    = typename Plato::fad_type_t<BodyElementBase,StateScalarType,ConfigScalarType>;
  /// @brief input problem parameters
  Teuchos::ParameterList & mParamList;
  /// @brief evaluates boundary trial and test stress tensors
  std::shared_ptr<Plato::BoundaryFluxEvaluator<EvaluationType>> mBoundaryStressEvaluator;

public:
  NitscheTrialThermalHyperElasticStressEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams),
    mParamList(aParamList)
  {
    // create boundary stress evaluator
    //
    Plato::Elliptic::FactoryNitscheHyperElasticStressEvaluator<EvaluationType> tFactory;
    mBoundaryStressEvaluator = tFactory.createTrialEvaluator(aParamList,aNitscheParams);
  }
  ~NitscheTrialThermalHyperElasticStressEvaluator()
  {}

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  )
  {
    // unpack worksets
    //
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
    Plato::ScalarMultiVectorT<NodeStateScalarType> tTempWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<NodeStateScalarType>>(aWorkSets.get("node states"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    // create local functors
    //
    Plato::StateGradient<EvaluationType> tComputeDispGradient;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::NominalStressTensor<EvaluationType> tComputeNominalStressTensor;
    Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
    Plato::KineticPullBackOperation<EvaluationType> tApplyKineticPullBackOperation;
    Plato::DeformationGradient<EvaluationType> tComputeMechanicalDeformationGradient;
    Plato::InterpolateFromNodal<BodyElementBase,mNumNodeStatePerNode> tInterpolateFromNodal;
    Plato::ThermoElasticDeformationGradient<EvaluationType> tComputeThermoElasticDeformationGradient;
    Plato::ThermalDeformationGradient<EvaluationType> tComputeThermalDeformationGradient(mMaterialName,mParamList);
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // evaluate boundary trial stress tensors
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Plato::ScalarArray4DT<ResultScalarType> tWorkset2PKS(
      "stress tensors",tNumCellsOnSideSet,mNumGaussPointsPerFace,mNumSpatialDims,mNumSpatialDims
    );
    mBoundaryStressEvaluator->evaluate(aSpatialModel,aWorkSets,tWorkset2PKS,aCycle);
    // evaluate integral
    //
    Kokkos::parallel_for("nitsche stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      auto tCubPointOnParentFaceElem = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsOnParentFaceElem = FaceElementBase::basisGrads(tCubPointOnParentFaceElem);
      // quadrature data to evaluate integral on the body surface of interest
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      auto tCubWeightOnParentBodyElemSurface = tCubWeightsOnParentBodyElemSurface(aPointOrdinal);
      auto tBasisValuesOnParentBodyElemSurface = BodyElementBase::basisValues(tCubPointOnParentBodyElemSurface);
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute surface area weighted normal vector
      //
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Array<mNumSpatialDims, ConfigScalarType> tWeightedNormalVector;
      tComputeWeightedNormalVector(
        tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tWeightedNormalVector
      );
      // compute gradient of interpolation functions
      //
      ConfigScalarType tVolume(0.0);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
      tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
      // compute displacement gradient
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> tDispGradient(StrainScalarType(0.));
      tComputeDispGradient(tCellOrdinal,tStateWS,tGradient,tDispGradient);
      // compute mechanical deformation gradient 
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> tMechanicalDefGradient(StrainScalarType(0.));
      tComputeMechanicalDeformationGradient(tDispGradient,tMechanicalDefGradient);
      // interpolate temperature field from nodes to integration points on the parent body element surface
      NodeStateScalarType tTemperature = 
        tInterpolateFromNodal(tCellOrdinal,tBasisValuesOnParentBodyElemSurface,tTempWS);
      // compute thermal deformation gradient 
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,NodeStateScalarType> tThermalDefGradient(NodeStateScalarType(0.));
      tComputeThermalDeformationGradient(tTemperature,tThermalDefGradient);
      // compute multiplicative decomposition of the thermo-elastic deformation gradient 
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tThermoElasticDefGradient(ResultScalarType(0.));
      tComputeThermoElasticDeformationGradient(tThermalDefGradient,tMechanicalDefGradient,tThermoElasticDefGradient);
      // pull back second Piola-Kirchhoff stress from deformed to undeformed configuration
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tDefConfig2PKS(ResultScalarType(0.));
      Plato::Elliptic::getCell2PKS<mNumSpatialDims>(tCellOrdinal,aPointOrdinal,tWorkset2PKS,tDefConfig2PKS);
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tUnDefConfig2PKS(ResultScalarType(0.));
      tApplyKineticPullBackOperation(tThermalDefGradient,tDefConfig2PKS,tUnDefConfig2PKS);
      // compute nominal stress
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tNominalStressTensor(ResultScalarType(0.));
      tComputeNominalStressTensor(tThermoElasticDefGradient,tUnDefConfig2PKS,tNominalStressTensor);
      // evaluate: int_{\Gamma_D} \delta{u}\cdot(\mathbf{P}\cdot{n}) d\Gamma_D
      //
      for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++){
        for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++){
          auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
          ResultScalarType tStressTimesSurfaceWeightedNormal(0.0);
          for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++){
            tStressTimesSurfaceWeightedNormal += 
              tNominalStressTensor(tDimI,tDimJ) * tWeightedNormalVector[tDimJ];
          }
          ResultScalarType tValue = -aScale * tBasisValuesOnParentBodyElemSurface(tNode) 
            * tCubWeightOnParentBodyElemSurface * tStressTimesSurfaceWeightedNormal;
          Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
        }
      }
    });
  }
};


template<typename EvaluationType>
class NitscheTestThermalHyperElasticStressEvaluator : public Plato::NitscheEvaluator
{
private:
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of degrees of freedom per parent body element vertex/node
  static constexpr auto mNumDofsPerNode = BodyElementBase::mNumDofsPerNode;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of nodes per parent body element surface
  static constexpr auto mNumNodesPerFace = BodyElementBase::mNumNodesPerFace;
  /// @brief number of temperature degrees of freedom per node
  static constexpr auto mNumNodeStatePerNode = BodyElementBase::mNumNodeStatePerNode;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;  
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to the material constitutive model
  using BaseClassType::mMaterialName;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType     = typename EvaluationType::StateScalarType;
  using ResultScalarType    = typename EvaluationType::ResultScalarType;
  using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
  using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
  using StrainScalarType    = typename Plato::fad_type_t<BodyElementBase,StateScalarType,ConfigScalarType>;
  /// @brief input problem parameters
  Teuchos::ParameterList & mParamList;
  /// @brief evaluates boundary trial and test stress tensors
  std::shared_ptr<Plato::BoundaryFluxEvaluator<EvaluationType>> mBoundaryStressEvaluator;

public:
  NitscheTestThermalHyperElasticStressEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams),
    mParamList(aParamList)
  {
    // create boundary stress evaluator
    //
    Plato::Elliptic::FactoryNitscheHyperElasticStressEvaluator<EvaluationType> tFactory;
    mBoundaryStressEvaluator = tFactory.createTrialEvaluator(aParamList,aNitscheParams);
  }
  ~NitscheTestThermalHyperElasticStressEvaluator()
  {}

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  )
  {
    // unpack worksets
    //
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    // create local functors
    //
    Plato::StateGradient<EvaluationType> tComputeDispGradient;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::NominalStressTensor<EvaluationType> tComputeNominalStressTensor;
    Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
    Plato::KineticPullBackOperation<EvaluationType> tApplyKineticPullBackOperation;
    Plato::DeformationGradient<EvaluationType> tComputeMechanicalDeformationGradient;
    Plato::InterpolateFromNodal<BodyElementBase,mNumNodeStatePerNode> tInterpolateFromNodal;
    Plato::ThermoElasticDeformationGradient<EvaluationType> tComputeThermoElasticDeformationGradient;
    Plato::ThermalDeformationGradient<EvaluationType> tComputeThermalDeformationGradient(mMaterialName,mParamList);
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // evaluate boundary trial stress tensors
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Plato::ScalarArray4DT<ResultScalarType> tWorkset2PKS(
      "stress tensors",tNumCellsOnSideSet,mNumGaussPointsPerFace,mNumSpatialDims,mNumSpatialDims
    );
    mBoundaryStressEvaluator->evaluate(aSpatialModel,aWorkSets,tWorkset2PKS,aCycle);
    // evaluate integral
    //
    Kokkos::parallel_for("nitsche stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      auto tCubPointOnParentFaceElem = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsOnParentFaceElem = FaceElementBase::basisGrads(tCubPointOnParentFaceElem);
      // quadrature data to evaluate integral on the body surface of interest
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      auto tCubWeightOnParentBodyElemSurface = tCubWeightsOnParentBodyElemSurface(aPointOrdinal);
      auto tBasisValuesOnParentBodyElemSurface = BodyElementBase::basisValues(tCubPointOnParentBodyElemSurface);
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute surface area weighted normal vector
      //
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Array<mNumSpatialDims, ConfigScalarType> tWeightedNormalVector;
      tComputeWeightedNormalVector(
        tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tWeightedNormalVector
      );
      // compute gradient of interpolation functions
      //
      ConfigScalarType tVolume(0.0);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
      tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
      // compute displacement gradient
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> tDispGradient(StrainScalarType(0.));
      tComputeDispGradient(tCellOrdinal,tGradient,tDispGradient);
      // compute mechanical deformation gradient 
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> tMechanicalDefGradient(StrainScalarType(0.));
      tComputeMechanicalDeformationGradient(tDispGradient,tMechanicalDefGradient);
      // interpolate temperature field from nodes to integration points on the parent body element surface
      Plato::Scalar tTemperature = tInterpolateFromNodal(tCellOrdinal,tBasisValuesOnParentBodyElemSurface);
      // compute thermal deformation gradient 
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,NodeStateScalarType> tThermalDefGradient(NodeStateScalarType(0.));
      tComputeThermalDeformationGradient(tTemperature,tThermalDefGradient);
      // compute multiplicative decomposition of the thermo-elastic deformation gradient 
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tThermoElasticDefGradient(ResultScalarType(0.));
      tComputeThermoElasticDeformationGradient(tThermalDefGradient,tMechanicalDefGradient,tThermoElasticDefGradient);
      // pull back second Piola-Kirchhoff stress from deformed to undeformed configuration
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tDefConfig2PKS(ResultScalarType(0.));
      Plato::Elliptic::getCell2PKS<mNumSpatialDims>(tCellOrdinal,aPointOrdinal,tWorkset2PKS,tDefConfig2PKS);
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tUnDefConfig2PKS(ResultScalarType(0.));
      tApplyKineticPullBackOperation(tThermalDefGradient,tDefConfig2PKS,tUnDefConfig2PKS);
      // compute nominal stress
      //
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tNominalStressTensor(ResultScalarType(0.));
      tComputeNominalStressTensor(tThermoElasticDefGradient,tUnDefConfig2PKS,tNominalStressTensor);
      // evaluate: int_{\Gamma_D} \delta{u}\cdot(\mathbf{P}\cdot{n}) d\Gamma_D
      //
      for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++){
        for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++){
          auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
          ResultScalarType tStressTimesSurfaceWeightedNormal(0.0);
          for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++){
            tStressTimesSurfaceWeightedNormal += 
              tNominalStressTensor(tDimI,tDimJ) * tWeightedNormalVector[tDimJ];
          }
          ResultScalarType tValue = -aScale * tBasisValuesOnParentBodyElemSurface(tNode) 
            * tCubWeightOnParentBodyElemSurface * tStressTimesSurfaceWeightedNormal;
          Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
        }
      }
    });
  }
};


template<typename EvaluationType>
class NitscheDispMisfitEvaluator : public Plato::NitscheEvaluator
{
private:
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of degrees of freedom per parent body element vertex/node
  static constexpr auto mNumDofsPerNode = BodyElementBase::mNumDofsPerNode;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of nodes per parent body element surface
  static constexpr auto mNumNodesPerFace = BodyElementBase::mNumNodesPerFace;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<BodyElementBase,StateScalarType,ConfigScalarType>;
  /// @brief penalty for nitsche's method
  Plato::Scalar mNitschePenalty = 1.0;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;

public:
  NitscheDispMisfitEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // create material constitutive model
    //
    Plato::FactoryMechanicalMaterials<EvaluationType> tFactory;
    mMaterialModel = tFactory.create(mMaterialName,aParamList);
    // parse penalty parameter
    //
    if(aParamList.isType<Plato::Scalar>("Penalty")){
      mNitschePenalty = aParamList.get<Plato::Scalar>("Penalty");
    }
  }

  ~NitscheDispMisfitEvaluator()
  {}

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
        Plato::Scalar           aCycle = 0.0,
        Plato::Scalar           aScale = 1.0
  )
  {
    // unpack worksets
    //
    Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    Plato::ScalarMultiVector tDirichletWS = 
      Plato::unpack<Plato::ScalarMultiVector>(aWorkSets.get("dirichlet"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // compute characteristic length
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Plato::ComputeCharacteristicLength<EvaluationType> tComputeCharacteristicLength(mSideSetName);
    Plato::ScalarVectorT<ConfigScalarType> tCharacteristicLength("characteristic length",tNumCellsOnSideSet);
    tComputeCharacteristicLength(aSpatialModel, aWorkSets, tCharacteristicLength);
    // compute numerator for nitsche's penalty parameter
    //
    auto tYoungsModulus  = mMaterialModel->getScalarConstant("youngs modulus");
    auto tNitschePenaltyTimesModulus = mNitschePenalty * tYoungsModulus;
    // evaluate integral
    //
    Plato::SurfaceArea<BodyElementBase> tComputeSurfaceArea;
    Kokkos::parallel_for("nitsche stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      auto tCubPointOnParentFaceElem = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsOnParentFaceElem = FaceElementBase::basisGrads(tCubPointOnParentFaceElem);
      // quadrature data to evaluate integral on the body surface of interest
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      auto tCubWeightOnParentBodyElemSurface = tCubWeightsOnParentBodyElemSurface(aPointOrdinal);
      auto tBasisValuesOnParentBodyElemSurface = BodyElementBase::basisValues(tCubPointOnParentBodyElemSurface);
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute surface area
      //
      ConfigScalarType tSurfaceArea(0.0);
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      tComputeSurfaceArea(tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tSurfaceArea);
      // project state from nodes to quadrature/cubature point
      //
      Plato::Array<mNumSpatialDims,StateScalarType> tProjectedStates;
      for(Plato::OrdinalType tDof = 0; tDof < mNumDofsPerNode; tDof++)
      {
        tProjectedStates(tDof) = 0.0;
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
          Plato::OrdinalType tCellDofIndex = (mNumDofsPerNode * tNodeIndex) + tDof;
          tProjectedStates(tDof) += tStateWS(tCellOrdinal, tCellDofIndex) * 
            tBasisValuesOnParentBodyElemSurface(tNodeIndex);
        }
      }
      // evaluate int_{\Gamma_D}\gamma_N^u \delta{u}\cdot(u - u_D) d\Gamma_D
      //
      ConfigScalarType tGamma = tNitschePenaltyTimesModulus / tCharacteristicLength(aSideOrdinal);
      for(Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++)
      {
        for(Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
        {
          auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
          ResultScalarType tValue = aScale * tGamma * tBasisValuesOnParentBodyElemSurface(tNode)
            * ( tProjectedStates[tDimI] - tDirichletWS(tCellOrdinal,tLocalDofOrdinal) ) 
            * tSurfaceArea * tCubWeightOnParentBodyElemSurface;
          Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
        }
      }
    });
  }
};

template<typename EvaluationType>
class NitscheTrialHeatFluxEvaluator : public Plato::NitscheEvaluator
{
private:
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of degrees of freedom per parent body element vertex/node
  static constexpr auto mNumDofsPerNode = BodyElementBase::mNumDofsPerNode;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of nodes per parent body element surface
  static constexpr auto mNumNodesPerFace = BodyElementBase::mNumNodesPerFace;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to material constitutive model
  using BaseClassType::mMaterialName;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using GradScalarType   = typename Plato::fad_type_t<BodyElementBase,StateScalarType,ConfigScalarType>;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;

public:
  NitscheTrialHeatFluxEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // create material constitutive model
    //
    Plato::FactoryThermalConductionMaterial<EvaluationType> tFactory(aParamList);
    mMaterialModel = tFactory.create(mMaterialName);
  }

  ~NitscheTrialHeatFluxEvaluator()
  {}

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  )
  {
    // unpack worksets
    //
    Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // create local functors
    //
    Plato::ScalarGrad<BodyElementBase> tScalarGrad;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
    Plato::ThermalFlux<EvaluationType> tThermalFlux(mMaterialModel);
    Plato::InterpolateFromNodal<BodyElementBase,mNumDofsPerNode> tProjectFromNodes;
    // evaluate integral
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Kokkos::parallel_for("evaluate integral", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      auto tCubPointOnParentFaceElem = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsOnParentFaceElem = FaceElementBase::basisGrads(tCubPointOnParentFaceElem);
      // quadrature data to evaluate integral on the body surface of interest
      //
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      auto tCubWeightOnParentBodyElemSurface = tCubWeightsOnParentBodyElemSurface(aPointOrdinal);
      auto tBasisValuesOnParentBodyElemSurface = BodyElementBase::basisValues(tCubPointOnParentBodyElemSurface);
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute surface area weighted normal vector
      //
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Array<mNumSpatialDims,ConfigScalarType> tWeightedNormalVector;
      tComputeWeightedNormalVector(
        tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tWeightedNormalVector
      );
      // compute configuration gradient
      //
      ConfigScalarType tVolume(0.0);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient(ConfigScalarType(0.));
      tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
      // compute temperature gradient
      //
      Plato::Array <mNumSpatialDims,GradScalarType> tTempGrad(GradScalarType(0.));
      tScalarGrad(tCellOrdinal,tTempGrad,tStateWS,tGradient);
      // compute heat flux
      //
      StateScalarType tProjectedTemp =
        tProjectFromNodes(tCellOrdinal,tBasisValuesOnParentBodyElemSurface,tStateWS);
      Plato::Array <mNumSpatialDims,ResultScalarType> tFlux(ResultScalarType(0.));
      tThermalFlux(tFlux,tTempGrad,tProjectedTemp);
      // evaluate: int_{\Gamma_D} \delta{T}\cdot(q\cdot{n}) d\Gamma_D
      //
      for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++)
      {
        ResultScalarType tValue(0.0);
        Plato::OrdinalType tLocalDofOrdinal = tNode * mNumDofsPerNode;
        for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
        {
          tValue += -aScale * tBasisValuesOnParentBodyElemSurface(tNode) * tCubWeightOnParentBodyElemSurface
            * ( tFlux(tDimI) * tWeightedNormalVector(tDimI) );
        }
        Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal),tValue);
      }
    });
  }
};

template<typename EvaluationType>
class NitscheTestHeatFluxEvaluator : public Plato::NitscheEvaluator
{
private:
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of degrees of freedom per parent body element vertex/node
  static constexpr auto mNumDofsPerNode = BodyElementBase::mNumDofsPerNode;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of nodes per parent body element surface
  static constexpr auto mNumNodesPerFace = BodyElementBase::mNumNodesPerFace;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to material constitutive model
  using BaseClassType::mMaterialName;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using GradScalarType   = typename Plato::fad_type_t<BodyElementBase,StateScalarType,ConfigScalarType>;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;

public:
  NitscheTestHeatFluxEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // create material constitutive model
    //
    Plato::FactoryThermalConductionMaterial<EvaluationType> tFactory(aParamList);
    mMaterialModel = tFactory.create(mMaterialName);
  }

  ~NitscheTestHeatFluxEvaluator()
  {}

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  )
  {
    // unpack worksets
    //
    Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    Plato::ScalarMultiVector tDirichletWS = 
      Plato::unpack<Plato::ScalarMultiVector>(aWorkSets.get("dirichlet"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // create local functors
    //
    Plato::ScalarGrad<BodyElementBase> tScalarGrad;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
    Plato::ThermalFlux<EvaluationType> tThermalFlux(mMaterialModel);
    Plato::InterpolateFromNodal<BodyElementBase,mNumDofsPerNode> tProjectFromNodes;
    // evaluate integral
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Kokkos::parallel_for("evaluate integral", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      auto tCubPointOnParentFaceElem = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsOnParentFaceElem = FaceElementBase::basisGrads(tCubPointOnParentFaceElem);
      // quadrature data to evaluate integral on the body surface of interest
      //
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      auto tCubWeightOnParentBodyElemSurface = tCubWeightsOnParentBodyElemSurface(aPointOrdinal);
      auto tBasisValuesOnParentBodyElemSurface = BodyElementBase::basisValues(tCubPointOnParentBodyElemSurface);
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute surface area weighted normal vector
      //
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Array<mNumSpatialDims,ConfigScalarType> tWeightedNormalVector;
      tComputeWeightedNormalVector(
        tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tWeightedNormalVector
      );

      // compute configuration gradient
      //
      ConfigScalarType tVolume(0.0);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient(ConfigScalarType(0.));
      tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
      // compute temperature gradient
      //
      Plato::Array<mNumSpatialDims,ConfigScalarType> tVirtualTempGrad(ConfigScalarType(0.));
      tScalarGrad(tVirtualTempGrad,tGradient);
      // compute virtual heat flux
      //
      StateScalarType tProjectedTemp =
        tProjectFromNodes(tCellOrdinal,tBasisValuesOnParentBodyElemSurface,tStateWS);
      Plato::Array<mNumSpatialDims,ResultScalarType> tVirtualFlux(ResultScalarType(0.));
      tThermalFlux(tVirtualFlux,tVirtualTempGrad,tProjectedTemp);
      // evaluate: int_{\Gamma_D} \delta(q\cdot{n})\cdot(T - T_D) d\Gamma_D
      //
      for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++)
      {
        ResultScalarType tValue(0.0);
        Plato::OrdinalType tLocalDofOrdinal = tNode * mNumDofsPerNode;
        for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
        {
          tValue += aScale * tCubWeightOnParentBodyElemSurface
            * ( tVirtualFlux(tDimI) * tWeightedNormalVector(tDimI) )
            * ( tProjectedTemp - tDirichletWS(tCellOrdinal,tLocalDofOrdinal) );
        }
        Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal),tValue);
      }
    });
  }
};



template<typename EvaluationType>
class NitscheTempMisfitEvaluator : public Plato::NitscheEvaluator
{
private:
  /// @brief local topological parent body and face element typenames
  using BodyElementBase = typename EvaluationType::ElementType;
  using FaceElementBase = typename BodyElementBase::Face;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = BodyElementBase::mNumSpatialDims;
  /// @brief number of degrees of freedom per parent body element vertex/node
  static constexpr auto mNumDofsPerNode = BodyElementBase::mNumDofsPerNode;
  /// @brief number of nodes per parent body element
  static constexpr auto mNumNodesPerCell = BodyElementBase::mNumNodesPerCell;
  /// @brief number of nodes per parent body element surface
  static constexpr auto mNumNodesPerFace = BodyElementBase::mNumNodesPerFace;
  /// @brief number of integration points per parent body element surface
  static constexpr auto mNumGaussPointsPerFace = BodyElementBase::mNumGaussPointsPerFace;
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief side set name where dirichlet boundary conditions are 
  using BaseClassType::mSideSetName;
  /// @brief name assigned to material constitutive model
  using BaseClassType::mMaterialName;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using GradScalarType   = typename Plato::fad_type_t<BodyElementBase,StateScalarType,ConfigScalarType>;
  /// @brief penalty for nitsche's method
  Plato::Scalar mNitschePenalty = 1.0;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;

public:

public:
  NitscheTempMisfitEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // create material constitutive model
    //
    Plato::FactoryThermalConductionMaterial<EvaluationType> tFactory(aParamList);
    mMaterialModel = tFactory.create(mMaterialName);
    // parse penalty parameter
    //
    if(aParamList.isType<Plato::Scalar>("Penalty")){
      mNitschePenalty = aParamList.get<Plato::Scalar>("Penalty");
    }
  }

  ~NitscheTempMisfitEvaluator()
  {}

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  )
  {
    // unpack worksets
    //
    Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
    Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
      Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
    Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
      Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
    Plato::ScalarMultiVector tDirichletWS = 
      Plato::unpack<Plato::ScalarMultiVector>(aWorkSets.get("dirichlet"));
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
    // compute characteristic length
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Plato::ComputeCharacteristicLength<EvaluationType> tComputeCharacteristicLength(mSideSetName);
    Plato::ScalarVectorT<ConfigScalarType> tCharacteristicLength("characteristic length",tNumCellsOnSideSet);
    tComputeCharacteristicLength(aSpatialModel,aWorkSets,tCharacteristicLength);
    // create local functors
    //
    Plato::SurfaceArea<BodyElementBase> tComputeSurfaceArea;
    Plato::InterpolateFromNodal<BodyElementBase,mNumDofsPerNode> tProjectFromNodes;
    // evaluate integral
    //
    auto tNitschePenalty = mNitschePenalty;
    auto tConductivityTensor = mMaterialModel->getTensorConstant("Thermal Conductivity");
    Kokkos::parallel_for("evaluate integral", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      auto tCubPointOnParentFaceElem = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsOnParentFaceElem = FaceElementBase::basisGrads(tCubPointOnParentFaceElem);
      // quadrature data to evaluate integral on the body surface of interest
      //
      Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      auto tCubWeightOnParentBodyElemSurface = tCubWeightsOnParentBodyElemSurface(aPointOrdinal);
      auto tBasisValuesOnParentBodyElemSurface = BodyElementBase::basisValues(tCubPointOnParentBodyElemSurface);
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute surface area
      //
      ConfigScalarType tSurfaceArea(0.0);
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      tComputeSurfaceArea(tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tSurfaceArea);
      // project temperature from nodes to integration points
      //
      StateScalarType tProjectedTemp =
        tProjectFromNodes(tCellOrdinal,tBasisValuesOnParentBodyElemSurface,tStateWS);
      // evaluate: int_{\Gamma_D}\gamma_N^T \delta{T}\cdot(T - T_D) d\Gamma_D
      //
      for(Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++)
      {
        ResultScalarType tValue(0.0);
        Plato::OrdinalType tLocalDofOrdinal = tNode * mNumDofsPerNode;
        for(Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
        {
          ConfigScalarType tGamma = 
            ( tNitschePenalty * tConductivityTensor(tDimI,tDimI) ) / tCharacteristicLength(aSideOrdinal);
          tValue += aScale * tGamma * tBasisValuesOnParentBodyElemSurface(tNode)
            * ( tProjectedTemp - tDirichletWS(tCellOrdinal,tLocalDofOrdinal) )
            * tCubWeightOnParentBodyElemSurface * tSurfaceArea;
        }
        Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal),tValue);
      }
    });
  }
};

template<typename EvaluationType>
class NitscheLinearThermoStatics : public Plato::NitscheEvaluator
{
private:
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief list of nitsche boundary condition evaluators 
  std::vector<std::shared_ptr<Plato::NitscheEvaluator>> mEvaluators;

public:
  NitscheLinearThermoStatics(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // trial heat flux evaluator
    //
    mEvaluators.push_back(
      std::make_shared<Plato::Elliptic::NitscheTrialHeatFluxEvaluator<EvaluationType>>(
        aParamList,aNitscheParams
      )
    );
    // test heat flux evaluator
    //
    mEvaluators.push_back(
      std::make_shared<Plato::Elliptic::NitscheTestHeatFluxEvaluator<EvaluationType>>(
        aParamList,aNitscheParams
      )
    );
    // temperature misfit evaluator
    //
    mEvaluators.push_back(
      std::make_shared<Plato::Elliptic::NitscheTempMisfitEvaluator<EvaluationType>>(
        aParamList,aNitscheParams
      )
    );
  }

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  )
  {
    if(mEvaluators.empty()){
      ANALYZE_THROWERR( std::string("ERROR: Found an empty list of Nitsche evaluators, weak Dirichlet boundary " )
        + "conditions cannot be enforced" )
    }
    for(auto& tEvaluator : mEvaluators){
      tEvaluator->evaluate(aSpatialModel,aWorkSets,aCycle,aScale);
    }
  }
};

template<typename EvaluationType>
class NitscheElasticMechanics : public Plato::NitscheEvaluator
{
private:
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief list of nitsche boundary condition evaluators 
  std::vector<std::shared_ptr<Plato::NitscheEvaluator>> mEvaluators;

public:
  NitscheElasticMechanics(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // trial stress evaluator
    //
    mEvaluators.push_back(
      std::make_shared<Plato::Elliptic::NitscheTrialElasticStressEvaluator<EvaluationType>>(aParamList,aNitscheParams)
    );
    // test stress evaluator
    //
    mEvaluators.push_back(
      std::make_shared<Plato::Elliptic::NitscheTestElasticStressEvaluator<EvaluationType>>(aParamList,aNitscheParams)
    );
    // displacement misfit evaluator
    //
    mEvaluators.push_back(
      std::make_shared<Plato::Elliptic::NitscheDispMisfitEvaluator<EvaluationType>>(aParamList,aNitscheParams)
    );
  }

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  )
  {
    if(mEvaluators.empty()){
      ANALYZE_THROWERR( std::string("ERROR: Found an empty list of Nitsche evaluators, weak Dirichlet boundary " )
        + "conditions cannot be enforced" )
    }
    for(auto& tEvaluator : mEvaluators){
      tEvaluator->evaluate(aSpatialModel,aWorkSets,aCycle,aScale);
    }
  }
};

template<typename EvaluationType>
class NitscheHyperElasticMechanics : public Plato::NitscheEvaluator
{
private:
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief list of nitsche boundary condition evaluators 
  std::vector<std::shared_ptr<Plato::NitscheEvaluator>> mEvaluators;

public:
  NitscheHyperElasticMechanics(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // trial stress evaluator
    //
    mEvaluators.push_back(
      std::make_shared<Plato::Elliptic::NitscheTrialHyperElasticStressEvaluator<EvaluationType>>(
        aParamList,aNitscheParams
      )
    );
    // test stress evaluator
    //
    mEvaluators.push_back(
      std::make_shared<Plato::Elliptic::NitscheTestHyperElasticStressEvaluator<EvaluationType>>(
        aParamList,aNitscheParams
      )
    );
    // displacement misfit evaluator
    //
    mEvaluators.push_back(
      std::make_shared<Plato::Elliptic::NitscheDispMisfitEvaluator<EvaluationType>>(
        aParamList,aNitscheParams
      )
    );
  }

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  )
  {
    if(mEvaluators.empty()){
      ANALYZE_THROWERR( std::string("ERROR: Found an empty list of Nitsche evaluators, weak Dirichlet boundary " )
        + "conditions cannot be enforced" )
    }
    for(auto& tEvaluator : mEvaluators){
      tEvaluator->evaluate(aSpatialModel,aWorkSets,aCycle,aScale);
    }
  }
};

template<typename EvaluationType>
class NitscheHyperElasticThermoMechanics : public Plato::NitscheEvaluator
{
private:
  /// @brief local typename for base class
  using BaseClassType = Plato::NitscheEvaluator;
  /// @brief list of nitsche boundary condition evaluators 
  std::vector<std::shared_ptr<Plato::NitscheEvaluator>> mEvaluators;

public:
  NitscheHyperElasticThermoMechanics(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // trial stress evaluator
    //
    mEvaluators.push_back(
      std::make_shared<Plato::Elliptic::NitscheTrialThermalHyperElasticStressEvaluator<EvaluationType>>(
        aParamList,aNitscheParams
      )
    );
    // test stress evaluator
    //
    mEvaluators.push_back(
      std::make_shared<Plato::Elliptic::NitscheTestThermalHyperElasticStressEvaluator<EvaluationType>>(
        aParamList,aNitscheParams
      )
    );
    // displacement misfit evaluator
    //
    mEvaluators.push_back(
      std::make_shared<Plato::Elliptic::NitscheDispMisfitEvaluator<EvaluationType>>(
        aParamList,aNitscheParams
      )
    );
  }

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  )
  {
    if(mEvaluators.empty()){
      ANALYZE_THROWERR( std::string("ERROR: Found an empty list of Nitsche evaluators, weak Dirichlet boundary " )
        + "conditions cannot be enforced" )
    }
    for(auto& tEvaluator : mEvaluators){
      tEvaluator->evaluate(aSpatialModel,aWorkSets,aCycle,aScale);
    }
  }
};

template<typename EvaluationType>
class FactoryNitscheEvaluator
{
public:
  std::shared_ptr<Plato::NitscheEvaluator>
  create(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  )
  {
    if( !aParamList.isParameter("Physics") ){
      ANALYZE_THROWERR("ERROR: Argument ('Physics') is not defined, nitsche's evaluator cannot be created")
    }
    Plato::PhysicsEnum tS2E;
    auto tResponse = aParamList.get<std::string>("Response","Linear");
    auto tResponseEnum = tS2E.response(tResponse);
    switch (tResponseEnum)
    {
    case Plato::response_t::LINEAR:
      return ( this->createLinearNitscheEvaluator(aParamList,aNitscheParams) );
      break;
    case Plato::response_t::NONLINEAR:
      return ( this->createNonlinearNitscheEvaluator(aParamList,aNitscheParams) );
      break;
    default:
      ANALYZE_THROWERR(std::string("ERROR: Response '") + tResponse 
        + "' does not support weak enforcement of Dirichlet boundary conditions")
      break;
    }
  }

private:
  std::shared_ptr<Plato::NitscheEvaluator>
  createLinearNitscheEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  )
  {
    if( !aParamList.isParameter("Physics") ){
      ANALYZE_THROWERR("ERROR: Argument ('Physics') is not defined, nitsche's evaluator cannot be created")
    }
    Plato::PhysicsEnum tS2E;
    auto tPhysics = aParamList.get<std::string>("Physics");
    auto tPhysicsEnum = tS2E.physics(tPhysics);
    switch (tPhysicsEnum)
    {
    case Plato::physics_t::THERMAL:
      return ( std::make_shared<Plato::Elliptic::NitscheLinearThermoStatics<EvaluationType>>(aParamList,aNitscheParams) );
      break;
    case Plato::physics_t::MECHANICAL:
      return ( std::make_shared<Plato::Elliptic::NitscheElasticMechanics<EvaluationType>>(aParamList,aNitscheParams) );
      break;
    default:
      ANALYZE_THROWERR(std::string("ERROR: Physics '") + tPhysics 
        + "' does not support weak enforcement of Dirichlet boundary conditions")
      break;
    }
  }

  std::shared_ptr<Plato::NitscheEvaluator>
  createNonlinearNitscheEvaluator(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  )
  {
    if( !aParamList.isParameter("Physics") ){
      ANALYZE_THROWERR("ERROR: Argument ('Physics') is not defined, nitsche's evaluator cannot be created")
    }
    Plato::PhysicsEnum tS2E;
    auto tPhysics = aParamList.get<std::string>("Physics");
    auto tPhysicsEnum = tS2E.physics(tPhysics);
    switch (tPhysicsEnum)
    {
    case Plato::physics_t::MECHANICAL:
      return 
      ( 
        std::make_shared<Plato::Elliptic::NitscheHyperElasticMechanics<EvaluationType>>(
          aParamList,aNitscheParams) 
      );
      break;
    case Plato::physics_t::THERMOMECHANICAL:
      return 
      ( 
        std::make_shared<Plato::Elliptic::NitscheHyperElasticThermoMechanics<EvaluationType>>(
          aParamList,aNitscheParams) 
      );
      break;
    default:
      ANALYZE_THROWERR(std::string("ERROR: Physics '") + tPhysics 
        + "' does not support weak enforcement of Dirichlet boundary conditions")
      break;
    }
  }
};

/// @class NitscheBC
///
/// @brief weak enforcement of the diriechlet boundary conditions in linear mechanical problems with nitsche's method:
///
/// \[
///  -\int_{\Gamma_D}\delta\mathbf{u}\cdot\left(\sigma\cdot\mathbf{n}_{\Gamma}\right) d\Gamma
///  + \int_{\Gamma_D}\delta\left(\sigma\mathbf{n}_{\Gamma}\right)\cdot\left(\mathbf{u}-\mathbf{u}_{D}\right) d\Gamma
///  + \int_{\Gamma_D}\gamma_{N}^{\mathbf{u}}\delta\mathbf{u}\cdot\left(\mathbf{u}-\mathbf{u}_{D}\right) d\Gamma
/// \]
///
/// A non-symmetric nitsche formulation is considered in this work, see Burman (2012) and Schillinger et al. 
/// (2016a). The term \f$\mathbf{u}_D\f$ is the displacement imposed on boundary \f$\Gamma_D\f$. The parameter 
/// \f$\gamma_{N}^{\mathbf{u}}\f$ is chosen to reach a desired accuracy in satisfying the weakly enforce 
/// dirichlet boundary condition values.
///
/// @tparam EvaluationType automatic differentiation evaluation type, which sets the scalar types for the problem
template<typename EvaluationType>
class NitscheBC : public NitscheBase
{
private:
  std::shared_ptr<Plato::NitscheEvaluator> mNitscheBC;

public:
  NitscheBC(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aNitscheParams
  )
  {
    Plato::Elliptic::FactoryNitscheEvaluator<EvaluationType> tFactory;
    mNitscheBC = tFactory.create(aParamList,aNitscheParams);
  }
  ~NitscheBC(){}

  void 
  evaluate(
    const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  )
  {
    mNitscheBC->evaluate(aSpatialModel,aWorkSets,aCycle,aScale);
  }

};

} // namespace Elliptic

} // namespace Plato

namespace EllipticNitscheTests
{

TEUCHOS_UNIT_TEST( EllipticNitscheTests, MaterialIsotropicElastic )
{
  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                       \n"
    "<ParameterList name='Material Models'>                                   \n"
      "<ParameterList name='Unobtainium'>                                     \n"
        "<ParameterList name='Isotropic Linear Elastic'>                      \n"
          "<Parameter  name='Youngs Modulus'  type='double' value='1.0'/>     \n"
          "<Parameter  name='Poissons Ratio'  type='double' value='0.3'/>     \n"
        "</ParameterList>                                                     \n"
      "</ParameterList>                                                       \n"
    "</ParameterList>                                                         \n"
  "</ParameterList>                                                           \n"
  ); 

  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  Plato::FactoryElasticMaterial<Residual> tFactory(tParamList.operator*());

  auto tTol = 1e-6;
  auto tMaterialModel = tFactory.create("Unobtainium");
  auto tYoungsModulus = tMaterialModel->getScalarConstant("youngs Modulus");
  TEST_FLOATING_EQUALITY(1.0,tYoungsModulus,tTol);
  auto tPoissonsRatio = tMaterialModel->getScalarConstant("Poissons ratio");
  TEST_FLOATING_EQUALITY(0.3,tPoissonsRatio,tTol);
  auto tLameMu = tMaterialModel->getScalarConstant("MU");
  TEST_FLOATING_EQUALITY(0.3846153846153846,tLameMu,tTol);
  auto tLameLambda = tMaterialModel->getScalarConstant("lambda");
  TEST_FLOATING_EQUALITY(0.5769230769230769,tLameLambda,tTol);
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, StrainTensor )
{
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create compute strain functor
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  // create configuration workset 
  //
  auto tNumCells = tMesh->NumElements();
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  Plato::ScalarArray3DT<ConfigScalarType> 
    tConfigWS("Config Workset",tNumCells,ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims);
  tWorksetBase.worksetConfig(tConfigWS);
  // create displacement/state workset
  //
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  Plato::ScalarMultiVectorT<StateScalarType> 
    tStateWS("State Workset",tNumCells,ElementType::mNumDofsPerCell);
  tWorksetBase.worksetState(tDisp, tStateWS);
  // create local functors
  //
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::ComputeStrainTensor<Residual> tComputeStrainTensor;
  // create output worksets
  //
  Plato::ScalarArray4DT<ConfigScalarType> 
    tVirtualStrains("Virtual Strains",tNumCells,ElementType::mNumGaussPoints,tSpaceDim,tSpaceDim);
  Plato::ScalarArray4DT<StrainScalarType> 
    tStateStrains("State Strains",tNumCells,ElementType::mNumGaussPoints,tSpaceDim,tSpaceDim);
  // compute test and trial strains
  //
  Kokkos::parallel_for("evaluate", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCells, ElementType::mNumGaussPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & jGpOrdinal)
  {
    // compute gradients
    //
    ConfigScalarType tVolume(0.);
    Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigScalarType> tGradient;
    tComputeGradient(iCellOrdinal,jGpOrdinal,tConfigWS,tGradient,tVolume);
    // compute virtual strains
    //
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ConfigScalarType> 
      tVirtualStrainTensor(ConfigScalarType(0.0));
    tComputeStrainTensor(iCellOrdinal,tGradient,tVirtualStrainTensor);
    // compute state strains
    //
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> 
      tStrainTensor(StrainScalarType(0.0));
    tComputeStrainTensor(iCellOrdinal,tStateWS,tGradient,tStrainTensor);
    // set output worksets
    //
    for(Plato::OrdinalType tDimI=0; tDimI<ElementType::mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ=0; tDimJ<ElementType::mNumSpatialDims; tDimJ++){
        tStateStrains(iCellOrdinal,jGpOrdinal,tDimI,tDimJ)   = tStrainTensor(tDimI,tDimJ);
        tVirtualStrains(iCellOrdinal,jGpOrdinal,tDimI,tDimJ) = tVirtualStrainTensor(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tStrainGold = 
    { {0.4,0.3,0.3,0.2}, {0.4,0.3,0.3,0.2} };
  auto tHostStateStrains = Kokkos::create_mirror(tStateStrains);
  Kokkos::deep_copy(tHostStateStrains, tStateStrains);
  std::vector<std::vector<Plato::Scalar>> tVirtualStrainGold = 
    { {0.,0.,0.,0.}, {0.,0.,0.,0.} };
  auto tHostVirtualStrains = Kokkos::create_mirror(tVirtualStrains);
  Kokkos::deep_copy(tHostVirtualStrains, tVirtualStrains);
  //
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(
          tStrainGold[tCell][tDimI*tSpaceDim+tDimJ],tHostStateStrains(tCell,0,tDimI,tDimJ),tTol
        );
        TEST_FLOATING_EQUALITY(
          tVirtualStrainGold[tCell][tDimI*tSpaceDim+tDimJ],tHostVirtualStrains(tCell,0,tDimI,tDimJ),tTol
        );
      }
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, StressTensor )
{
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create compute strain functor
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  // create configuration workset 
  //
  auto tNumCells = tMesh->NumElements();
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  Plato::ScalarArray3DT<ConfigScalarType> 
    tConfigWS("Config Workset",tNumCells,ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims);
  tWorksetBase.worksetConfig(tConfigWS);
  // create displacement/state workset
  //
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  Plato::ScalarMultiVectorT<StateScalarType> 
    tStateWS("State Workset",tNumCells,ElementType::mNumDofsPerCell);
  tWorksetBase.worksetState(tDisp, tStateWS);
  // create material model
  //
  Plato::MaterialIsotropicElastic<Residual> tMaterialModel;
  tMaterialModel.mu(0.3846153846153846);
  tMaterialModel.lambda(0.5769230769230769);
  // create local functors
  //
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::ComputeStrainTensor<Residual> tComputeStrainTensor;
  Plato::ComputeIsotropicElasticStressTensor tComputeStressTensor(tMaterialModel);
  // create output worksets
  //
  Plato::ScalarArray4DT<StrainScalarType> 
    tStressTensors("Stress",tNumCells,ElementType::mNumGaussPoints,tSpaceDim,tSpaceDim);
  // compute test and trial strains
  //
  Kokkos::parallel_for("evaluate", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCells, ElementType::mNumGaussPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & jGpOrdinal)
  {
    // compute gradients
    //
    ConfigScalarType tVolume(0.);
    Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigScalarType> tGradient;
    tComputeGradient(iCellOrdinal,jGpOrdinal,tConfigWS,tGradient,tVolume);
    // compute state strains
    //
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> 
      tStrainTensor(StrainScalarType(0.0));
    tComputeStrainTensor(iCellOrdinal,tStateWS,tGradient,tStrainTensor);
    // compute stress tensor
    //
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultScalarType> 
      tStressTensor(ResultScalarType(0.0));
    tComputeStressTensor(tStrainTensor,tStressTensor);
    // set output worksets
    //
    for(Plato::OrdinalType tDimI=0; tDimI<ElementType::mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ=0; tDimJ<ElementType::mNumSpatialDims; tDimJ++){
        tStressTensors(iCellOrdinal,jGpOrdinal,tDimI,tDimJ) = tStressTensor(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tStressGold = 
    { {0.65384615,0.23076923,0.23076923,0.5}, {0.65384615,0.23076923,0.23076923,0.5} };
  auto tHostStressTensors = Kokkos::create_mirror(tStressTensors);
  Kokkos::deep_copy(tHostStressTensors, tStressTensors);
  //
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(
          tStressGold[tCell][tDimI*tSpaceDim+tDimJ],tHostStressTensors(tCell,0,tDimI,tDimJ),tTol
        );
      }
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, ComputeSideCellVolumes )
{
  // create inputs
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                                                      \n"
    "<ParameterList name='Spatial Model'>                                                                    \n"
      "<ParameterList name='Domains'>                                                                        \n"
        "<ParameterList name='Design Volume'>                                                                \n"
          "<Parameter name='Element Block' type='string' value='body'/>                                      \n"
          "<Parameter name='Material Model' type='string' value='Unobtainium'/>                              \n"
        "</ParameterList>                                                                                    \n"
      "</ParameterList>                                                                                      \n"
    "</ParameterList>                                                                                        \n"
  "</ParameterList>                                                                                          \n"
  ); 
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create compute strain functor
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create configuration workset 
  //
  auto tNumCells = tMesh->NumElements();
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  auto tConfigWS = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigScalarType> > >
    ( Plato::ScalarArray3DT<ConfigScalarType>(
        "Config Workset",tNumCells,ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims) 
    );
  tWorksetBase.worksetConfig(tConfigWS->mData);
  Plato::WorkSets tWorkSets;
  tWorkSets.set("configuration",tConfigWS);  
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  // compute volumes
  //
  auto tSideSetName = std::string("x+");
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  Plato::ComputeSideCellVolumes<Residual> tComputeSideCellVolumes(tSideSetName);
  Plato::ScalarVectorT<ConfigScalarType> tCellVolumes("volumes",tNumSideCells);
  tComputeSideCellVolumes(tSpatialModel,tWorkSets,tCellVolumes);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-4;
  std::vector<Plato::Scalar> tVolumeGold = { 0.5 };
  auto tHostCellVolumes = Kokkos::create_mirror(tCellVolumes);
  Kokkos::deep_copy(tHostCellVolumes, tCellVolumes);
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    TEST_FLOATING_EQUALITY(tVolumeGold[tCell],tHostCellVolumes(tCell),tTol);
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, ComputeSideCellFaceAreas )
{
  // create inputs
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                                                      \n"
    "<ParameterList name='Spatial Model'>                                                                    \n"
      "<ParameterList name='Domains'>                                                                        \n"
        "<ParameterList name='Design Volume'>                                                                \n"
          "<Parameter name='Element Block' type='string' value='body'/>                                      \n"
          "<Parameter name='Material Model' type='string' value='Unobtainium'/>                              \n"
        "</ParameterList>                                                                                    \n"
      "</ParameterList>                                                                                      \n"
    "</ParameterList>                                                                                        \n"
  "</ParameterList>                                                                                          \n"
  ); 
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create compute strain functor
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create configuration workset 
  //
  auto tNumCells = tMesh->NumElements();
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  auto tConfigWS = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigScalarType> > >
    ( Plato::ScalarArray3DT<ConfigScalarType>(
        "Config Workset",tNumCells,ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims) 
    );
  tWorksetBase.worksetConfig(tConfigWS->mData);
  Plato::WorkSets tWorkSets;
  tWorkSets.set("configuration",tConfigWS);  
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  // compute volumes
  //
  auto tSideSetName = std::string("x+");
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  Plato::ComputeSideCellFaceAreas<Residual> tComputeSideCellFaceAreas(tSideSetName);
  Plato::ScalarVectorT<ConfigScalarType> tCellAreas("areas",tNumSideCells);
  tComputeSideCellFaceAreas(tSpatialModel,tWorkSets,tCellAreas);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-4;
  std::vector<Plato::Scalar> tGoldAreas = { 1.0 };
  auto tHostCellAreas = Kokkos::create_mirror(tCellAreas);
  Kokkos::deep_copy(tHostCellAreas, tCellAreas);
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    TEST_FLOATING_EQUALITY(tGoldAreas[tCell],tHostCellAreas(tCell),tTol);
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, ComputeCharacteristicLength )
{
  // create inputs
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                                                      \n"
    "<ParameterList name='Spatial Model'>                                                                    \n"
      "<ParameterList name='Domains'>                                                                        \n"
        "<ParameterList name='Design Volume'>                                                                \n"
          "<Parameter name='Element Block' type='string' value='body'/>                                      \n"
          "<Parameter name='Material Model' type='string' value='Unobtainium'/>                              \n"
        "</ParameterList>                                                                                    \n"
      "</ParameterList>                                                                                      \n"
    "</ParameterList>                                                                                        \n"
  "</ParameterList>                                                                                          \n"
  ); 
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create compute strain functor
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create configuration workset 
  //
  auto tNumCells = tMesh->NumElements();
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  auto tConfigWS = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigScalarType> > >
    ( Plato::ScalarArray3DT<ConfigScalarType>(
        "Config Workset",tNumCells,ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims) 
    );
  tWorksetBase.worksetConfig(tConfigWS->mData);
  Plato::WorkSets tWorkSets;
  tWorkSets.set("configuration",tConfigWS);  
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  // compute volumes
  //
  auto tSideSetName = std::string("x+");
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  Plato::ComputeCharacteristicLength<Residual> tComputeCharacteristicLength(tSideSetName);
  Plato::ScalarVectorT<ConfigScalarType> tCharLength("areas",tNumSideCells);
  tComputeCharacteristicLength(tSpatialModel,tWorkSets,tCharLength);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-4;
  std::vector<Plato::Scalar> tGoldLength = { 0.5 };
  auto tHostCharLength = Kokkos::create_mirror(tCharLength);
  Kokkos::deep_copy(tHostCharLength, tCharLength);
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    TEST_FLOATING_EQUALITY(tGoldLength[tCell],tHostCharLength(tCell),tTol);
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTrialElasticStressEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                             \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                           \n"
    "    <ParameterList name='Domains'>                                               \n"
    "      <ParameterList name='Design Volume'>                                       \n"
    "        <Parameter name='Element Block' type='string' value='body'/>             \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Material Models'>                                         \n"
    "    <ParameterList name='Unobtainium'>                                           \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>           \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "</ParameterList>                                                                 \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create displacement workset
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tDisp);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  tWorksetBuilder.build(tOnlyDomainDefined,tDatabase,tWorkSets);
  // create results workset
  //
  auto tNumCells = tMesh->NumElements();
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("x+");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTrialElasticStressEvaluator<Residual> 
    tNitscheTrialElasticStressEvaluator(*tParamList,tNitscheParams);
  tNitscheTrialElasticStressEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.,0.,-57692.307692307702382,-125000,-57692.307692307702382,-125000} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTestElasticStressEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                             \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                           \n"
    "    <ParameterList name='Domains'>                                               \n"
    "      <ParameterList name='Design Volume'>                                       \n"
    "        <Parameter name='Element Block' type='string' value='body'/>             \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Material Models'>                                         \n"
    "    <ParameterList name='Unobtainium'>                                           \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>           \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "</ParameterList>                                                                 \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create displacement data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tDisp);
  // create dirichlet displacement data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumDofs);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("x+");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTestElasticStressEvaluator<Residual> 
    tNitscheTestElasticStressEvaluator(*tParamList,tNitscheParams);
  tNitscheTestElasticStressEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.,0.,0.,0.,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheDispMisfitEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                             \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                           \n"
    "    <ParameterList name='Domains'>                                               \n"
    "      <ParameterList name='Design Volume'>                                       \n"
    "        <Parameter name='Element Block' type='string' value='body'/>             \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Material Models'>                                         \n"
    "    <ParameterList name='Unobtainium'>                                           \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0'/>           \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "</ParameterList>                                                                 \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create displacement data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tDisp);
  // create dirichlet displacement data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumDofs);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("x+");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheDispMisfitEvaluator<Residual> tNitscheDispMisfitEvaluator(*tParamList,tNitscheParams);
  tNitscheDispMisfitEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.,0.,0.23333333,0.28333333,0.26666667,0.31666667} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, LinearMechanicalNitscheBC )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                             \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                           \n"
    "    <ParameterList name='Domains'>                                               \n"
    "      <ParameterList name='Design Volume'>                                       \n"
    "        <Parameter name='Element Block' type='string' value='body'/>             \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Material Models'>                                         \n"
    "    <ParameterList name='Unobtainium'>                                           \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0'/>           \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "</ParameterList>                                                                 \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create displacement data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tDisp);
  // create dirichlet displacement data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumDofs);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("x+");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheBC<Residual> tNitscheBC(*tParamList,tNitscheParams);
  tNitscheBC.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.,0.,0.175641025641026,0.158333333333333,0.208974358974359,0.191666666666667} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTrialHeatFluxEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics' type='string' value='Thermal'/>                       \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "  <ParameterList name='Material Models'>                                          \n"
    "    <ParameterList name='Unobtainium'>                                            \n"
    "      <ParameterList name='Thermal Conduction'>                                   \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0'/>        \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::ThermalElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 3);
  // create temperature data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tTemp);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTrialHeatFluxEvaluator<Residual> 
    tNitscheHeatFluxEvaluator(*tParamList,tNitscheParams);
  tNitscheHeatFluxEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.05,0.05,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  } 
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTestHeatFluxEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics' type='string' value='Thermal'/>                       \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "  <ParameterList name='Material Models'>                                          \n"
    "    <ParameterList name='Unobtainium'>                                            \n"
    "      <ParameterList name='Thermal Conduction'>                                   \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0'/>        \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::ThermalElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 3);
  // create temperature data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tTemp);
  // create dirichlet temperature data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumVerts);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTestHeatFluxEvaluator<Residual> 
    tNitscheVirtualHeatFluxEvaluator(*tParamList,tNitscheParams);
  tNitscheVirtualHeatFluxEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  } 
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheDispMisfitEvaluator2 )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                             \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                           \n"
    "    <ParameterList name='Domains'>                                               \n"
    "      <ParameterList name='Design Volume'>                                       \n"
    "        <Parameter name='Element Block' type='string' value='body'/>             \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Material Models'>                                         \n"
    "    <ParameterList name='Unobtainium'>                                           \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0'/>           \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "</ParameterList>                                                                 \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create displacement data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tDisp);
  // create dirichlet displacement data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumDofs);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheDispMisfitEvaluator<Residual> tNitscheDispMisfitEvaluator(*tParamList,tNitscheParams);
  tNitscheDispMisfitEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.066666666666667,0.116666666666667,0.133333333333333,0.183333333333333,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTempMisfitEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics' type='string' value='Thermal'/>                       \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "  <ParameterList name='Material Models'>                                          \n"
    "    <ParameterList name='Unobtainium'>                                            \n"
    "      <ParameterList name='Thermal Conduction'>                                   \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0'/>        \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::ThermalElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 3);
  // create temperature data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tTemp);
  // create dirichlet temperature data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumVerts);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTempMisfitEvaluator<Residual> 
    tNitscheTempMisfitEvaluator(*tParamList,tNitscheParams);
  tNitscheTempMisfitEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.066666666667,0.133333333333,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  } 
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, LinearThermalNitscheBC )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics' type='string' value='Thermal'/>                       \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "  <ParameterList name='Material Models'>                                          \n"
    "    <ParameterList name='Unobtainium'>                                            \n"
    "      <ParameterList name='Thermal Conduction'>                                   \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0'/>        \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::ThermalElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 3);
  // create temperature data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tTemp);
  // create dirichlet temperature data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumVerts);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheBC<Residual> tNitscheBC(*tParamList,tNitscheParams);
  tNitscheBC.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.11666666667,0.183333333333,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTrialHyperElasticStressEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics'  type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Response' type='string' value='Nonlinear'/>                    \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "<ParameterList name='Material Models'>                                            \n"
      "<ParameterList name='Unobtainium'>                                              \n"
        "<ParameterList name='Hyperelastic Kirchhoff'>                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>              \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>               \n"
        "</ParameterList>                                                              \n"
      "</ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 6);
  // create displacement data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tDisp);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTrialHyperElasticStressEvaluator<Residual> 
    tNitscheEvaluator(*tParamList,tNitscheParams);
  tNitscheEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {-0.40123454,-0.14197522, -0.40123454,-0.14197522,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTestHyperElasticStressEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics'  type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Response' type='string' value='Nonlinear'/>                    \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "<ParameterList name='Material Models'>                                            \n"
      "<ParameterList name='Unobtainium'>                                              \n"
        "<ParameterList name='Hyperelastic Kirchhoff'>                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>              \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>               \n"
        "</ParameterList>                                                              \n"
      "</ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 6);
  // create displacement data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tDisp);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTestHyperElasticStressEvaluator<Residual> 
    tNitscheEvaluator(*tParamList,tNitscheParams);
  tNitscheEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.,0.,0.,0.,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTrialThermalHyperElasticStressEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics'  type='string' value='Thermomechanical'/>             \n"
    "  <Parameter name='Response' type='string' value='Nonlinear'/>                    \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "<ParameterList name='Material Models'>                                            \n"
      "<ParameterList name='Unobtainium'>                                              \n"
        "<ParameterList name='Thermal Conduction'>                                     \n"
          "<Parameter  name='Thermal Expansivity'   type='double' value='0.5'/>        \n"
          "<Parameter  name='Reference Temperature' type='double' value='1.0'/>        \n"
          "<Parameter  name='Thermal Conductivity'  type='double' value='2.0'/>        \n"
        "</ParameterList>                                                              \n"
        "<ParameterList name='Hyperelastic Kirchhoff'>                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>              \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>               \n"
        "</ParameterList>                                                              \n"
      "</ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType         = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual            = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType     = typename Residual::StateScalarType;
  using ResultScalarType    = typename Residual::ResultScalarType;
  using ConfigScalarType    = typename Residual::ConfigScalarType;
  using NodeStateScalarType = typename Residual::NodeStateScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 6);
  // create displacement data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tDisp);
  // create temperature data
  //
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("node states",tTemp);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTrialThermalHyperElasticStressEvaluator<Residual> 
    tNitscheEvaluator(*tParamList,tNitscheParams);
  tNitscheEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {-0.213991754666667,-0.075720117333333,-0.227366239333333,-0.080452624666667,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTestThermalHyperElasticStressEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics'  type='string' value='Thermomechanical'/>             \n"
    "  <Parameter name='Response' type='string' value='Nonlinear'/>                    \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "<ParameterList name='Material Models'>                                            \n"
      "<ParameterList name='Unobtainium'>                                              \n"
        "<ParameterList name='Thermal Conduction'>                                     \n"
          "<Parameter  name='Thermal Expansivity'   type='double' value='0.5'/>        \n"
          "<Parameter  name='Reference Temperature' type='double' value='1.0'/>        \n"
          "<Parameter  name='Thermal Conductivity'  type='double' value='2.0'/>        \n"
        "</ParameterList>                                                              \n"
        "<ParameterList name='Hyperelastic Kirchhoff'>                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>              \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>               \n"
        "</ParameterList>                                                              \n"
      "</ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType         = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual            = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType     = typename Residual::StateScalarType;
  using ResultScalarType    = typename Residual::ResultScalarType;
  using ConfigScalarType    = typename Residual::ConfigScalarType;
  using NodeStateScalarType = typename Residual::NodeStateScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 6);
  // create displacement data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tDisp);
  // create temperature data
  //
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("node states",tTemp);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTestThermalHyperElasticStressEvaluator<Residual> 
    tNitscheEvaluator(*tParamList,tNitscheParams);
  tNitscheEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {-0.2765432,-0.0703703,-0.2765432,-0.0703703,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheHyperElasticMechanics )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics'  type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Response' type='string' value='Nonlinear'/>                    \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "<ParameterList name='Material Models'>                                            \n"
      "<ParameterList name='Unobtainium'>                                              \n"
        "<ParameterList name='Hyperelastic Kirchhoff'>                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>              \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>               \n"
        "</ParameterList>                                                              \n"
      "</ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 6);
  // create displacement data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tDisp);
  // create dirichlet displacement data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumDofs);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheHyperElasticMechanics<Residual> 
    tNitscheEvaluator(*tParamList,tNitscheParams);
  tNitscheEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {-0.334567873333333,-0.02530855333333301,-0.26790120666666695,0.041358113333332974,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheHyperElasticThermoMechanics )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics'  type='string' value='Thermomechanical'/>             \n"
    "  <Parameter name='Response' type='string' value='Nonlinear'/>                    \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "<ParameterList name='Material Models'>                                            \n"
      "<ParameterList name='Unobtainium'>                                              \n"
        "<ParameterList name='Hyperelastic Kirchhoff'>                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>              \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>               \n"
        "</ParameterList>                                                              \n"
        "<ParameterList name='Thermal Conduction'>                                     \n"
          "<Parameter  name='Thermal Expansivity'   type='double' value='0.5'/>        \n"
          "<Parameter  name='Reference Temperature' type='double' value='1.0'/>        \n"
          "<Parameter  name='Thermal Conductivity'  type='double' value='2.0'/>        \n"
        "</ParameterList>                                                              \n"
      "</ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 6);
  // create displacement data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tDisp);
  // create temperature data
  //
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("node states",tTemp);
  // create dirichlet displacement data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumDofs);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheHyperElasticThermoMechanics<Residual> 
    tNitscheEvaluator(*tParamList,tNitscheParams);
  tNitscheEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {-0.423868288,-0.029423750666666,-0.370576106,0.03251040866666599,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, FactoryNitscheEvaluator_NonlinearThermoMechanics )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics'  type='string' value='Thermomechanical'/>             \n"
    "  <Parameter name='Response' type='string' value='Nonlinear'/>                    \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "<ParameterList name='Material Models'>                                            \n"
      "<ParameterList name='Unobtainium'>                                              \n"
        "<ParameterList name='Hyperelastic Kirchhoff'>                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>              \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>               \n"
        "</ParameterList>                                                              \n"
        "<ParameterList name='Thermal Conduction'>                                     \n"
          "<Parameter  name='Thermal Expansivity'   type='double' value='0.5'/>        \n"
          "<Parameter  name='Reference Temperature' type='double' value='1.0'/>        \n"
          "<Parameter  name='Thermal Conductivity'  type='double' value='2.0'/>        \n"
        "</ParameterList>                                                              \n"
      "</ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 6);
  // create displacement data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tDisp);
  // create temperature data
  //
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("node states",tTemp);
  // create dirichlet displacement data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumDofs);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator
  //
  Plato::Elliptic::FactoryNitscheEvaluator<Residual> tFactory;
  auto tEvaluator = tFactory.create(*tParamList,tNitscheParams);
  tEvaluator->evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {-0.423868288,-0.029423750666666,-0.370576106,0.03251040866666599,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

} // namespace EllipticNitscheTests

