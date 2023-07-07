/*
 * LinearMechanicsNitscheTest.cpp
 *
 *  Created on: July 6, 2023
 */

/// @include trilinos includes
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

/// @include analyze includes
#include "FadTypes.hpp"
#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "SurfaceArea.hpp"
#include "SpatialModel.hpp"
#include "AnalyzeMacros.hpp"
#include "MaterialModel.hpp"
#include "GradientMatrix.hpp"
#include "WeightedNormalVector.hpp"
#include "InterpolateFromNodal.hpp"

/// @include analyze unit test includes
#include "util/PlatoTestHelpers.hpp"

namespace Plato
{

template<typename EvaluationType>
class MaterialIsotropicElastic : public MaterialModel<EvaluationType>
{
public:
  MaterialIsotropicElastic(
    const Teuchos::ParameterList& aParamList
  )
  {
    this->parse(aParamList);
    this->computeLameConstants();
  }

  Plato::Scalar 
  mu() 
  const
  { return this->getScalarConstant("mu"); }

  Plato::Scalar 
  lambda() 
  const
  { return this->getScalarConstant("lambda"); }

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
        return ( std::make_shared<MaterialIsotropicElastic<EvaluationType>>( 
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
      + "Supported material constitutive models for mechanical analysis are: ";
    for(const auto& tElement : mSupportedMaterials)
    {
      tMsg = tMsg + "'" + tElement + "', ";
    }
    auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
    return tSubMsg;
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
    const Teuchos::ParameterList & aParamList
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
    const Teuchos::ParameterList & aParamList
  )
  {
    if( !aParamList.isType<Plato::Scalar>("Sides") ){
      ANALYZE_THROWERR( std::string("ERROR: Input argument ('Side') is not defined, ") + 
        "side set for Nitsche's method cannot be determined" )
    }
    mSideSetName = aParamList.get<std::string>("Sides");
    
    if( !aParamList.isType<Plato::Scalar>("Material Model") ){
      ANALYZE_THROWERR( std::string("ERROR: Input argument ('Material Model') is not defined, ") + 
        "material constitutive model for Nitsche's method cannot be determined" )
    }
    mMaterialName = aParamList.get<std::string>("Material Model");
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
    const Plato::MaterialModel<EvaluationType> & aMaterialModel
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
    for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
    {
      tFirstStrainInvariant += aStrainTensor(tDim,tDim);
    }
    // add contribution from first stress invariant to the stress tensor
    for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++)
    {
      aStressTensor(tDim,tDim) += mLambda * tFirstStrainInvariant;
    }
    // add shear stress contribution to the stress tensor
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
    {
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
      {
        aStressTensor(tDimI,tDimJ) += 2.0 * mMu * aStrainTensor(tDimI,tDimJ);
      }
    }
  }
};

template<typename EvaluationType>
class NitscheStressEvaluator : public Plato::NitscheEvaluator
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
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;

public:
  NitscheStressEvaluator(
    const Teuchos::ParameterList& aParamList,
    const Teuchos::ParameterList& aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // create material constitutive model
    //
    Plato::FactoryElasticMaterial<EvaluationType> tFactory(aParamList);
    mMaterialModel = tFactory.create(mMaterialName);
  }
  ~NitscheStressEvaluator()
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
    // get side set connectivity information
    //
    auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
    auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    // create local functors
    //
    Plato::ComputeStrainTensor<EvaluationType> tComputeStrainTensor;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
    Plato::ComputeIsotropicElasticStressTensor<EvaluationType> tComputeStressTensor(mMaterialModel.operator*());
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnBodyParentElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnBodyParentElemSurface = BodyElementBase::getFaceCubWeights();
    // evaluate integral
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Kokkos::parallel_for("nitsche stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      auto tCubPointInFaceParentElem = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsInFaceParentElem = FaceElementBase::basisGrads(tCubPointInFaceParentElem);
      // quadrature data to evaluate integral on the body surface of interest
      Plato::Array<mNumSpatialDims> tCubPointOnBodyParentElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnBodyParentElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnBodyParentElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      auto tCubWeightOnBodyParentElemSurface = tCubWeightsOnBodyParentElemSurface(aPointOrdinal);
      auto tBasisValuesOnBodyParentElemSurface = BodyElementBase::basisValues(tCubPointOnBodyParentElemSurface);
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute surface area weighted normal vector
      //
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Array<mNumSpatialDims, ConfigScalarType> tWeightedNormalVector;
      tComputeWeightedNormalVector(
        tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsInFaceParentElem,tConfigWS,tWeightedNormalVector
      );
      // compute strains and stresses for this quadrature point
      ConfigScalarType tVolume(0.0);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims, ConfigScalarType> tGradient;
      tComputeGradient(tCellOrdinal,tCubPointOnBodyParentElemSurface,tConfigWS,tGradient,tVolume);
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims, StrainScalarType>  tStrainTensor(0.0);
      tComputeStrainTensor(tCellOrdinal,tStateWS, tGradient, tStrainTensor);
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ResultScalarType>  tStressTensor(0.0);
      tComputeStressTensor(tStrainTensor,tStressTensor);
      // term 1: int_{\Gamma_D} \delta{u}\cdot(\sigma\cdot{n}) d\Gamma_D
      for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++){
        for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++){
          auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
          for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++){
            ResultScalarType tValue = -aScale * tBasisValuesOnBodyParentElemSurface(tNode)
              * ( tStressTensor(tDimI,tDimJ) * tWeightedNormalVector[tDimJ] ) * tCubWeightOnBodyParentElemSurface;
            Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
          }
        }
      }
    });
  }
};

template<typename EvaluationType>
class NitscheVirtualStressEvaluator : public Plato::NitscheEvaluator
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
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;

public:
  NitscheVirtualStressEvaluator(
    const Teuchos::ParameterList& aParamList,
    const Teuchos::ParameterList& aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // create material constitutive model
    //
    Plato::FactoryElasticMaterial<EvaluationType> tFactory(aParamList);
    mMaterialModel = tFactory.create(mMaterialName);
  }
  ~NitscheVirtualStressEvaluator()
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
    Plato::ComputeStrainTensor<EvaluationType> tComputeStrainTensor;
    Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
    Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
    Plato::ComputeIsotropicElasticStressTensor<EvaluationType> tComputeStressTensor(mMaterialModel.operator*());
    // get integration points and weights
    //
    auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
    auto tCubPointsOnBodyParentElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnBodyParentElemSurface = BodyElementBase::getFaceCubWeights();
    // evaluate integral
    //
    Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
    Kokkos::parallel_for("nitsche stress evaluator", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
      KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      auto tCubPointInFaceParentElem = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsInFaceParentElem = FaceElementBase::basisGrads(tCubPointInFaceParentElem);
      // quadrature data to evaluate integral on the body surface of interest
      //
      Plato::Array<mNumSpatialDims> tCubPointOnBodyParentElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnBodyParentElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnBodyParentElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      auto tCubWeightOnBodyParentElemSurface = tCubWeightsOnBodyParentElemSurface(aPointOrdinal);
      auto tBasisValuesOnBodyParentElemSurface = BodyElementBase::basisValues(tCubPointOnBodyParentElemSurface);
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute surface area weighted normal vector
      //
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      Plato::Array<mNumSpatialDims, ConfigScalarType> tWeightedNormalVector;
      tComputeWeightedNormalVector(
        tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsInFaceParentElem,tConfigWS,tWeightedNormalVector
      );
      // project state from nodes to quadrature/cubature point
      //
      Plato::Array<mNumSpatialDims,StateScalarType> tProjectedStates;
      for(Plato::OrdinalType tDof = 0; tDof < mNumDofsPerNode; tDof++){
        tProjectedStates(tDof) = 0.0;
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++){
          Plato::OrdinalType tCellDofIndex = (mNumDofsPerNode * tNodeIndex) + tDof;
          tProjectedStates(tDof) += tStateWS(tCellOrdinal, tCellDofIndex) * 
            tBasisValuesOnBodyParentElemSurface(tNodeIndex);
        }
      }
      // evaluate int_{\Gamma_D} \delta(\sigma\cdot{n})\cdot(u - u_D) d\Gamma_D
      //
      ConfigScalarType tVolume(0.0);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims, ConfigScalarType> tGradient;
      tComputeGradient(tCellOrdinal,tCubPointOnBodyParentElemSurface,tConfigWS,tGradient,tVolume);
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ConfigScalarType>  tVirtualStrainTensor(0.0);
      tComputeStrainTensor(tCellOrdinal,tGradient,tVirtualStrainTensor);
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ConfigScalarType>  tVirtualStressTensor(0.0);
      tComputeStressTensor(tVirtualStrainTensor,tVirtualStressTensor);
      for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++)
      {
        for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
        {
          auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
          for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++)
          {
            ResultScalarType tValue = aScale * tCubWeightOnBodyParentElemSurface
              * ( tVirtualStressTensor(tDimI,tDimJ) * tWeightedNormalVector[tDimJ] )
              * ( tProjectedStates[tDimI] - tDirichletWS(tCellOrdinal,tLocalDofOrdinal) );
            Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
          }
        }
      }
    });
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
    const Plato::SpatialModel                  & aSpatialModel,
    const Plato::WorkSets                      & aWorkSets,
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
  ComputeSideCellFaceAreas(const std::string& aEntitySetName) :
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
  ComputeCharacteristicLength(const std::string& aEntitySetName) :
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
    const Teuchos::ParameterList& aParamList,
    const Teuchos::ParameterList& aNitscheParams
  ) : 
    BaseClassType(aNitscheParams)
  {
    // create material constitutive model
    //
    Plato::FactoryElasticMaterial<EvaluationType> tFactory(aParamList);
    mMaterialModel = tFactory.create(mMaterialName);
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
    auto tCubPointsOnBodyParentElemSurfaces = BodyElementBase::getFaceCubPoints();
    auto tCubWeightsOnBodyParentElemSurface = BodyElementBase::getFaceCubWeights();
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
      auto tCubPointInFaceParentElem = tCubPointsOnParentFaceElem(aPointOrdinal);
      auto tBasisGradsInFaceParentElem = FaceElementBase::basisGrads(tCubPointInFaceParentElem);
      // quadrature data to evaluate integral on the body surface of interest
      Plato::Array<mNumSpatialDims> tCubPointOnBodyParentElemSurface;
      Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
      auto tCubPointsOnBodyParentElemSurface = tCubPointsOnBodyParentElemSurfaces(tLocalFaceOrdinal);
      for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
        Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
        tCubPointOnBodyParentElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
      }
      auto tCubWeightOnBodyParentElemSurface = tCubWeightsOnBodyParentElemSurface(aPointOrdinal);
      auto tBasisValuesOnBodyParentElemSurface = BodyElementBase::basisValues(tCubPointOnBodyParentElemSurface);
      Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
      for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
        tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
      }
      // compute surface area
      //
      ConfigScalarType tSurfaceArea(0.0);
      auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
      tComputeSurfaceArea(tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsInFaceParentElem,tConfigWS,tSurfaceArea);
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
            tBasisValuesOnBodyParentElemSurface(tNodeIndex);
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
          ResultScalarType tValue = aScale * tGamma * tBasisValuesOnBodyParentElemSurface(tNode)
            * ( tProjectedStates[tDimI] - tDirichletWS(tCellOrdinal,tLocalDofOrdinal) ) 
            * tSurfaceArea * tCubWeightOnBodyParentElemSurface;
          Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
        }
      }
    });
  }
};

/// @class NitscheBC
///
/// @brief enforce the diriechlet boundary conditions in linear mechanical problems using nitsche's method:
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
/*
template<typename EvaluationType>
class NitscheBC : public NitscheBase<EvaluationType>
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
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ResultScalarType = typename EvaluationType::ResultScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<BodyElementBase,StateScalarType,ConfigScalarType>;
  /// @brief penalty for nitsche's method
  Plato::Scalar mNitschePenalty = 1.0
  /// @brief side set name
  const std::string mSideSetName;
  /// @brief name assigned to the material model applied on this side set
  const std::string mMaterialName;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterialModel;

public:
    NitscheBC(
      const Teuchos::ParameterList& aParamList,
      const Teuchos::ParameterList& aNitscheParams
    )
    {
      // create material constitutive model
      //
      Plato::FactoryElasticMaterial<EvaluationType> tFactory(aParamList);
      mMaterialModel = tFactory.create(mMaterialName);
      // parse penalty parameter
      //
      if(aSubList.isType<Plato::Scalar>("Penalty")){
        mNitschePenalty = aSubList.get<Plato::Scalar>("Penalty");
      }

      if( !aParamList.isType<Plato::Scalar>("Sides") ){
        ANALYZE_THROWERR( std::string("ERROR: Input argument ('Side') is not defined, ") + 
          "side set for Nitsche's method cannot be determined" )
      }
      mSideSetName = aParamList.get<std::string>("Sides");
      
      if( !aParamList.isType<Plato::Scalar>("Material Model") ){
        ANALYZE_THROWERR( std::string("ERROR: Input argument ('Material Model') is not defined, ") + 
          "material constitutive model for Nitsche's method cannot be determined" )
      }
      mMaterialName = aParamList.get<std::string>("Material Model");
    }
    ~NitscheBC()
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
      Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
      // compute characteristic length
      //
      Plato::ComputeCharacteristicLength<EvaluationType> tComputeCharacteristicLength(mSideSetName);
      Plato::ScalarVectorT<ConfigScalarType> tCharacteristicLength("characteristic length",tNumCellsOnSideSet);
      tComputeCharacteristicLength(aSpatialModel, aWorkSets, tCharacteristicLength);
      // create element strain and stress tensor evaluators 
      //
      Plato::ComputeStrainTensor<EvaluationType> tComputeStrainTensor;
      Plato::ComputeIsotropicElasticStressTensor<EvaluationType> tComputeStressTensor(mMaterial.operator*());
      Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
      // create compute surface area and normal functors
      //
      Plato::SurfaceArea<BodyElementBase> tComputeFaceArea;
      Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;      
      // get integration points and weights
      //
      auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
      auto tCubPointsOnBodyParentElemSurfaces = BodyElementBase::getFaceCubPoints();
      auto tCubWeightsOnBodyParentElemSurface = BodyElementBase::getFaceCubWeights();
      // evaluate nitsche's boundary conditions
      //
      auto tYoungsModulus  = mMaterialModel->getScalarConstant("youngs modulus");
      auto tNitschePenaltyTimesModulus = mNitschePenalty * tYoungsModulus;
      Kokkos::parallel_for("nitsche bcs", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
        KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
      {
        auto tCubPointInFaceParentElem = tCubPointsOnParentFaceElem(aPointOrdinal);
        auto tBasisGradsInFaceParentElem = FaceElementBase::basisGrads(tCubPointInFaceParentElem);
        // quadrature data to evaluate integral on the body surface of interest
        Plato::Array<mNumSpatialDims> tCubPointOnBodyParentElemSurface;
        Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
        auto tCubPointsOnBodyParentElemSurface = tCubPointsOnBodyParentElemSurfaces(tLocalFaceOrdinal);
        for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
          Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
          tCubPointOnBodyParentElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
        }
        auto tCubWeightOnBodyParentElemSurface = tCubWeightsOnBodyParentElemSurface(aPointOrdinal);
        auto tBasisValuesOnBodyParentElemSurface = BodyElementBase::basisValues(tCubPointOnBodyParentElemSurface);
        Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
        for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
          tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
        }
        // TERM 1
        //
        // compute normal vector weighted by the entity area
        auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
        Plato::Array<mNumSpatialDims, ConfigScalarType> tWeightedNormalVector;
        tComputeWeightedNormalVector(
          tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsInFaceParentElem,tConfigWS,tWeightedNormalVector
        );
        // compute entity area
        ConfigScalarType tFaceArea(0.0);
        tComputeFaceArea(tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsInFaceParentElem,tConfigWS,tFaceArea);
        // compute strains and stresses for this quadrature point
        ConfigScalarType tVolume(0.0);
        Plato::Matrix<mNumNodesPerCell,mNumSpatialDims, ConfigScalarType> tGradient;
        tComputeGradient(tCellOrdinal,tCubPointOnBodyParentElemSurface,tConfigWS,tGradient,tVolume);
        Plato::Matrix<mNumSpatialDims,mNumSpatialDims, StrainScalarType>  tStrainTensor(0.0);
        tComputeStrainTensor(tCellOrdinal,tStateWS, tGradient, tStrainTensor);
        Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ResultScalarType>  tStressTensor(0.0);
        tComputeStressTensor(tStrainTensor,tStressTensor);
        // term 1: int_{\Gamma_D} \delta{u}\cdot(\sigma\cdot{n}) d\Gamma_D
        for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++){
          for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++){
            auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
            for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++){
              ResultScalarType tValue = -aMultiplier * tBasisValuesOnBodyParentElemSurface(tNode)
                  * ( tStressTensor(tDimI,tDimJ) * tWeightedNormalVector[tDimJ] ) 
                  * tCubWeightOnBodyParentElemSurface * tFaceArea;
              Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
            }
          }
        }
        // TERM 2
        //
        // project state from nodes to quadrature/cubature point
        Plato::Array<mNumSpatialDims,StateScalarType> tProjectedStates;
        for(Plato::OrdinalType tDof = 0; tDof < mNumDofsPerNode; tDof++){
          tProjectedStates(tDof) = 0.0;
          for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++){
            Plato::OrdinalType tCellDofIndex = (mNumDofsPerNode * tNodeIndex) + tDof;
            tProjectedStates(tDof) += tStateWS(tCellOrdinal, tCellDofIndex) * tBasisValuesOnBodyParentElemSurface(tNodeIndex);
          }
        }
        Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ConfigScalarType>  tVirtualStrainTensor(0.0);
        tComputeStrainTensor(tCellOrdinal,tGradient,tVirtualStrainTensor);
        Plato::Matrix<mNumSpatialDims,mNumSpatialDims, ConfigScalarType>  tVirtualStressTensor(0.0);
        tComputeStressTensor(tVirtualStrainTensor,tVirtualStressTensor);
        // term 2: int_{\Gamma_D} \delta(\sigma\cdot{n})\cdot(u - u_D) d\Gamma_D
        for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++){
          for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++){
            auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
            for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++){
              ResultScalarType tValue = aMultiplier * tCubWeightOnBodyParentElemSurface
                  * tFaceArea * ( tVirtualStressTensor(tDimI,tDimJ) * tWeightedNormalVector[tDimJ] )
                  * ( tProjectedStates[tDimI] - tDirichletWS(tCellOrdinal,tLocalDofOrdinal) );
              Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
            }
          }
        }
        // TERM 3
        //   int_{\Gamma_D}\gamma_N^u \delta{u}\cdot(u - u_D) d\Gamma_D
        ConfigScalarType tGamma = tNitschePenaltyTimesModulus / tCharacteristicLength(aSideOrdinal);
        for(Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++){
          for(Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++){
            auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
            ResultScalarType tValue = aMultiplier * tGamma * tBasisValuesOnBodyParentElemSurface(tNode)
                * ( tProjectedStates[tDimI] - tDirichletWS(tCellOrdinal,tLocalDofOrdinal) ) * tFaceArea
                * tCubWeightOnBodyParentElemSurface;
            Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
          }
        }
      });
    }
};
*/

} // namespace Plato

namespace LinearMechanicsNitscheTest
{

TEUCHOS_UNIT_TEST( LinearMechanicsNitscheTest, MaterialIsotropicElastic )
{
}

} // namespace LinearMechanicsNitscheTest

