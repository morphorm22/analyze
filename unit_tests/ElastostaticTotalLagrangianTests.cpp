/*
 * ElastostaticTotalLagrangianTests.cpp
 *
 *  Created on: May 10, 2023
 */

// trilinos includes
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

// unit test includes
#include "util/PlatoTestHelpers.hpp"

// plato
#include "Tri3.hpp"
#include "BLAS1.hpp"
#include "WorksetBase.hpp"
#include "MaterialModel.hpp"
#include "GradientMatrix.hpp"
#include "MechanicsElement.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace mechanical
{

/// @enum property
/// @brief supported mechanical material property enums
enum struct property
{
  /// @brief Supported mechanical material property enums
  YOUNGS_MODULUS=0, 
  POISSON_RATIO=1, 
  MASS_DENSITY=2, 
  LAME_LAMBDA=3, 
  LAME_MU=4,
  TO_ERSATZ_MATERIAL_EXPONENT=5,
  TO_MIN_ERSATZ_MATERIAL_VALUE=6
};

/// @struct PropEnum
/// @brief interface between input mechanical material property string and supported mechanical material property enum
struct PropEnum
{
private:
  /// @brief map from input mechanical material property string to supported mechanical material property enum
  std::unordered_map<std::string,Plato::mechanical::property> s2e = {
      {"youngs modulus"  ,Plato::mechanical::property::YOUNGS_MODULUS},
      {"poissons ratio"  ,Plato::mechanical::property::POISSON_RATIO},
      {"density"         ,Plato::mechanical::property::MASS_DENSITY},
      {"lame lambda"     ,Plato::mechanical::property::LAME_LAMBDA},
      {"lame mu"         ,Plato::mechanical::property::LAME_MU},
      {"penalty exponent",Plato::mechanical::property::TO_ERSATZ_MATERIAL_EXPONENT},
      {"minimum value"   ,Plato::mechanical::property::TO_MIN_ERSATZ_MATERIAL_VALUE}    
  };

public:
  /// @brief Return mechanical property enum associated with input string, 
  ///   throw if requested mechanical property is not supported
  /// @param [in] aInput property identifier
  /// @return mechanical property enum
  Plato::mechanical::property 
  get(
    const std::string &aInput
  ) const
  {
    auto tLower = Plato::tolower(aInput);
    auto tItr = s2e.find(tLower);
    if( tItr == s2e.end() ){
        auto tMsg = this->getErrorMsg(tLower);
        ANALYZE_THROWERR(tMsg)
    }
    return tItr->second;
  }

private:
  std::string
  getErrorMsg(
    const std::string & aInProperty
  ) const
  {
    auto tMsg = std::string("Did not find matching enum for input mechanical property '") 
            + aInProperty + "'. Supported mechanical property keywords are: ";
    for(const auto& tPair : s2e)
    {
        tMsg = tMsg + "'" + tPair.first + "', ";
    }
    auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
    return tSubMsg;
  }
};

/// @enum material
/// @brief supported mechanical material enums
enum struct material
{
  /// @brief Supported mechanical material enums
  KIRCHHOFF=0,
  NEO_HOOKEAN=1,
};

/// @struct PropEnum
/// @brief interface between input mechanical material string and supported mechanical material enum
struct MaterialEnum
{
private:
  /// @brief map from input mechanical material string to supported mechanical material enum
  std::unordered_map<std::string,Plato::mechanical::material> s2e = {
    {"kirchhoff"  ,Plato::mechanical::material::KIRCHHOFF},
    {"neo-hookean",Plato::mechanical::material::NEO_HOOKEAN},
  };

public:
  /// @brief Return mechanical material enum associated with input string, 
  ///   throw if requested mechanical material is not supported
  /// @param [in] aInput mechanical material identifier
  /// @return mechanical material enum
  Plato::mechanical::material 
  get(
    const std::string &aInput
  ) const
  {
    auto tLower = Plato::tolower(aInput);
    auto tItr = s2e.find(tLower);
    if( tItr == s2e.end() ){
        auto tMsg = this->getErrorMsg(tLower);
        ANALYZE_THROWERR(tMsg)
    }
    return tItr->second;
  }

private:
  std::string
  getErrorMsg(
    const std::string & aInProperty
  ) const
  {
    auto tMsg = std::string("Did not find matching enum for input mechanical material '") 
            + aInProperty + "'. Supported mechanical material keywords are: ";
    for(const auto& tPair : s2e)
    {
        tMsg = tMsg + "'" + tPair.first + "', ";
    }
    auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
    return tSubMsg;
  }
};

}
// namespace mechanical

template<typename EvaluationType>
class StateGradient
{
private:
  using ElementType = typename EvaluationType::ElementType;

  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;

  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;

public:
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::OrdinalType                                               & aCellIndex,
    const Plato::ScalarMultiVectorT<StateScalarType>                       & aStates,
    const Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> & aGradient,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStateGrad
  ) const
  {
    for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
    {
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
      {
        Plato::OrdinalType tDof = (tNode * mNumDofsPerNode) + tDimI;
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
        {
          aStateGrad(tDimI,tDimJ) += aStates(aCellIndex,tDof) * aGradient(tNode,tDimJ);
        }
      }
    }
  }
};

template<typename EvaluationType>
class DeformationGradient
{
private:
  using ElementType = typename EvaluationType::ElementType;
  
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;

  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;

public:
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStateGrad,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefGrad
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        aDefGrad(tDimI,tDimJ) += aStateGrad(tDimI,tDimJ);
      }
      aDefGrad(tDimI,tDimI) += 1.0;
    }
  }
};

template<typename EvaluationType>
class RightDeformationTensor
{
private:
  using ElementType = typename EvaluationType::ElementType;
  
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;

  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;

public:
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefGradT,
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefGrad,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefTensor
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        for(Plato::OrdinalType tDimK = 0; tDimK < mNumSpatialDims; tDimK++){
          aDefTensor(tDimI,tDimJ) += aDefGradT(tDimI,tDimK) * aDefGrad(tDimK,tDimJ);
        }
      }
    }
  }
};

template<typename EvaluationType>
class GreenLagrangeStrainTensor
{
private:
  using ElementType = typename EvaluationType::ElementType;
  
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;

  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;

public:
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefTensor,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStrainTensor
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
    {
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
      {
        aStrainTensor(tDimI,tDimJ) += 0.5 * aDefTensor(tDimI,tDimJ);
      }
      aStrainTensor(tDimI,tDimI) -= 0.5;
    }
  }
};

/// @class MaterialKirchhoff
/// @brief material interface for a kirchhoff material model
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class MaterialKirchhoff : public Plato::MaterialModel<EvaluationType>
{
private:
    /// @brief topological element typename
    using ElementType = typename EvaluationType::ElementType;
    /// @brief number of spatial dimensions 
    static constexpr int mNumSpatialDims = ElementType::mNumSpatialDims;
    /// @brief map from input string to supported mechanical property
    Plato::mechanical::PropEnum mS2E;
    /// @brief map from mechanical property enum to list of property values in string format
    std::unordered_map<Plato::mechanical::property,std::vector<std::string>> mProperties;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName input material parameter list name
  /// @param [in] aParamList    input material parameter list
  MaterialKirchhoff(
      const std::string            & aMaterialName,
      const Teuchos::ParameterList & aParamList
  )
  {
    // set material input parameter list name
    this->name(aMaterialName);
    // parse youngs modulus
    this->parseScalar("Youngs Modulus", aParamList);
    auto tYoungsModulus = this->getScalarConstant("Youngs Modulus");
    mProperties[mS2E.get("Youngs Modulus")].push_back( std::to_string(tYoungsModulus) );
    // parse poissons ratio
    this->parseScalar("Poissons Ratio", aParamList);
    auto tPoissonsRatio = this->getScalarConstant("Poissons Ratio");
    mProperties[mS2E.get("Poissons Ratio")].push_back( std::to_string(tPoissonsRatio) );
    // compute Lame constants
    auto tMu = tYoungsModulus / ( 2.*(1.+tPoissonsRatio) );
    mProperties[mS2E.get("Lame Mu")].push_back( std::to_string(tMu) );
    auto tLambda = (tPoissonsRatio * tYoungsModulus) / ( (1.+tPoissonsRatio)*(1.-2.*tPoissonsRatio) );
    mProperties[mS2E.get("Lame Lambda")].push_back( std::to_string(tLambda) );
  }
  /// @brief class destructor
  ~MaterialKirchhoff(){}
  /// @fn property
  /// @brief return list of property values
  /// @param aPropertyID 
  /// @return standard vector of strings
  std::vector<std::string> 
  property(const std::string & aPropertyID)
  const 
  {
    auto tEnum = mS2E.get(aPropertyID);
    auto tItr = mProperties.find(tEnum);
    if( tItr == mProperties.end() ){
        return {};
    }
    return tItr->second;
  }
};

/// @class MaterialNeoHookean
/// @brief Material interface for a neo-hookean material model with stored energy potential:
///  \f[
///    \Psi(\mathbf{C})=\frac{1}{2}\lambda(\ln(J))^2 - \mu\ln(J) + \frac{1}{2}\mu(\mbox{trace}(\mathbf{C})-3),
///  \f]
/// where \f$\mathbf{C}\f$ is the right deformation tensor, \f$J=\det(\mathbf{F})\f$, \f$F\f$ 
/// is the deformation gradient, and \f$\lambda\f$ and \f$\mu\f$ are the Lame constants. The 
/// second Piola-Kirchhoff stress tensor is given by:
///  \f[
///    \mathbf{S}=\lambda\ln(J)\mathbf{C}^{-1} + \mu(\mathbf{I}-\mathbf{C}^{-1}),
///  \f]
/// where \f$\mathbf{I}\f$ is the second order identity tensor.
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class MaterialNeoHookean : public Plato::MaterialModel<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions 
  static constexpr int mNumSpatialDims = ElementType::mNumSpatialDims;
  /// @brief map from input string to supported mechanical property
  Plato::mechanical::PropEnum mS2E;
  /// @brief map from mechanical property enum to list of property values in string format
  std::unordered_map<Plato::mechanical::property,std::vector<std::string>> mProperties;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName input material parameter list name
  /// @param [in] aParamList    input material parameter list
  MaterialNeoHookean(
      const std::string            & aMaterialName,
      const Teuchos::ParameterList & aParamList
  )
  {
    // set material input parameter list name
    this->name(aMaterialName);
    // parse youngs modulus
    this->parseScalar("Youngs Modulus", aParamList);
    auto tYoungsModulus = this->getScalarConstant("Youngs Modulus");
    mProperties[mS2E.get("Youngs Modulus")].push_back( std::to_string(tYoungsModulus) );
    // parse poissons ratio
    this->parseScalar("Poissons Ratio", aParamList);
    auto tPoissonsRatio = this->getScalarConstant("Poissons Ratio");
    mProperties[mS2E.get("Poissons Ratio")].push_back( std::to_string(tPoissonsRatio) );
    // compute Lame constants
    auto tMu = tYoungsModulus / ( 2.*(1.+tPoissonsRatio) );
    mProperties[mS2E.get("Lame Mu")].push_back( std::to_string(tMu) );
    auto tLambda = (tPoissonsRatio * tYoungsModulus) / ( (1.+tPoissonsRatio)*(1.-2.*tPoissonsRatio) );
    mProperties[mS2E.get("Lame Lambda")].push_back( std::to_string(tLambda) );
  }

  /// @brief destructor
  ~MaterialNeoHookean(){}
  
  /// @fn property
  /// @brief return list of property values
  /// @param aPropertyID 
  /// @return standard vector of strings
  std::vector<std::string> 
  property(const std::string & aPropertyID)
  const 
  {
    auto tEnum = mS2E.get(aPropertyID);
    auto tItr = mProperties.find(tEnum);
    if( tItr == mProperties.end() ){
        return {};
    }
    return tItr->second;
  }
};

/// @class FactoryNonlinearMechanicalMaterial
/// @brief factroy for hyperelastic material models
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class FactoryNonlinearMechanicalMaterial
{
private:
    /// @brief const reference to input problem parameter list
    const Teuchos::ParameterList& mParamList;
    /// @brief supported mechanical materials interface
    Plato::mechanical::MaterialEnum mS2E;

public:
  /// @brief class constructor
  /// @param aParamList input problem parameters
  FactoryNonlinearMechanicalMaterial(
    Teuchos::ParameterList& aParamList
  ) :
    mParamList(aParamList)
  {}
  /// @brief class destructor
  ~FactoryNonlinearMechanicalMaterial()
  {}
  /// @brief create material model
  /// @param aMaterialName name of input material parameter list
  /// @return shared pointed
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> 
  create(std::string aMaterialName)
  {
    if (!mParamList.isSublist("Material Models"))
    {
      ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
    }
    else
    {
      auto tMaterialModelParamList = mParamList.get < Teuchos::ParameterList > ("Material Models");
      if (!tMaterialModelParamList.isSublist(aMaterialName))
      {
          auto tMsg = std::string("Requested a material model with name ('") + aMaterialName 
                      + "') that is not defined in the input deck";
          ANALYZE_THROWERR(tMsg);
      }
      auto tMaterialParamList = tMaterialModelParamList.sublist(aMaterialName);
      if(tMaterialParamList.isSublist("Kirchhoff")){
        auto tMaterial = std::make_shared<Plato::MaterialKirchhoff<EvaluationType>>
                          (aMaterialName, tMaterialParamList.sublist("Kirchhoff"));
        return tMaterial;
      }
      else
      if(tMaterialParamList.isSublist("Neo-Hookean")){
        auto tMaterial = std::make_shared<Plato::MaterialNeoHookean<EvaluationType>>
                          (aMaterialName, tMaterialParamList.sublist("Neo-Hookean"));
        return tMaterial;
      }
      else{
        mS2E.get("Not Supported"); // throws
        return nullptr;
      }
    }
  }
};

/// @class StressEvaluator
/// @brief base class for stress evaluators
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class StressEvaluator
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
  /// @brief contains mesh and model information
  const Plato::SpatialDomain & mSpatialDomain;
  /// @brief output database 
  Plato::DataMap & mDataMap;

public:
  /// @brief base class construtor
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output database
  explicit
  StressEvaluator(
      const Plato::SpatialDomain & aSpatialDomain,
            Plato::DataMap       & aDataMap
  ) :
    mSpatialDomain(aSpatialDomain),
    mDataMap(aDataMap)
  {}
  /// @brief base class destructor
  virtual ~StressEvaluator()
  {}

  /// @fn evaluate
  /// @brief pure virtual method: evaluates current density 
  /// @param aSpatialDomain contains meshed model information
  /// @param aState         state workset
  /// @param aControl       control workset
  /// @param aConfig        configuration workset
  /// @param aResult        result workset
  /// @param aScale         scalar 
  virtual 
  void 
  evaluate(
      const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
      const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
      const Plato::ScalarArray4DT<ResultScalarType>      & aResult
  ) const = 0;
};

/// @class StressEvaluatorKirchhoff
/// @brief Evaluate second Piola-Kirchhoff stress tensor for a Kirchhoff material:
///  \f[
///    \mathbf{S}=\lambda\mbox{trace}(\mathbf{E})\mathbf{I} + 2\mu\mathbf{E},
///  \f]
/// where \f$\lambda\f$ and \f$\mu\f$ are the Lame constants, \f$\mathbf{E}\f$ is 
/// the Green-Lagrange strain tensor, and \f$\mathbf{I}\f$ is the second order 
/// identity tensor.
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class StressEvaluatorKirchhoff : public Plato::StressEvaluator<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types for an evaluation type
  using StateT   = typename EvaluationType::StateScalarType;
  using ConfigT  = typename EvaluationType::ConfigScalarType;
  using ResultT  = typename EvaluationType::ResultScalarType;
  using ControlT = typename EvaluationType::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  /// @brief local typename for base class
  using BaseClassType = Plato::StressEvaluator<EvaluationType>;
  /// @brief contains mesh and model information
  using BaseClassType::mSpatialDomain;
  /// @brief output database 
  using BaseClassType::mDataMap;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterial;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName  name of material parameter list
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aSpatialDomain contains mesh and model information 
  /// @param [in] aDataMap       output database
  StressEvaluatorKirchhoff(
    const std::string            & aMaterialName,
          Teuchos::ParameterList & aParamList,
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap) : 
    Plato::StressEvaluator<EvaluationType>(aSpatialDomain,aDataMap)
  {
    Plato::FactoryNonlinearMechanicalMaterial<EvaluationType> tFactory(aParamList);
    mMaterial = tFactory.create(aMaterialName);
  }

  /// @brief evaluate stress tensor
  /// @param aState   2D state workset
  /// @param aControl 2D control workset
  /// @param aConfig  3D configuration workset
  /// @param aResult  4D result workset
  void 
  evaluate(
      const Plato::ScalarMultiVectorT<StateT>   & aState,
      const Plato::ScalarMultiVectorT<ControlT> & aControl,
      const Plato::ScalarArray3DT<ConfigT>      & aConfig,
      const Plato::ScalarArray4DT<ResultT>      & aResult
  ) const
  {
    // get lame constants
    Plato::Scalar tMu     = std::stod(mMaterial->property("lame mu").front());
    Plato::Scalar tLambda = std::stod(mMaterial->property("lame lambda").front());
    // get integration rule information
    auto tCubPoints = ElementType::getCubPoints();
    auto tNumPoints = ElementType::mNumGaussPoints;
    // compute state gradient
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
    Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
    Plato::RightDeformationTensor<EvaluationType> tComputeRightDeformationTensor;
    Plato::GreenLagrangeStrainTensor<EvaluationType> tGreenLagrangeStrainTensor;
    // evaluate stress tensor
    auto tNumCells = mSpatialDomain.numCells();
    Kokkos::parallel_for("compute state gradient", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        // compute gradient functions
        ConfigT tVolume(0.0);
        Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
        auto tCubPoint = tCubPoints(iGpOrdinal);
        tComputeGradient(iCellOrdinal,tCubPoint,aConfig,tGradient,tVolume);
        // compute state gradient
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tStateGrad(StrainT(0.));
        tComputeStateGradient(iCellOrdinal,aState,tGradient,tStateGrad);
        // compute deformation gradient 
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tDefGrad(StrainT(0.));
        tComputeDeformationGradient(tStateGrad,tDefGrad);
        // apply transpose to deformation gradient
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tDefGradT = 
          Plato::transpose(tDefGrad);
        // compute cauchy-green deformation tensor
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tDefTensor(StrainT(0.));
        tComputeRightDeformationTensor(tDefGradT,tDefGrad,tDefTensor);
        // compute green-lagrange strain tensor
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tStrainTensor(StrainT(0.));
        tGreenLagrangeStrainTensor(tDefTensor,tStrainTensor);
        // compute stress
        StrainT tTrace = Plato::trace(tStrainTensor);
        for(Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
          aResult(iCellOrdinal,iGpOrdinal,tDimI,tDimI) += tLambda*tTrace;
          for(Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
            aResult(iCellOrdinal,iGpOrdinal,tDimI,tDimJ) += 2.0*tMu*tStrainTensor(tDimI,tDimJ);
          }
        }
    });
  }
};

/// @class StressEvaluatorNeoHookean
/// @brief Evaluate second Piola-Kirchhoff stress tensor for a Neo-Hookean material: 
///  \f[
///    \mathbf{S}=\lambda\ln(J)\mathbf{C}^{-1} + \mu(\mathbf{I}-\mathbf{C}^{-1}),
///  \f]
/// where \f$\lambda\f$ and \f$\mu\f$ are the Lame constants, \f$J=\det(\mathbf{F})\f$, 
/// \f$\mathbf{F}\f is the deformation gradient, \f$\mathbf{C}\f$ is the right deformation 
/// tensor, and \f$\mathbf{I}\f$ is the second order identity tensor.
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class StressEvaluatorNeoHookean : public Plato::StressEvaluator<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types for an evaluation type
  using StateT   = typename EvaluationType::StateScalarType;
  using ConfigT  = typename EvaluationType::ConfigScalarType;
  using ResultT  = typename EvaluationType::ResultScalarType;
  using ControlT = typename EvaluationType::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  /// @brief local typename for base class
  using BaseClassType = Plato::StressEvaluator<EvaluationType>;
  /// @brief contains mesh and model information
  using BaseClassType::mSpatialDomain;
  /// @brief output database 
  using BaseClassType::mDataMap;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterial;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName  name of material parameter list
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aSpatialDomain contains mesh and model information 
  /// @param [in] aDataMap       output database
  StressEvaluatorNeoHookean(
    const std::string            & aMaterialName,
          Teuchos::ParameterList & aParamList,
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap) : 
    Plato::StressEvaluator<EvaluationType>(aSpatialDomain,aDataMap)
  {
    Plato::FactoryNonlinearMechanicalMaterial<EvaluationType> tFactory(aParamList);
    mMaterial = tFactory.create(aMaterialName);
  }

  /// @brief class destructor
  ~StressEvaluatorNeoHookean(){}

  /// @brief evaluate stress tensor
  /// @param aState   2D state workset
  /// @param aControl 2D control workset
  /// @param aConfig  3D configuration workset
  /// @param aResult  4D result workset
  void 
  evaluate(
      const Plato::ScalarMultiVectorT<StateT>   & aState,
      const Plato::ScalarMultiVectorT<ControlT> & aControl,
      const Plato::ScalarArray3DT<ConfigT>      & aConfig,
      const Plato::ScalarArray4DT<ResultT>      & aResult
  ) const
  {
    // get lame constants
    Plato::Scalar tMu     = std::stod(mMaterial->property("lame mu").front());
    Plato::Scalar tLambda = std::stod(mMaterial->property("lame lambda").front());
    // get integration rule information
    auto tCubPoints = ElementType::getCubPoints();
    auto tNumPoints = ElementType::mNumGaussPoints;
    // compute state gradient
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
    Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
    Plato::RightDeformationTensor<EvaluationType> tComputeRightDeformationTensor;
    // evaluate stress tensor
    auto tNumCells = mSpatialDomain.numCells();
    Kokkos::parallel_for("compute state gradient", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        // compute gradient functions
        ConfigT tVolume(0.0);
        Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
        auto tCubPoint = tCubPoints(iGpOrdinal);
        tComputeGradient(iCellOrdinal,tCubPoint,aConfig,tGradient,tVolume);
        // compute state gradient
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tStateGrad(StrainT(0.));
        tComputeStateGradient(iCellOrdinal,aState,tGradient,tStateGrad);
        // compute deformation gradient 
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tDefGrad(StrainT(0.));
        tComputeDeformationGradient(tStateGrad,tDefGrad);
        // apply transpose to deformation gradient
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tDefGradT = 
          Plato::transpose(tDefGrad);
        // compute cauchy-green deformation tensor
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
          tRightDeformationTensor(StrainT(0.));
        tComputeRightDeformationTensor(tDefGradT,tDefGrad,tRightDeformationTensor);
        // compute determinant of deformation gradient
        StrainT tDetDefGrad = Plato::determinant(tDefGrad);
        // invert right deformation tensor
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
          tInverseRightDeformationTensor = Plato::invert(tRightDeformationTensor);
        // compute stress tensor
        for(Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
          aResult(iCellOrdinal,iGpOrdinal,tDimI,tDimI) += tMu;
          for(Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
            aResult(iCellOrdinal,iGpOrdinal,tDimI,tDimJ) += 
              tLambda * log(tDetDefGrad) * tInverseRightDeformationTensor(tDimI,tDimJ);
            aResult(iCellOrdinal,iGpOrdinal,tDimI,tDimJ) -= 
              tMu * tInverseRightDeformationTensor(tDimI,tDimJ);
          }
        }
    });
  }
};

/// @class FactoryStressEvaluator
/// @brief Factory of mechanical stress evaluators
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class FactoryStressEvaluator
{
private:
  /// @brief name of input material parameter list
  std::string mMaterialName; 
  /// @brief supported mechanical materials interface
  Plato::mechanical::MaterialEnum mS2E;

public:
  /// @brief class constructor
  /// @param aParamList input problem parameters
  FactoryStressEvaluator(
    const std::string & aMaterialName
  ) :
    mMaterialName(aMaterialName)
  {}

  /// @brief class destructor
  ~FactoryStressEvaluator(){}

  /// @brief create stress evaluator
  /// @param aMaterialName  name of input material parameter list
  /// @param aSpatialDomain contains mesh and model information
  /// @param aDataMap       output database
  /// @return shared pointer
  std::shared_ptr<Plato::StressEvaluator<EvaluationType>> 
  create(
          Teuchos::ParameterList & aParamList,
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap
  )
  {
    if (!aParamList.isSublist("Material Models"))
    {
      ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
    }
    else
    {
      auto tMaterialModelParamList = aParamList.get<Teuchos::ParameterList>("Material Models");
      if (!tMaterialModelParamList.isSublist(mMaterialName))
      {
          auto tMsg = std::string("Requested a material model with name ('") + mMaterialName 
            + "') that is not defined in the input deck";
          ANALYZE_THROWERR(tMsg);
      }
      auto tMaterialParamList = tMaterialModelParamList.sublist(mMaterialName);
      if(tMaterialParamList.isSublist("Kirchhoff")){
        auto tStressEvaluator = std::make_shared<Plato::StressEvaluatorKirchhoff<EvaluationType>>
                                (mMaterialName,aParamList,aSpatialDomain,aDataMap);
        return tStressEvaluator;
      }
      else
      if(tMaterialParamList.isSublist("Neo-Hookean")){
        auto tStressEvaluator = std::make_shared<Plato::StressEvaluatorNeoHookean<EvaluationType>>
                                (mMaterialName,aParamList,aSpatialDomain,aDataMap);
        return tStressEvaluator;
      }
      else{
        mS2E.get("Not Supported"); // throws
        return nullptr;
      }
    }
  }
};

template<typename EvaluationType>
class ResidualElastostaticTotalLagrangian
{

};

}

namespace ElastostaticTotalLagrangianTests
{

Teuchos::RCP<Teuchos::ParameterList> tGenericParamList = Teuchos::getParametersFromXmlString(
"<ParameterList name='Plato Problem'>                                                                  \n"
  "<ParameterList name='Spatial Model'>                                                                \n"
    "<ParameterList name='Domains'>                                                                    \n"
      "<ParameterList name='Design Volume'>                                                            \n"
        "<Parameter name='Element Block' type='string' value='body'/>                                  \n"
        "<Parameter name='Material Model' type='string' value='Mystic'/>                               \n"
      "</ParameterList>                                                                                \n"
    "</ParameterList>                                                                                  \n"
  "</ParameterList>                                                                                    \n"
  "<ParameterList name='Material Models'>                                                              \n"
    "<ParameterList name='Mystic'>                                                                     \n"
      "<ParameterList name='Kirchhoff'>                                                                \n"
        "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>                                \n"
        "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>                                 \n"
      "</ParameterList>                                                                                \n"
    "</ParameterList>                                                                                  \n"
  "</ParameterList>                                                                                    \n"
"</ParameterList>                                                                                      \n"
);

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, tComputeStateGradient )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute state gradient
  Plato::StateGradient<Residual> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Kokkos::parallel_for("compute state gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStateGrad(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGrad);
      // copy results
      for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
        for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
          tResultsWS(iCellOrdinal,tDimI,tDimJ) = tStateGrad(tDimI,tDimJ);
        }
      }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.4,0.2,0.4,0.2}, {0.4,0.2,0.4,0.2} };
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

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, DeformationGradient )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute state gradient
  Plato::StateGradient<Residual> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::DeformationGradient<Residual> tComputeDeformationGradient;
  Kokkos::parallel_for("compute state gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStateGrad(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGrad);
      // compute deformation gradient 
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefGrad(StrainT(0.));
      tComputeDeformationGradient(tStateGrad,tDefGrad);
      // copy result
      for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
        for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
          tResultsWS(iCellOrdinal,tDimI,tDimJ) = tDefGrad(tDimI,tDimJ);
        }
      }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.4,0.2,0.4,1.2}, {1.4,0.2,0.4,1.2} };
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

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, RightDeformationTensor )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute state gradient
  Plato::StateGradient<Residual> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::DeformationGradient<Residual> tComputeDeformationGradient;
  Plato::RightDeformationTensor<Residual> tComputeRightDeformationTensor;
  Kokkos::parallel_for("compute state gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStateGrad(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGrad);
      // compute deformation gradient 
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefGrad(StrainT(0.));
      tComputeDeformationGradient(tStateGrad,tDefGrad);
      // apply transpose to deformation gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefGradT
        = Plato::transpose(tDefGrad);
      // compute cauchy-green deformation tensor
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefTensor(StrainT(0.));
      tComputeRightDeformationTensor(tDefGradT,tDefGrad,tDefTensor);
      // copy result
      for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
        for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
          tResultsWS(iCellOrdinal,tDimI,tDimJ) = tDefTensor(tDimI,tDimJ);
        }
      }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {2.12,0.76,0.76,1.48}, {2.12,0.76,0.76,1.48} };
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

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, GreenLagrangeStrainTensor )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute state gradient
  Plato::StateGradient<Residual> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::DeformationGradient<Residual> tComputeDeformationGradient;
  Plato::RightDeformationTensor<Residual> tComputeRightDeformationTensor;
  Plato::GreenLagrangeStrainTensor<Residual> tGreenLagrangeStrainTensor;
  Kokkos::parallel_for("compute state gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStateGrad(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGrad);
      // compute deformation gradient 
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefGrad(StrainT(0.));
      tComputeDeformationGradient(tStateGrad,tDefGrad);
      // apply transpose to deformation gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefGradT
        = Plato::transpose(tDefGrad);
      // compute cauchy-green deformation tensor
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefTensor(StrainT(0.));
      tComputeRightDeformationTensor(tDefGradT,tDefGrad,tDefTensor);
      // compute green-lagrange strain tensor
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStrainTensor(StrainT(0.));
      tGreenLagrangeStrainTensor(tDefTensor,tStrainTensor);
      // copy result
      for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
        for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
          tResultsWS(iCellOrdinal,tDimI,tDimJ) = tStrainTensor(tDimI,tDimJ);
        }
      }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.56,0.38,0.38,0.24}, {0.56,0.38,0.38,0.24} };
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

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, StressEvaluatorKirchhoffTensor )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  // create output database and spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(1.0, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  // results workset
  Plato::OrdinalType tNumGaussPoints = ElementType::mNumGaussPoints;
  Plato::ScalarArray4DT<ResultT> tResultsWS("stress",tNumCells,tNumGaussPoints,tSpaceDim,tSpaceDim);
  Plato::StressEvaluatorKirchhoff<Residual> tStressEvaluator("Mystic",*tGenericParamList,tOnlyDomainDefined,tDataMap);
  tStressEvaluator.evaluate(tStateWS,tControlWS,tConfigWS,tResultsWS);
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.106172,0.281481,0.281481,0.869135}, {1.106172,0.281481,0.281481,0.869135} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,0,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, StressEvaluatorNeoHookeanTensor )
{
  // create parameter list
  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                                                  \n"
    "<ParameterList name='Spatial Model'>                                                                \n"
      "<ParameterList name='Domains'>                                                                    \n"
        "<ParameterList name='Design Volume'>                                                            \n"
          "<Parameter name='Element Block' type='string' value='body'/>                                  \n"
          "<Parameter name='Material Model' type='string' value='Mystic'/>                               \n"
        "</ParameterList>                                                                                \n"
      "</ParameterList>                                                                                  \n"
    "</ParameterList>                                                                                    \n"
    "<ParameterList name='Material Models'>                                                              \n"
      "<ParameterList name='Mystic'>                                                                     \n"
        "<ParameterList name='Neo-Hookean'>                                                              \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>                                \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>                                 \n"
        "</ParameterList>                                                                                \n"
      "</ParameterList>                                                                                  \n"
    "</ParameterList>                                                                                    \n"
  "</ParameterList>                                                                                      \n"
  );
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  // create output database and spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(1.0, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  // results workset
  Plato::OrdinalType tNumGaussPoints = ElementType::mNumGaussPoints;
  Plato::ScalarArray4DT<ResultT> tResultsWS("stress",tNumCells,tNumGaussPoints,tSpaceDim,tSpaceDim);
  Plato::StressEvaluatorNeoHookean<Residual> 
    tStressEvaluator("Mystic",*tParamList,tOnlyDomainDefined,tDataMap);
  tStressEvaluator.evaluate(tStateWS,tControlWS,tConfigWS,tResultsWS);
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.39107049,-0.01062979,-0.01062979,0.40002189}, {0.39107049,-0.01062979,-0.01062979,0.40002189} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,0,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

}